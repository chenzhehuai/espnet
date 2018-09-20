#!/usr/bin/env python

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import copy
import json
import logging
import math
import os

# torch related
import torch

# espnet related
from asr_utils import adadelta_eps_decay
from asr_utils import add_results_to_json
from asr_utils import CompareValueTrigger
from asr_utils import get_model_conf
from asr_utils import load_inputs_and_targets
from asr_utils import make_batchset
from asr_utils import PlotAttentionReport
from asr_utils import restore_snapshot
from asr_utils import torch_load
from asr_utils import torch_resume
from asr_utils import torch_save
from asr_utils import torch_snapshot
from e2e_asr_th import E2E
from e2e_asr_th import Loss
from e2e_asr_th import pad_list

from results import EpochResult, GlobalResult

# for kaldi io
import kaldi_io_py

# rnnlm
import extlm_pytorch
import lm_pytorch

# matplotlib related
import matplotlib
import numpy as np
matplotlib.use('Agg')

REPORT_INTERVAL = 100

class open_kaldi_feat:
    def __enter__(self, batch):
        yield load_inputs_and_targets(batch)
        




class CustomConverter(object):
    """CUSTOM CONVERTER"""

    def __init__(self, subsamping_factor=1):
        self.subsamping_factor = subsamping_factor
        self.ignore_id = -1

    def transform(self, item):
        return load_inputs_and_targets(item)

    def __call__(self, batch, device):
        # batch should be located in list
        assert len(batch) == 1
        xs, ys = batch[0]

        # perform subsamping
        if self.subsamping_factor > 1:
            xs = [x[::self.subsampling_factor, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.shape[0] for x in xs])

        # perform padding and convert to tensor
        xs_pad = pad_list([torch.from_numpy(x).float() for x in xs], 0).to(device)
        ilens = torch.from_numpy(ilens).to(device)
        ys_pad = pad_list([torch.from_numpy(y).long() for y in ys], self.ignore_id).to(device)

        return xs_pad, ilens, ys_pad


def train(args):
    '''Run training'''
    # seed setting
    torch.manual_seed(args.seed)

    # debug mode setting
    # 0 would be fastest, but 1 seems to be reasonable
    # by considering reproducability
    # revmoe type check
    if args.debugmode < 2:
        logging.info('torch type check is disabled')
    # use determinisitic computation or not
    if args.debugmode < 1:
        torch.backends.cudnn.deterministic = False
        logging.info('torch cudnn deterministic is disabled')
    else:
        torch.backends.cudnn.deterministic = True

    # check cuda availability
    if not torch.cuda.is_available():
        logging.warning('cuda is not available')

    # get input and output dimension info
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']
    utts = list(valid_json.keys())
    idim = int(valid_json[utts[0]]['input'][0]['shape'][1])
    odim = int(valid_json[utts[0]]['output'][0]['shape'][1])
    logging.info('#input dims : ' + str(idim))
    logging.info('#output dims: ' + str(odim))

    # specify attention, CTC, hybrid mode
    if args.mtlalpha == 1.0:
        mtl_mode = 'ctc'
        logging.info('Pure CTC mode')
    elif args.mtlalpha == 0.0:
        mtl_mode = 'att'
        logging.info('Pure attention mode')
    else:
        mtl_mode = 'mtl'
        logging.info('Multitask learning mode')

    # specify model architecture
    e2e = E2E(idim, odim, args)
    model = Loss(e2e, args.mtlalpha)

    # write model config
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    model_conf = args.outdir + '/model.json'
    with open(model_conf, 'wb') as f:
        logging.info('writing a model config file to ' + model_conf)
        f.write(json.dumps((idim, odim, vars(args)), indent=4, sort_keys=True).encode('utf_8'))
    for key in sorted(vars(args).keys()):
        logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))

    reporter = model.reporter

    # check the use of multi-gpu
    if args.ngpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.ngpu)))
        logging.info('batch size is automatically increased (%d -> %d)' % (
            args.batch_size, args.batch_size * args.ngpu))
        args.batch_size *= args.ngpu

    # set torch device
    device = torch.device("cuda" if args.ngpu > 0 else "cpu")
    model = model.to(device)

    # Setup an optimizer
    if args.opt == 'adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(), rho=0.95, eps=args.eps)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters())

    # FIXME: TOO DIRTY HACK
    setattr(optimizer, "target", reporter)
    setattr(optimizer, "serialize", lambda s: reporter.serialize(s))

    # Setup a converter
    converter = CustomConverter(e2e.subsample[0])

    # read json data
    with open(args.train_json, 'rb') as f:
        train_json = json.load(f)['utts']
    with open(args.valid_json, 'rb') as f:
        valid_json = json.load(f)['utts']

    # make minibatch list (variable length)
    train = make_batchset(train_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    valid = make_batchset(valid_json, args.batch_size,
                          args.maxlen_in, args.maxlen_out, args.minibatches)
    
    # Resume from a snapshot
    if args.resume:
        logging.info('TODO resumed from %s' % args.resume)
        #torch_resume(args.resume, trainer)
    
    best = dict(loss=float("inf"), acc=-float("inf"))
    opt_key = "eps" if args.opt == "adadelta" else "lr"

    def get_opt_param():
        return optimizer.param_groups[0][opt_key]


    # training loop
    result = GlobalResult(args.epochs, args.outdir)
    for epoch in range(args.epochs):
        model.train()
        with result.epoch("main", train=True) as train_result:
            for batch in np.random.permutation(train):
                x,y,z=converter([converter.transform(batch)], device)
                # forward
                loss_ctc, loss_att, acc = model.predictor(x,y,z)
                loss = args.mtlalpha * loss_ctc + (1 - args.mtlalpha) * loss_att
                # backward
                optimizer.zero_grad()  # Clear the parameter gradients
                loss.backward()  # Backprop
                loss.detach()  # Truncate the graph
                # compute the gradient norm to check if it is normal or not
                grad_norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
                logging.info('grad norm={}'.format(grad_norm))
                if math.isnan(grad_norm):
                    logging.warning('grad norm is nan. Do not update model.')
                else:
                    optimizer.step()
                # print/plot stats to args.outdir/results
                train_result.report({
                    "loss": float(loss),
                    "acc": float(acc),
                    "loss_ctc": float(loss_ctc),
                    "loss_att": float(loss_att),
                    "grad_norm": grad_norm,
                    opt_key: get_opt_param()
                })

        with result.epoch("validation/main", train=False) as valid_result:
            model.eval()
            for batch in valid:
                x,y,z=converter([converter.transform(batch)], device)
                # forward (without backward)
                loss_ctc, loss_att, acc = model.predictor(x,y,z)
                loss = args.mtlalpha * loss_ctc + (1 - args.mtlalpha) * loss_att
                # print/plot stats to args.outdir/results
                valid_result.report({
                    "loss": float(loss),
                    "acc": float(acc),
                    "loss_ctc": float(loss_ctc),
                    "loss_att": float(loss_att),
                    opt_key: get_opt_param()
                })

        # save/load model
        valid_avg = valid_result.average()
        degrade = False
        if best["loss"] > valid_avg["loss"]:
            best["loss"] = valid_avg["loss"]
            torch.save(model.state_dict(), args.outdir + "/model.loss.best")
        elif args.criterion == "loss":
            degrade = True

        if best["acc"] < valid_avg["acc"]:
            best["acc"] = valid_avg["acc"]
            torch.save(model.state_dict(), args.outdir + "/model.acc.best")
        elif args.criterion == "acc":
            degrade = True

        if degrade:
            key = "eps" if args.opt == "adadelta" else "lr"
            for p in optimizer.param_groups:
                p[key] *= args.eps_decay
            model.load_state_dict(torch.load(args.outdir + "/model." + args.criterion + ".best"))
    

def recog(args):
    '''Run recognition'''
    # seed setting
    torch.manual_seed(args.seed)

    # read training config
    idim, odim, train_args = get_model_conf(args.model, args.model_conf)

    # load trained model parameters
    logging.info('reading model parameters from ' + args.model)
    e2e = E2E(idim, odim, train_args)
    model = Loss(e2e, train_args.mtlalpha)
    torch_load(args.model, model)

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(
                len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            len(word_dict), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM(word_rnnlm.predictor,
                                           rnnlm.predictor, word_dict, char_dict))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor,
                                              word_dict, char_dict))

    # read json data
    with open(args.recog_json, 'rb') as f:
        js = json.load(f)['utts']

    # decode each utterance
    new_js = {}
    with torch.no_grad():
        for idx, name in enumerate(js.keys(), 1):
            logging.info('(%d/%d) decoding ' + name, idx, len(js.keys()))
            feat = kaldi_io_py.read_mat(js[name]['input'][0]['feat'])
            nbest_hyps = e2e.recognize(feat, args, train_args.char_list, rnnlm)
            new_js[name] = add_results_to_json(js[name], nbest_hyps, train_args.char_list)

    # TODO(watanabe) fix character coding problems when saving it
    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, sort_keys=True).encode('utf_8'))
