from __future__ import print_function
import os
import contextlib
import json
import time

import torch
import matplotlib

matplotlib.use('Agg')

from matplotlib import pyplot


def to_float(x, name):
    if isinstance(x, float):
        return x
    if isinstance(x, int):
        return float(x)
    elif isinstance(x, torch.autograd.Variable):
        return x.data[0]
    else:
        raise NotImplementedError("{} is unknown-type: {} of {}".format(name, x, type(x)))


def plot_seq(d, path):
    fig, ax = pyplot.subplots()
    for k, xs in d.items():
        ax.plot(range(len(xs)), xs, label=k, marker="x")

    l = ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.grid()
    fig.savefig(path, bbox_extra_artists=(l,), bbox_inches='tight')
    pyplot.close()


def default_print(*args, **kwargs):
    print(*args, flush=True, **kwargs)


class GlobalResult(object):
    def __init__(self, max_epoch, outdir=None, float_fmt="{}",
                 report_every=100, logfun=default_print, optional=dict()):
        self.outdir = outdir
        if outdir is not None:
            self.log_path = outdir + "/log"
            with open(self.log_path, "w") as f:
                json.dump([], f)
        else:
            self.log_path = None
        self.logfun = logfun
        self.max_epoch = max_epoch
        self.report_every = report_every
        self.float_fmt = float_fmt
        self.start_time = time.time()
        self.current_epoch = 0
        self.plot_dict = dict()

    def elapsed_time(self):
        return time.time() - self.start_time

    @contextlib.contextmanager
    def epoch(self, prefix, train):
        try:
            if train:
                self.current_epoch += 1
            e_result = EpochResult(self, prefix, train)
            yield e_result
        finally:
            self.logfun("[{}] {}-epoch: {}\t{}".format(
                prefix, "train" if train else "valid",
                self.current_epoch, e_result.summary()))
            e_result.dump()

            if self.outdir is not None:
                avg = e_result.average()
                for k, v in avg.items():
                    if k not in self.plot_dict.keys():
                        self.plot_dict[k] = dict()
                    if prefix not in self.plot_dict[k].keys():
                        self.plot_dict[k][prefix] = []
                    self.plot_dict[k][prefix].append(avg[k])
                    plot_seq(self.plot_dict[k], self.outdir + "/" + k + ".png")


class EpochResult(object):
    def __init__(self, global_result, prefix, train):
        self.global_result = global_result
        self.train = train
        self.sum_dict = dict()
        self.iteration = 0
        self.logfun = global_result.logfun
        self.log_path = global_result.log_path
        self.prefix = prefix
        self.float_fmt = global_result.float_fmt

    def summary(self):
        s = ""
        fmt = "{}: " + self.float_fmt + "\t"
        for k, v in self.average().items():
            s += fmt.format(k, v)
        s += "elapsed: " + time.strftime("%X", time.gmtime(self.global_result.elapsed_time()))
        return s

    def dump(self):
        if self.log_path is None:
            return

        with open(self.log_path, "r") as f:
            d = json.load(f)

        elem = {
            "epoch": self.global_result.current_epoch,
            "iteration": self.iteration,
            "elapsed_time": self.global_result.elapsed_time()
        }

        for k, v in self.average().items():
            elem[self.prefix + "/" + k] = v

        d.append(elem)
        with open(self.log_path, "w") as f:
            json.dump(d, f, indent=4)

    def report(self, d):
        for k, v in d.items():
            if k not in self.sum_dict.keys():
                self.sum_dict[k] = to_float(v, k)
            else:
                self.sum_dict[k] += to_float(v, k)
        self.iteration += 1
        if self.train and self.iteration % self.global_result.report_every == 0:
            self.logfun("train-iter: {}\t{}".format(self.iteration, self.summary()))
            self.dump()

    def average(self):
        return {k: v / self.iteration for k, v in self.sum_dict.items()}
