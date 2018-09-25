# https://our.intern.facebook.com/intern/diff/D8837876/
# https://phabricator.intern.facebook.com/diffusion/E/browse/tfb/trunk/www/flib/langtech/cu/service/core/entity/CuNluNamedEntityType.php
set -x
stage=0
if [ $stage -le 0 ]; then
dirs=" \
  data/test_other \
  data/test_clean \
  data/dev_clean \
  data/dev_other \
  data/train_clean_100 \
  "
for dir in $dirs; do
  rm -rf $dir.tag
  mkdir $dir.tag
  awk '{$1="";print}' $dir/text  | /home/chenzhehuai/local/fbsource/fbcode/buck-out/gen/langtech/transformers/entity_tagger.par --batch_size 1000 2>/dev/null \
    | local/tagger_post.py \
    | awk 'NR==FNR{d[NR]=$1}NR!=FNR{print d[FNR],$0}' $dir/text  - \
     > $dir.tag/text
  ln -s `pwd`/$dir/* $dir.tag/
done
fi

