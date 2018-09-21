#!/bin/bash
#based on local/run_100.sh & ~/fbsource/fbcode/langtech/ninja/kaldi_interface/mkgraph.sh

. path.sh

set -x 

lex=/mnt/homedir/yqw/Work/Projects/ctc-kaldi/HCLG/lib/kaldi-tree/lexicon.txt
lang=data/lang_classlm
dict=data/lang_1char/train_100_units.txt
stage=0
clname="@name @number"
clv="calling_full_1k calling_full_1k"

if [ $stage -le 0 ]; then
  
rm -rf $lang
mkdir -p $lang
ln -s `pwd`/lms/lm-2gram.arpa  $lang/G.arpa

# from utils/prepare_lang.sh
awk '{print $1}' < "$lex"  | sort | uniq  | awk '
  BEGIN {
    print "<eps> 0";
  }
  {
    if ($1 == "<s>") {
      print "<s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    if ($1 == "</s>") {
      print "</s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    printf("%s %d\n", $1, NR);
  }
  END {
    printf("#0 %d\n", NR+1);
    printf("<s> %d\n", NR+2);
    printf("</s> %d\n", NR+3);
  }' \
    | awk -v clname="$clname" '{print; n=$2}END{split(clname, a, " "); for (i=1;i<=length(a);i++){n+=1;print a[i],n}}' > $lang/words.txt

awk -v clname="$clname" 'BEGIN{print "<eps> 0"}{print; n=$2}END{split(clname, a, " "); for (i=1;i<=length(a);i++){n+=1;print a[i],n}}' $dict > $lang/phones.txt
fi

if [ $stage -le 1 ]; then
  awk 'NR==FNR{d[$1]=$1}$0~"\@"{print $1,$1; next}NR!=FNR&&$0!~"#0"&&$0!~"<s>"&&$0!~"</s>"&&$0!~"<eps>"{printf $1;for (i=1;i<=length($1);i++){printf " "d[toupper(substr($1,i,1))]}printf " <space>\n"}'  data/lang_1char/train_100_units.txt $lang/words.txt > $lang/lexicon.txt
MODEL_DIR=$lang
if [[ ! -f "$MODEL_DIR/G.fst" ]]; then
arpa2fst --disambig-symbol=#0 \
  "--read-symbol-table=$MODEL_DIR/words.txt" \
  "$MODEL_DIR/G.arpa" \
  "$MODEL_DIR/G.fst"
fi

ndisambig=$("$KALDI_ROOT/egs/wsj/s5/utils/add_lex_disambig.pl" \
  "$MODEL_DIR/lexicon.txt" \
  "$MODEL_DIR/lexicon_disambig.txt")

# Next, create a phones.txt file that includes the disambiguation symbols.
# the --include-zero includes the #0 symbol we pass through from the grammar.
"$KALDI_ROOT/egs/wsj/s5/utils/add_disambig.pl" \
  --include-zero "$MODEL_DIR/phones.txt" "$ndisambig" \
  > "$MODEL_DIR/phones_disambig.txt"

phone_disambig_symbol=$(grep \#0 "$MODEL_DIR/phones_disambig.txt" | \
  awk '{print $2}')
word_disambig_symbol=$(grep \#0 "$MODEL_DIR/words.txt" | awk '{print $2}')


if [[ ! -s "$MODEL_DIR/L_disambig.fst" ]]; then
  "$KALDI_ROOT/egs/wsj/s5/utils/make_lexicon_fst.pl" \
    "$MODEL_DIR/lexicon_disambig.txt"  | \
    fstcompile "--isymbols=$MODEL_DIR/phones_disambig.txt" \
      "--osymbols=$MODEL_DIR/words.txt" \
      --keep_isymbols=false --keep_osymbols=false | \
    "fstaddselfloops" \
      "echo $phone_disambig_symbol |" \
      "echo $word_disambig_symbol |" | \
    "fstarcsort" --sort_type=olabel > "$MODEL_DIR/L_disambig.fst"
fi

grep '#' "$MODEL_DIR/phones_disambig.txt" | \
  awk '{print $2}' > "$MODEL_DIR/disambig_phones.int"
fi



if [ ${stage} -le 6 ]; then

   fsttablecompose $lang/L_disambig.fst   $lang/G.fst \
  | fstproject --project_output=false \
  | fstdeterminizestar --use-log=true \
  | fstminimizeencoded | fstarcsort --sort_type=ilabel \
  | fstrmepslocal \
  | fstarcsort \
  > $lang/SLG.fst
   fstprint $lang/SLG.fst | awk 'NF>3{s+=$5;c+=1}END{print NR,c,s/c}'
   #project the input so that still output grapheme; improve determinize?
   #as phones.txt and train_100_units.txt are mismatch, map the in-id
   #also make SIL SP to unk, make #* to 0
   #the last fstminimizeencoded is to avoid more than one backoff (which is assumed by kaldi BackoffDeterministicOnDemandFst)
 
fi 

if [ ${stage} -le 7 ]; then
c=1
for i in $clv; do 

#gen linear fst using namelist
#det & min
cat $i | sed 's/  */ /g;s/\t/ /g' \
  | awk 'NR==FNR{if ($1=="<space>"){d[" "]=$1}else{d[$1]=$1}}NR!=FNR&&$0!~"#0"&&$0!~"<s>"&&$0!~"</s>"&&$0!~"<eps>"{for (i=1;i<=length($0);i++){printf " "d[toupper(substr($0,i,1))]}printf " <space>\n"}'  $dict - \
  | awk 'BEGIN{s=0;e=1;c=2}NR==FNR{d[$1]=$2}NR!=FNR{for (i=1;i<=NF;i++){if (i>1){print c,c+1,d[$i],d[$i];c+=1} else {print s,c,d[$i],d[$i]}}print c,e,0,0;c+=1}END{print e}' $dict - \
  | fstcompile \
  | fstdeterminizestar  \
  | fstminimizeencoded | fstarcsort --sort_type=ilabel \
  > $lang/$c.fst
c=`echo $c|awk '{print $1+1}'`
done

#obtain @name id 
awk -v di=$lang -v clname="$clname" 'BEGIN{split(clname,a," ");for (i=1;i<=length(a);i++){d[a[i]]=i}}{if (d[$1]){print di"/"d[$1]".fst",$2,$1}}' $lang/phones.txt \
> $lang/num.pid.p

#replace
clv_str=`awk '{printf "%s %i ",$1,$2}' $lang/num.pid.p`
fstreplace $lang/SLG.fst -1 $clv_str  $lang/SLG.fst.fill

fi 
