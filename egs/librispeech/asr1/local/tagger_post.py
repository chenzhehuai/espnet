#!/usr/bin/env python2

import sys, json
for line in sys.stdin:
  la=line.split("\t")
  if len(la)<3:
    continue
  snt0=la[0].split()
  snt_js=eval(la[1])
  lab_js=eval(la[3])
  out=''
  i=0
  for j in range(len(snt0)):
    if i>=len(snt_js) or snt0[j]!=snt_js[i]: # we do this because deeptext will miss some words
      out+=snt0[j]+" "
      continue
    if lab_js[i].find("NoLab") == -1:
      lab=lab_js[i].split(":")[1]
      out+="<"+lab+"> "+snt_js[i]+" </"+lab+">"+" "
    else:
      out+=snt_js[i]+" "
    i+=1
  print out
