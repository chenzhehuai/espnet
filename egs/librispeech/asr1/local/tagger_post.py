#!/usr/bin/env python2

import sys, json
for line in sys.stdin:
  la=line.split("\t")
  if len(la)<3:
    continue
  snt_js=eval(la[1])
  lab_js=eval(la[3])
  out=''
  for i in range(len(snt_js)):
    if lab_js[i].find("NoLab") == -1:
      lab=lab_js[i].split(":")[1]
      out+="<"+lab+"> "+snt_js[i]+" </"+lab+">"+" "
    else:
      out+=snt_js[i]+" "
  print out
