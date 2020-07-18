#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 22:48:21 2020

@author: trb
"""
import random
import numpy as np
from collections import Counter




def random_choose_ref(stype,sty,chooserefname):
    chooserefname.sort()
    subtype=[]
    for i in chooserefname:
        if i.split('/')[1].split('_')[0]=='A':
            subtype.append('A9')
        else: 
            subtype.append(i.split('/')[1].split('_')[0])
    d=Counter(subtype)
    refname=[]
    s=0
    summ=0
    for key ,value in d.items():
        summ+=value
        
        crfnum=[]
        if value>5:
            crfnum.append(random.sample(range(s,s+value),5))
            s=s+value
            num=np.array(crfnum)
            for i in range(num.shape[1]):
                refname.append(chooserefname[num[0,i]])
        else:
            for j in range(s,s+value):
                refname.append(chooserefname[j])
            s=s+value 
            
    f=open('ref/'+stype+sty+'_ref_name','w')

    f.write(str(len(refname))+'\n')
    for i in refname:
        f.write(i+'\n')
    f.close()
    
    
stypes=['pol','cg']
stys=['pure','crf']

for stype in stypes:
    for sty in stys:
        f=open('dataname/'+stype+sty+'train').readlines()
        chooserefname=[i.replace('\n','') for i in f]
        random_choose_ref(stype,sty,chooserefname[1:])