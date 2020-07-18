#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:11:14 2020

@author: trb
"""
from collections import Counter
import random

  
def randomchoose(datafile):
    data=open(datafile).readlines()
    n=[i.replace('\n','').split('/')[1].split('_')[0] for i in data]
    
    d=Counter(n)
    i=0
    choose=[];randum=[];s=0;
    for key ,value in d.items():
           choose.append(int(value*0.2))
           randum.append(random.sample(range(s,s+value),int(value*0.2)))
           s=s+value
    for i in range(len(randum)-1):
        randum[0].extend(randum[i+1])
    randumm=randum[0]
    return randumm

dd=['cgpure.txt','cgcrf.txt','polpure.txt','polcrf.txt']

test=[randomchoose(i) for i in dd]


name=['cgpure','cgcrf','polpure','polcrf']

for i in range(4):
    fftest = open(name[i]+'test','w')
    fftrain= open(name[i]+'train','w')
    f=open(dd[i]).readlines()
    for j in range(len(f)):
        if j in test[i]:
            fftest.write(f[j])
            
        elif j not in test[i]:
            fftrain.write(f[j])

fftest.close()
fftrain.close()



        
    




