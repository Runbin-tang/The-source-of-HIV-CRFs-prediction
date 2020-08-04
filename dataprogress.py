#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 09:25:54 2019

@author: trb
"""
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter
import numpy as np
import random,os

def subtype(c):
    #if c=='01' : 
    #    return ['A','E']
    #if c=='02': 
    #    return ['A','G']
    if c=='03': 
        return ['A','B']
    if c=='04': 
        return ['A','G','H','K','U']
    if c=='05' : 
        return ['D','F']
    #if c=='06' :
    #    return ['A','G','J','K']
    if c in['08' ,'31' , '57' , '60' , '61' , '62' , '64' , '85' , '86' , '88'] : 
        return ['B','C']
    if c in ['09' , '25'] :
       return ['A','G','U']
    if c in ['10' , '41'] : 
        return ['C','D']
    if c=='11' :
      return ['A','E','G','J','U']
    if c in ['12' , '17' , '28' , '29' , '38' , '39' , '40' , '42' , '44' , '46' , '47' , '70' , '71' , '72','90']: 
        return ['B','F1']
    if c=='13':
      return ['01','A','G','J','U']
    if c in ['14' , '20' , '23' , '24' , '73'] : 
        return ['B','G']
    if c in ['15' , '33' , '34' , '48' , '51' , '52' , '53' , '54' , '55' , '58' , '59' , '67' , '68' , '69' , '74' , '76'] : 
        return ['01','B']
    if c in [ '16' , '21'] : 
        return ['A2','D']
    if c=='18' :
      return ['A1','F','G','H','K','U']
    if c=='19' :
      return ['A1','D','G']
    if c=='22' : 
        return ['01','A1']
    if c=='26': 
        return ['A5','U']
    if c=='27':
      return ['A','E','G','H','J','K','U']
    if c=='30' : 
        return ['02','06']
    if c=='32' : 
        return ['06','A6']
    if c=='35' : 
        return ['A','D']
    if c=='36' :
      return ['01','02','A','G']
    if c=='37' :
      return ['01','02','A','G','U']
    if c=='43' : 
        return ['02','G']
    if c=='45' :
      return ['A','K','U']
    if c=='49' :
      return ['A1','C','J','K','U']
    if c=='50' : 
        return ['A1','D']
    if c=='56' :
      return ['02','B','G']
    if c=='63' : 
        return ['02','A6']
    if c in ['65' , '77' , '78' , '82' , '83' , '87','96'] :
      return ['01','B','C']
    if c in ['79' , '80' ]: 
        return ['01','07']
    if c=='92' : 
        return ['C','U']
    if c=='93' :
      return ['A1','A5','C','02','U']
    if c=='94' :
      return ['02','B','F2']
    if c=='98' : 
      return ['06','B']
    else:
       #print(c)
        return [c]
def get_label(data):
    label_list=['01','02','06','07','A','A1','A2','A3','A4','A5',
                'A6','B','C','D','E','F','F1','F2','G','H','J','K','U']
    label=[]
    label.append(label_list)
    for i in range(len(data)):
        s=subtype(data[i][0].split('/')[-1].split('_')[0])
        label.append(s)
        if s[0] not in label_list:
            print(i,data[i][0],'not in the test set, please rewrite the "ifte" in config)
        
    #d=Counter(label)
    mlb = MultiLabelBinarizer()
    label = mlb.fit_transform(label)
    label=np.array(label[1:,:])#len(label)-1]
    return label 
def pure_get_label(data):

    label_list=['A','A1','A2','A3','A4','A5',
                'A6','B','C','D','E','F','F1','F2','G','H','J','K','U']
    label=[]
    label.append(label_list)
    for i in range(len(data)):
        s=subtype(data[i][0].split('/')[-1].split('_')[0])
        label.append(s)
        if s[0] not in label_list:
            print(i,data[i][0],'not in the test set, please rewrite the "ifte" in config')
        
    #d=Counter(label)
    mlb = MultiLabelBinarizer()
    label = mlb.fit_transform(label)
    label=np.array(label[1:,:])#len(label)-1]
    return label 
def binarylabel(p,stype):
    label=[]
    crflabel_list=['01','02','06','07','A','A1','A2','A3','A4','A5',
                'A6','B','C','D','E','F','F1','F2','G','H','J','K','U']
    purelabel_list=['A','A1','A2','A3','A4','A5',
                'A6','B','C','D','E','F','F1','F2','G','H','J','K','U']
    if stype=='crf':
        label.append(crflabel_list)
    if stype=='pure':
        label.append(purelabel_list)
    for i in range(len(p)):
        label.append(p[i])
    mlb = MultiLabelBinarizer()
    label = mlb.fit_transform(label)
    label=np.array(label[1:,:])#len(label)-1]
    return label        
def loadData(datafileName):
    #data = np.genfromtxt(datafileName,dtype=str,delimiter=',')
    data = np.loadtxt(datafileName,dtype=str,delimiter=',')
    return data
def mkdirfiles(filename):
    if not os.path.exists(filename): 
        os.mkdir('./'+filename)

def choose_crf_ref(filename):   #choose 3 sequences for every crf sequence 
    f=open(filename)  #filename= crfname.txt
    name=f.readlines()
    choosecrfname=[]
    for i in name:
        choosecrfname.append(i.split('/')[1])
    subtype=[]
    for i in choosecrfname:
        subtype.append(i.split('_')[0])
    d=Counter(subtype)
    crfname=[]
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
                crfname.append(choosecrfname[num[0,i]])
        else:
            for j in range(s,s+value):
                crfname.append(choosecrfname[j])
            s=s+value       
    return crfname 
        
    
    
