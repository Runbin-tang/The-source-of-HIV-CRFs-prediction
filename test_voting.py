#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 12:04:22 2019

@author: trb, email: tangrb@aliyun.com
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 02:00:11 2019

@author: trb
"""

#from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from functools import partial
from collections import Counter
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
import numpy as np
import dataprogress,os,string#,time,multiprocessing

def invert(p,data,stypee):
    
    labels=[]
    if stypee=='crf':
        label_list=['01','02','06','07','A','A1','A2','A3','A4','A5',
                    'A6','B','C','D','E','F','F1','F2','G','H','J','K','U']
    else :
         label_list=['A','A1','A2','A3','A4','A5',
                    'A6','B','C','D','E','F','F1','F2','G','H','J','K','U']
    for i in range(p.shape[0]):
        index=np.argwhere(p[i]==1)
        label=[]
        label.append(data[i])
        for j in range(index.shape[0]):
            label.append(label_list[index[j,0]])
        labels.append(label)
        
    return labels
def subtype_combine(subtype):  #  make the subtypes combinie the crf 
    
    if subtype==['A','E']:
        return ['01']
    if subtype==['A','G']:
        return ['02']
    if subtype==['A','B']:
        return ['03']
    if subtype==['A','G', 'H', 'K', 'U']:
        return ['04']
    if subtype==['D','F']:
        return ['05']
    if subtype==['A','G','J','K']:
        return ['06']
    if subtype==['B','C']:
        return ['07','08','31','57' , '60' , '61' , '62' , '64' , '85' , '86' , '88']
    if subtype==['A','G','U']:
        return ['09','25']
    if subtype==['C','D']:
        return ['10','41']
    if subtype==['A','E','G', 'J', 'U']:
        return ['11']
    if subtype==['B','F1']:
         return ['12','17','28','29','38','39','40','42','44','46','47','70','71','72','90']
    if subtype==['01','A','G','J','U']:
         return ['13']
    if subtype==['B','G']:
        return ['14','20','23','24','73']
    if subtype==['01','B']:
        return ['15','33' , '34' , '48' , '51' , '52' , '53' , '54' , '55' , '58' , '59' , '67' , '68' , '69' , '74' , '76']
    if subtype==['A2','D']:
        return ['16','21']
    if subtype==['A1','F','G','H','K','U']:
        return ['18']
    if subtype==['A1','D','G']:
        return ['19']
    if subtype==['01','A1']:
        return ['22']
    if subtype==['A5','U']:
        return ['26']
    if subtype==['A','E','G','H','J','K','U']:
        return ['27']
    if subtype==['02','06']:
        return ['30' ]   
    if subtype==['06','A6']:
        return ['32']
    if subtype==['A','D']:
        return ['35']  
    if subtype==['01','02','A','G']:
        return ['36' ]       
    if subtype==['01','02','A','G','U']:
        return ['37']
    if subtype==['02','G']:
        return ['43']   
    if subtype==['A','K','U']:
        return ['45']
    if subtype==['A1','C','J','K','U']:
        return ['49']     
    if subtype==['A1','D']:
        return ['50']       
    if subtype==['02','B','G']:
        return ['56'] 
    if subtype==['A1','D']:
        return ['50']
    if subtype==['02','A6']:
        return ['63']
    if subtype==['01','B','C']:
        return ['65', '77', '78' ,'82','83','87','96']
    if subtype==['01','07']:
        return ['79','80']
    if subtype==['C','U']:
        return ['92']
    if subtype==['A1','A5','C','02','U']:
        return ['93']
    if subtype==['02','B','F2']:
        return ['94']
    if subtype==['06','B']:
        return ['98']
    else:
       # print('new crf sequence')
        return '0'


def compute_dis(v1,v2):
    
    #v2=v2[1:5000].astype(float)
    
    if np.linalg.norm(v1)*np.linalg.norm(v2)==0:
        return 0
    else:
        v2=v2/np.linalg.norm(v2)
        v1=v1/np.linalg.norm(v1)
        d=np.linalg.norm(v1-v2)
        return d
    
def distance_compare(datai,refdata): 
    v1=datai#[1:];
    v1=v1.astype(float)
    refdata1=refdata[:,1:].astype(float)
    partial_work = partial(compute_dis,v1) # 提取v1作为partial函数的输入变量 
    d = pool.map(partial_work,refdata1)
    Count=[]
    if len(d)<5:
        ll=len(d)
    else:
        ll=5
    for k in range(0,ll):
        po=d.index(sorted(d)[k])
        Count.append(refdata[po,0].split('/')[1].split('_')[0])
    Coun=Counter(Count)
    s=max(zip(Coun.values(), Coun.keys()))[1]# find the max of occures  times
    
    return s,str(max(Coun.values())/5)
    
def getoriginaltype(data):  # get the original data 01 02 A,B,C
    label=[]
    for i in range(len(data)):
        label.append(data[i][0].split('/')[1].split('_')[0])
    return label
def getresult(result):
    pre=[]
    for i in range(len(result)):
            pre.append(result[i][1])
    return pre
def acc(l,p):
    c=0
    for i in range(len(l)):
        if l[i]==p[i]:
           c=c+1
    return c/len(l)   


def find_pure(data,p_label):
    
    fp_label=[]
    
    for i in range(len(p_label)):
        if len(p_label[i])==2:       # one label
            fp_label.append(p_label[i])
        else:
 
            if 'refpure' not in vars():
                
                refpure=np.load('refpure.npy')#dataprogress.loadData('refpure.txt')
                
            #if len(p_label[i])>2:
            #    print('are you sure '+p_label[i][0]+'\t is a pure sequence')
            #if len(p_label[i])==1:
            #    print(p_label[i][0]+'\t may be a new sequence. will find in ref pure set')
            st,sco=distance_compare(data[i][1:],refpure)
            fp_label.append([p_label[i][0],st,sco])  
    return fp_label
    
def find_crf(data,p_label):
    fp_label=[]
 
    
    if 'refcrf' not in vars():
        refcrf=np.load('refcrf.npy')#dataprogress.loadData('refcrf.txt')
        refnum=[]  #save the crf reference number,such as 01 02 03 ...
        for i in range(refcrf.shape[0]):
            refnum.append(refcrf[i][0].split('/')[1].split('_')[0])
        refnum=np.array(refnum)

    
    for i in range(len(p_label)):
 
        if len(p_label[i])==1:   # not label be predicted, compare with refpure and refcrf
           # print('not predict,start to compare ref crf and ref pure')
          
            # if len(refpure)==0:
           #     refpure=dataprogress.loadData('refpure.txt')
            #refal=np.vstack((refpure,refcrf))  #combine two ndarray .
            
            st,sco=distance_compare(data[i][1:],refcrf)
            fp_label.append([p_label[i][0],st,sco])

        elif len(p_label[i])==2:   # one label. except 01 02 06 07,compare with refcrf 
            if p_label[i][1] in ['01','02','06','07']:
                fp_label.append(p_label[i])
            else:
                st,sco=distance_compare(data[i][1:],refcrf)
                fp_label.append([p_label[i][0],st,sco])
            
        elif len(p_label[i])>2:  # multi label .1. find the crfnum and compare with those crfnum/
           same=[] #the crfnum data of same label
           crf_num=subtype_combine(p_label[i][1:])
           if crf_num=='0':
               #print(i,p_label[i],'*** may be a new crf')
               st,sco=distance_compare(data[i][1:],refcrf)
               fp_label.append([p_label[i][0],st,sco])
               
           else:    
        
               for j in range(len(crf_num)):
                   index=np.argwhere(refnum ==crf_num[j]).T   #finr the same label crf position
                   
                   for samei in range(index.shape[1]):
                        same.append(refcrf[index[0,samei]])     #write data to format compared data
               same=np.array(same)
               st,sco=distance_compare(data[i][1:],same)
               fp_label.append([p_label[i][0],st,sco])        
                     
    return fp_label       

def singlemodelpre(data,modelname):
    
    data1=np.array(data)[:,1:].astype(float)
    clf=joblib.load(modelname)
    #p is the binary result 
    p=clf.predict(data1)
    labeltype=modelname.split('_model')[1].split('.')[0]
    model=modelname.split('/')[1].split('_')[0]
    #print(model)
    if model=='breknna' or model=='mlknn':
        p=p.A
    if labeltype[len(labeltype)-2:len(labeltype)]=='rf':
        #p_label is make the binary label(0 1) change the true label (such as A,B,C)  
        p_label=invert(p,data[:,0],'crf')
        #result: labels-->detail type (CRF01 CRF02 et.al. )
        result=find_crf(data,p_label)       
        
        
    else:
            
        p_label=invert(p,data[:,0],'pure') 
        result=find_pure(data,p_label)     
       
    
    # test: can delete    
    if te=='test':
        if labeltype[len(labeltype)-2:len(labeltype)]=='rf':
            label1=dataprogress.get_label(data) #true multi-label
        if labeltype[len(labeltype)-2:len(labeltype)]=='re':
            label1=(dataprogress.pure_get_label(data))  #true multi-label
            
        result1=getresult(result)
        label2=getoriginaltype(data)
        l=model+' label_accuracy: '+str(accuracy_score(label1,p))
        num='type_accuracy: '+str(acc(label2,result1))
        print(l+'\t'+num)
        f.write(l+'\t'+num+'\n')
    ##
    
    t=(model,p,p_label,result)
    return t

def judgelabels(templabel):   # Voting the predicted result of 3 classifiers 
    a=[]; #save 3 classifiers predict labels to combine strings. such as 1label=['A','C'],2label=['C','E'] a=['AC','CE']
    for i in range(3):
        lab='';
        c=sorted(templabel[i])  #sorte the labels
        for j in range(len(c)):
            lab=lab+c[j]        # combine the labels
        a.append(lab)
    Cou=Counter(a)
    la=max(zip(Cou.values(), Cou.keys()))[1] #most times(>=2) label
    if Cou[la]>1:
        pos=a.index(la)   #pos is the position of first times occurence in a 
        
        return templabel[pos] 
    
    else:   # if the combine labels occurence times=1, make the all labels, choose the detail labels if times >1 
        aa=[];b=[];
        for i in range(len(templabel)):
            for j in range(len(templabel[i])):
                aa.append(templabel[i][j])
        Ca=Counter(aa)  #statistics a
        for i in Ca.keys():
            if Ca[i]>1:
                b.append(i)
        return b
  
def voting_pre(data,datatype,stype):   #data is the path of data, datatype is the one of cg and pol
    #data1=data[:,1:];
    #np.random.shuffle(data1)
    
    vot_p=[];  #save name and the result by voting the three classifiers
    vot_l=[];   #only save the labels by voting the thress classifiers
    name=['model/breknna_model','model/mlknn_model','model/mlarm_model']
    modelname=[]
    for i in name:
        #modelname.append(i+datatype+'pure.m')
        modelname.append(i+datatype+stype+'.m')
       
    result=[]
    for modeln in modelname:
        t=singlemodelpre(data,modeln)
        result.append(t)   
    print('start voting from the 3 classifiers')  
    for i in range(len(result[0][1])):
        tempname=[];templabel=[];
        for j in range(3):
            templabel.append(result[j][2][i][1:])
            tempname.append(result[j][2][i][0])
        tempn=Counter(tempname);
        #templ=Counter(templabel);
        if tempn[tempname[0]]==3:
            b=judgelabels(templabel)
            #if len(b)==0:
            #    print(tempname[0]+' can not be predicted. Directly finding type.')
            vot_p.append([tempname[0]]+b)
            vot_l.append(b)
        else: 
            print('error(0): data is not same, please check the input data.')
    print('end voting')      
    
    binary_vot_p=dataprogress.binarylabel(vot_l,stype)            
    if stype=='crf':  
        vot_result=find_crf(data,vot_p)
        
        if te=='test':
            label1=dataprogress.get_label(data) 
        
        
    if stype=='pure':
        vot_result=find_pure(data,vot_p) 
        
        if te=='test':
            label1=dataprogress.pure_get_label(data)
            
        
    if te=='test':
        l='vot_label_accuracy: '+str(accuracy_score(label1,binary_vot_p))
        print(l)
        #print('unvoting accuracy'+ str(result[5]))
        result1=getresult(vot_result)
        label2=getoriginaltype(data)
   
        num='vot_type_accuracy: '+str(acc(label2,result1))
        print(num)
        f.write(l+'\n'+num+'\n')
        #print('unvoting accuracy'+ str(result[4]))
    vot=[]
    if len(vot_p)==len(vot_result):
        for i in range(len(vot_p)): 
            if vot_p[i][0]==vot_result[i][0]:
                vot.append(vot_p[i]+[vot_result[i][1]])
            else:
                print('error: the order of data may be disrupted')
    else: print('error: the size of two variable vot_p and vot_result is not same')
    
    return vot

def judge_unkowstype(data):
    
    if 'refcrf' not in vars():
        refcrf=np.load('refcrf.npy')#dataprogress.loadData('refcrf.txt')
        
    if 'refpure' not in vars():
        refpure=np.load('refpure.npy')#dataprogress.loadData('refpure.txt')
    
    ref=np.vstack((refpure,refcrf))  #combine two ref. refcrf + refpure
    tempresult_pure=[];tempresult_crf=[];
    for i in range(len(data)):
        st,sco=distance_compare(data[i][1:],ref)
        if st in ['A','A1','A2','A3','A4','A5','A6','B','C','D','E','F','F1','F2','G','H','J','K','U']:
            tempresult_pure.append(data[i])
        else:
            tempresult_crf.append(data[i])
                
    tempresult_pure=np.array(tempresult_pure)
    tempresult_crf=np.array(tempresult_crf)
    t=(tempresult_pure,tempresult_crf)
    return  t
def writeresult(vot):
    f.write('name\t\t type\t\t label\n')
    for i in range(len(vot)):
        f.write(str(vot[i][0])+'\t'+str(vot[i][len(vot[i])-1])+'\t'+str(vot[i][1:len(vot[i])-1])+'\n')
        
def readconfigure(name):
    configure=open(name)
    l=configure.readlines()
    for i in l:
        if 'datafile=' in i:
            datafile=i.split('=')[1].replace('\n','')
        if 'savefile=' in i:
            savefile=i.split('=')[1].replace('\n','')
        if 'ifte=' in i:
            ifte=int(i.split('=')[1].replace('\n',''))
            if ifte:
                te='test'
            else:
                te='no'
        if 'datatype=' in i :
            datatype=int(i.split('=')[1].replace('\n',''))
            if datatype:
                datatype='pol'
            else: datatype='cg'
        if 'stype=' in i:
            ssty=i.split('=')[1].replace('\n','')
            if ssty=='0':
                stype='crf'
            if ssty=='1':
                stype=='pure'
            if ssty=='2':
                stype='no'
            else:
                print('Please check the configure')
        if 'poolnum=' in i:
            poolnum=int(i.split('=')[1].replace('\n',''))
    configure=(datafile,savefile,te,datatype,stype,poolnum)     
    return configure
if __name__ == '__main__': 
    
    global pool,refcrf,refpure,refnum,te,f     
    
    configure=readconfigure('configure')
    datafile=configure[0];savefile=configure[1];
    te=configure[2];datatype=configure[3];stype=configure[4];
    pool = Pool(configure[5])

    #te=''#test'        # check if the data is the testdata in our method. 
                     # if test, the accuracy can be print, or not.
    #datatype='cg'    # cg,  pol
    #stype='crf'      # crf,  pure
    #datafile='./testdata/pol_crf_testdata.txt'  #the path of datafile
    #datafile='./99100/99.txt'#99100cg.txt'
    if not os.path.exists('./predictionresult/'):
        os.mkdir('./predictionresult')
    f=open(savefile,'w')
    #f=open('./predictionresult/HIV_voting_multi-label_'+datatype+stype+'_result.txt','w')   # save the predicting result to the file
    #f=open('99100/'+datatype+stype+'_result.txt','w')
    data=dataprogress.loadData(datafile)
    print('mission is ' +datatype +stype)
    f.write('Your mission type is '+ datafile+'\n')
    f.write('The type is '+datatype +' '+stype+'\n')
    if stype in ['crf','pure']:
        vot=voting_pre(data,datatype,stype) 
        f.write('The predicting result:'+'\n')
        writeresult(vot)
    else:
        print('First to find the type, it will cost more time')
        t=judge_unkowstype(data)
        puredata=t[0];crfdata=t[1]
        if len(puredata)>0:
            f.write('This is the puresubtype data in your data'+'\n')
            vot_pure=voting_pre(puredata,datatype,'pure')
            writeresult(vot_pure)
        if len(crfdata)>0:
            f.write('This is the CRFs data in your data'+'\n')
            vot_crf=voting_pre(crfdata,datatype,'crf')
            writeresult(vot_crf)
            
    f.close()
        
        
            