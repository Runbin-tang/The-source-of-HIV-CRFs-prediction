from skmultilearn.adapt import MLkNN,MLARAM,BRkNNaClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit,cross_val_score
from sklearn.externals import joblib
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
from sklearn.model_selection import GridSearchCV
import random,dataprogress,os
import numpy as np

def search_bestparmaters(classifier,parameters,score,traindata,trainlabel):
    gclf = GridSearchCV(classifier, parameters, scoring=score,cv=10)
    search_result=gclf.fit(traindata, trainlabel)
    return search_result


def mlknn(traindata, trainlabel,ttype):  #,valdata,val_label):
    
    #knnscore=[]
    #print("[mlknn start to class>>>>]")
    ''' find the best parameters'''
    parameters = {'k': range(2,5), 's': np.arange(0.1,0.5,0.2)}
    score = 'accuracy'
    '''search parameters'''
    search_result=search_bestparmaters(MLkNN(),parameters,score,traindata,trainlabel)

    #print (search_result.best_params_, search_result.best_score_)
    
    k=search_result.best_params_['k']
    s=search_result.best_params_['s']
    save_score('score/record',('mlknn',ttype,k,s,search_result.best_score_))
    
    clf=MLkNN(k,s)
    clf.fit(traindata, trainlabel)
    joblib.dump(clf,'./model/mlknn'+"_model"+ttype+".m")
    



def mlarmc_parameters(traindata,trainlabel,vali_data,vali_label):
     
    score=[]
    ts=np.arange(0.01,0.06,0.02)
    vs=np.arange(0.93,1.0,0.02)
    for t in ts:
        for v in vs:
            clf=MLARAM(threshold=t, vigilance=v)
            clf.fit(traindata, trainlabel)
            pre=clf.predict(vali_data)
            score.append((accuracy_score(vali_label,pre),t,v))
    return max(score)
    
def mlarmc(puredata,ttype,crfdata=''):
#    
#    parameters = {'threshold': np.arange(0.01,0.06,0.02) ,'vigilance': np.arange(0.8,1,0.05)}
#    
#    score = 'accuracy'
#    '''search parameters'''
#    
#    search_result=search_bestparmaters(MLARAM(),parameters,score,traindata,trainlabel)
#    print (search_result.best_params_, search_result.best_score_)
#    
#    t=search_result.best_params_['threshold']
#    v=search_result.best_params_['vigilance']
#   

   
    if 'pure' in ttype:
        vali_num=randomchoose(puredata)
        all_num=[i for i in range(len(puredata))]
        train_num=list(set(all_num)-set(vali_num))
        
        traindata=puredata[train_num]
        vali_data=puredata[vali_num]
        
        np.random.shuffle(traindata)
        np.random.shuffle(vali_data)
        
        trainlabel=dataprogress.pure_get_label(traindata)
        vali_label=dataprogress.pure_get_label(vali_data)
        
        traindata1=np.array(traindata)[:,1:].astype(float)
        vali_data1=np.array(vali_data)[:,1:].astype(float)
        
        best=mlarmc_parameters(traindata1,trainlabel,vali_data1,vali_label)
    
        #print(best)

        
    elif 'crf' in ttype:
        
        vali_num=randomchoose(crfdata)
        all_num=[i for i in range(len(crfdata))]
        train_num=list(set(all_num)-set(vali_num))
        
        traindata=crfdata[train_num]
        vali_data=crfdata[vali_num]
        
        traindata=np.append(traindata,puredata,axis = 0)
        
        np.random.shuffle(traindata)
        np.random.shuffle(vali_data)
        
        trainlabel=dataprogress.get_label(traindata)
        vali_label=dataprogress.get_label(vali_data)
        
        traindata1=np.array(traindata)[:,1:].astype(float)
        vali_data1=np.array(vali_data)[:,1:].astype(float)
        
        best=mlarmc_parameters(traindata1,trainlabel,vali_data1,vali_label)
        #print(best)
    mn=('mlaram',ttype)
    save_score('score/record',mn)
    save_score('score/record',best)
    #clf=MLARAM(threshold=0.07,vigilance=0.93) 
    clf=MLARAM(threshold=best[1], vigilance=best[2])
    clf.fit(traindata1,trainlabel)
    joblib.dump(clf,'./model/mlaram'+"_model"+ttype+".m")
    


def breknna(traindata, trainlabel,ttype):
    
    parameters = {'k': range(2,5)}
    score = 'accuracy'
    '''search parameters'''
    search_result=search_bestparmaters(BRkNNaClassifier(),parameters,score,traindata,trainlabel)

    #print (search_result.best_params_, search_result.best_score_)
    k=search_result.best_params_['k']
    save_score('score/record',('breknna',ttype,k,search_result.best_score_))
    
    clf=BRkNNaClassifier(k)
    clf.fit(traindata, trainlabel)
    joblib.dump(clf,'./model/breknna'+"_model"+ttype+".m")
    
def randomchoose(data):
    n=[]
    for i in range(len(data)):
        n.append(data[i,0].split('/')[1].split('_')[0])
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

def save_score(filename,score):
    w=open(filename,'a+')
    for i in score:
        w.write(str(i))
        w.write('\t')
    w.write('\n')
    
def puretrain(puredata,ttype):
    
    #u=dataprogress.loadData('./data1/'+ttype+'_u.txt')
    traindata=puredata
    
    np.random.shuffle(traindata)
    trainlabel = dataprogress.pure_get_label(traindata)
    traindata1=np.array(traindata)[:,1:].astype(float)
    
    ttype=ttype+'pure'
    mlknn(traindata1,trainlabel,ttype)
    breknna(traindata1,trainlabel,ttype)
    mlarmc(puredata,ttype)
    
def purecrftrain(puredata,crfdata,ttype):
    
    #u=dataprogress.loadData('./data1/'+ttype+'_u.txt')
    
    traindata=np.append(puredata,crfdata,axis = 0)
    
    np.random.shuffle(traindata)        
    trainlabel = dataprogress.get_label(traindata)
    #train_vallabel = dataprogress.get_label(train_valdata)
    
    traindata1=np.array(traindata)[:,1:].astype(float)
    #train_valdata1=np.array(train_valdata)[:,1:].astype(float)
    
    ttype=ttype+'crf'
    ###
    mlknn(traindata1,trainlabel,ttype)
    breknna(traindata1,trainlabel,ttype)
    mlarmc(puredata,ttype,crfdata)


if __name__ == '__main__':
    global scorefile,p    #p is crf or pure
    
    for foldername in ['score','model']:
        if not os.path.exists(foldername):
            os.makedirs(foldername)

    print("[load data ...]")
    
    ttypee=['cg','pol']
    filename=[]
    
    for i in ttypee:
        for j in 'pure','crf':
            filename.append('./data/'+i+j+'train.txt')
        
    
    pool = ThreadPool()
    data=pool.map(dataprogress.loadData, filename)
    
    cgpuredata  = data[0]
    cgcrfdata   = data[1]
    polpuredata = data[2]
    polcrfdata  = data[3]

    print('start ')   
    p='pure'
    #puretrain(cgpuredata,'cg')
    puretrain(polpuredata,'pol')
    p='crf'
    #purecrftrain(cgpuredata,cgcrfdata,'cg')
    purecrftrain(polpuredata,polcrfdata,'pol')
    
