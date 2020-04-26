from skmultilearn.adapt import MLkNN,MLARAM,BRkNNaClassifier,MLTSVM,BRkNNbClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ShuffleSplit,cross_val_score
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import roc_auc_score
from sklearn.externals import joblib
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
import matplotlib.pyplot as plt
import random,dataprogress,os
import numpy as np

def kfoldcv(clf,traindata,trainlabel):
    print("[k-fold cv starting ...]")
    #k_scores = []
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        ##    loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error') # for regression
    scores = cross_val_score(clf, traindata, trainlabel, cv=cv, scoring='accuracy') # for classification
   #k_scores.append(scores.mean())
    return scores.mean()
    
def pre(model,clf,testdata,testlabel,data11,ttype): #p is crf or pure
    
    print(model+"  [starting test...]")
    predictions = clf.predict(testdata)
    #predictions = np.array(predictions)
    #print(predictions)
    accauracy_value=accuracy_score(testlabel,predictions)
    print (model+ '  train-val_acc: ', accauracy_value)
    print(model+"  [starting predict...]")
    data1=np.array(data11)[:,1:].astype(float)
   
    pre1=clf.predict(data1)
    if p=='pure':
        a=accuracy_score(dataprogress.pure_get_label(data11),pre1)
    else :
        a=accuracy_score(dataprogress.get_label(data11),pre1)
    
    print('[ ' + model+'  test_data]: ',a)
    
    ff=open('./result/'+ttype+'_restult.txt','a+')
    ff.write(model+'\t predict result: \n')
    ff.write( '\ttrain-val_acc:\t'+str(accauracy_value)+'\n')
    ff.write('\ttest_data:\t'+str(a )+'\n')
    #ff.write('\tcombine test and crfbig59:\t'+str(c)+'\n')
    ff.close()
    joblib.dump(clf,'./model/'+model+"_model"+ttype+".m")
    return pre1

def acu_curve(y,prob):
    fpr,tpr,threshold = roc_curve(y,prob) 
    roc_auc = auc(fpr,tpr) 
 
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right") 
    plt.show()

def rocp(model,testlabel,pre):
    fpr,tpr,thresholds=roc_curve(testlabel,pre)
    print(roc_auc_score(testlabel,pre))
    roc_auc=auc(fpr,tpr)
    plt.figure()
    lw=2
    plt.figure(figsize=(10,10))
    plt.plot(fpr,tpr,color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(model+'Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def mlknn(traindata, trainlabel,testdata,testlabel,data1,ttype):
    
    knnscore=[]
    print("[mlknn start to class>>>>]")
    ki=1;si=0.1
    clf=MLkNN(k=ki,s=si) 
    knnscore.append([ttype+'mlknn',ki,si,kfoldcv(clf,traindata,trainlabel)])
    
    clf.fit(traindata, trainlabel)
    mlknn_pre=pre('mlknn',clf,testdata,testlabel,data1,ttype)#.A
    print(ttype+ " mlknn end to class...]")
    print("<<<<<<")
    #rocp(mlknn,np.array(testlabel),np.array(mlknn_pre))
    return knnscore,mlknn_pre

def mlarmc(traindata, trainlabel,testdata,testlabel,data1,ttype):
    print(ttype+" mlarmc start to class>>>>")
    armkscore=[]
    ti=0.05;vi=0.95
    clf=MLARAM(threshold=ti, vigilance=vi)
    armkscore.append([' mlarm ',ti,vi,kfoldcv(clf,traindata,trainlabel)])
    
    clf.fit(traindata, trainlabel)
    mlarm_pre=pre('mlarm',clf,testdata,testlabel,data1,ttype)
    print(ttype+"mlarm end to class...")
    print("<<<<<<")
    return armkscore,mlarm_pre

def brknnb(traindata, trainlabel,testdata,testlabel,data1,ttype):
    
    print(ttype+  'brknnb start to class >>>')
    brknnbscore=[]
    clf = BRkNNbClassifier(k=10)
    brknnbscore.append([ttype+' brknnb',10,kfoldcv(clf,traindata,trainlabel)])
    print (ttype+'brknnb+start train')
    
    clf.fit(traindata, trainlabel)
    brekkpre=pre('brknnb',clf,testdata,testlabel,data1,ttype)
    
    return brekkpre,brknnbscore
    
def breknna(traindata, trainlabel,testdata,testlabel,data1,ttype):
    
    print(ttype+"  breknna start to class>>>>")
    brekkscore=[]
    clf=BRkNNaClassifier(k=3)
    brekkscore.append([ttype+' breknna',3,kfoldcv(clf,traindata,trainlabel)])
    print ('start train')
    
    clf.fit(traindata, trainlabel)
    brekkpre=pre('breknna',clf,testdata,testlabel,data1,ttype)
    print(ttype+" breknna end to class...")
    print("<<<<<<")
    return brekkscore,brekkpre

def mlsvm(traindata, trainlabel,testdata,testlabel,data1,ttype):
    
    print(ttype+" mlsvm start to class>>>>")

    for i in range(-1,2,1):
        clf=MLTSVM(c_k=2**i)
        clf.fit(traindata, trainlabel)
        svmpre=pre(clf,testdata,testlabel,data1,ttype)
    
    return svmpre
    
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
    
    u=dataprogress.loadData('../data1/'+ttype+'_u.txt')
    
    pure_random=randomchoose(puredata)
    
    traindata=[];testdata=[]
    for i in range(len(puredata)):
        if (i in pure_random):
            testdata.append(puredata[i,:])
        else:
            traindata.append(puredata[i,:])
    for i in range(len(u)):
        traindata.append(u[i,:])
        
    traindata=np.array(traindata)
    testdata=np.array(testdata)
    random.seed(42)  
    np.random.shuffle(testdata)
    np.savetxt("testdata/"+ttype+"_pure_testdata.txt", testdata, fmt="%s", delimiter=",")
    train_val_random=randomchoose(traindata)  #choose the valcation set from traindata  

    train_traindata=[];train_valdata=[];
    for i in range(len(traindata)):
        if (i in train_val_random):
            train_valdata.append(traindata[i,:])
        else:
            train_traindata.append(traindata[i,:])

    train_trainlabel = dataprogress.pure_get_label(train_traindata)
    train_vallabel = dataprogress.pure_get_label(train_valdata)
    
    train_traindata1=np.array(train_traindata)[:,1:].astype(float)
    train_valdata1=np.array(train_valdata)[:,1:].astype(float)
    
    ttype=ttype+'pure'
    ###train modles with three multi-label classifiers: mlknn breknna  mlarmc
    filename=('score/'+ttype+'score.txt')
    
    knnscore,kpre=mlknn(train_traindata1, train_trainlabel,
                        train_valdata1,train_vallabel,
                        testdata,ttype)
    save_score(filename,knnscore)
    
    brekkscore,brekkpre=breknna(train_traindata1, train_trainlabel,
                        train_valdata1,train_vallabel,
                        testdata,ttype)
    save_score(filename,brekkscore)

    armkscore,armpre=mlarmc(train_traindata1, train_trainlabel,train_valdata1,train_vallabel,testdata,ttype)
    save_score(filename,armkscore)

   
    
def purecrftrain(puredata,crfdata,ttype):
    
    u=dataprogress.loadData('../data1/'+ttype+'_u.txt')
    
    pure_random=randomchoose(puredata)
    crf_random=randomchoose(crfdata)


    traindata=[];testdata=[]
    for i in range(len(puredata)):
        if (i in pure_random):
            testdata.append(puredata[i,:])
        else:
            traindata.append(puredata[i,:])
    
    for i in range(len(crfdata)):
        if (int(i) in crf_random):
            testdata.append(crfdata[i,:])
        else:
            traindata.append(crfdata[i,:])   
    for i in range(len(u)):
        traindata.append(u[i,:])
    
    traindata=np.array(traindata)
    testdata=np.array(testdata)
    random.seed(42)  
    np.random.shuffle(testdata)
    np.savetxt("testdata/"+ttype+"_purecrf_testdata.txt", testdata, fmt="%s", delimiter=",")  # save the testdata after randly choose data
    train_val_random=randomchoose(traindata)  #choose the valcation set from traindata  

    train_traindata=[];train_valdata=[];
    for i in range(len(traindata)):
        if (i in train_val_random):
            train_valdata.append(traindata[i,:])
        else:
            train_traindata.append(traindata[i,:])
            
    train_trainlabel = dataprogress.get_label(train_traindata)
    train_vallabel = dataprogress.get_label(train_valdata)
    
    train_traindata1=np.array(train_traindata)[:,1:].astype(float)
    train_valdata1=np.array(train_valdata)[:,1:].astype(float)
    
    ttype=ttype+'crf'
    ###  train modles with three multi-label classifiers: mlknn breknna  mlarmc
    filename=('score/'+ttype+'score.txt')
    knnscore,kpre=mlknn(train_traindata1, train_trainlabel,
                        train_valdata1,train_vallabel,
                        testdata,ttype)
    save_score(filename,knnscore)

    brekkscore,brekkpre=breknna(train_traindata1, train_trainlabel,
                        train_valdata1,train_vallabel,
                        testdata,ttype)
    save_score(filename,brekkscore)

    armkscore,armpre=mlarmc(train_traindata1, train_trainlabel,train_valdata1,train_vallabel,testdata,ttype)
    save_score(filename,armkscore)




if __name__ == '__main__':
    global scorefile,p#,ttype   #p is crf or pure
    #scorefile=ttype+'scorefile.txt'
    for foldername in ['result','score','model','testdata']:
        if not os.path.exists(foldername):
            os.makedirs(foldername)
    
    print("[load data ...]")
    ttypee=['cg','pol']
    filename=[]
    for i in ttypee:
        filename.append('../data1/'+i+'pure2.txt')  # upload the path of data file
        filename.append('../data1/'+i+'crf2.txt')
    
    
    pool = ThreadPool()
    data=pool.map(dataprogress.loadData, filename)
    
    cgpuredata  = data[0]
    cgcrfdata   = data[1]
    polpuredata = data[2]
    polcrfdata  = data[3]
    # train
    p='pure'
    puretrain(cgpuredata,'cg')  
    puretrain(polpuredata,'pol')
    p='crf'
    purecrftrain(cgpuredata,cgcrfdata,'cg')
    purecrftrain(polpuredata,polcrfdata,'pol')
    
