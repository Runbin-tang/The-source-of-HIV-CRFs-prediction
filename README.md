# The-source-of-HIV-CRFs-prediction
Complete source prediction for HIV-1 circulating recombinant forms (CRFs) by multi-label learning

getting features folder is  to obtain the features from the sequences.\
model folder is the position of saving the models.\
data1 is the position of saving the data.

configure is the information of the upload files.\
dataprocess.py is the public module of train.py and test_voting.py \
train.py is to train data and get the models\
test_voting.py predictes the data with the three models.


##when training models (because the data file save at the data1/, so it only need to run the train.py, else you need to adjust the position of the data\
1.python train.py

##when predicting,
1. rewrite the configure 
2. python test_voting.py
