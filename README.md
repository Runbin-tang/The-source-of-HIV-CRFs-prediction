# The-source-of-HIV-CRFs-prediction
Genetic source completeness of HIV-1 circulating recombinant forms (CRFs) predicted by multi-label learning

getting features folder is  to obtain the features from the sequences.\
model folder is the position of saving the models.\
data1 is the position of saving the data.

configure is the information of the upload files.\
dataprocess.py is the public module of train.py and test_voting.py \
train.py is to train data and get the models\
test_voting.py predictes the data with the three models.

##getting the fearures, there is a start.bat file in the getting features folder, which can be directly used. You can also use the following command to generate data. The content of dataname_file is the absolute path where you save the data. save_filename is the name to save the generated file. satandard_k-mers is file name of the standard k-mers in paper (i.e. 10cg_5000.txt).
1. hiv_dltree_get-feature_position_frequency_new.exe dataname_file save_filename standard_k-mers

##when training models (because the data file save at the data1/, so it only need to run the train.py, else you need to adjust the position of the data, or replace the 302 and 303 lines in train.py as the path of your data ）

1. python train.py

##when predicting,
1. rewrite the configure 
2. python test_voting.py
