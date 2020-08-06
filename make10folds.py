import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import pandas as pd
import os

def make_10folds(path,new_path):
    csv = pd.read_csv(path,header=None)
    csv = np.array(csv)
    name = csv[0,:]
    name = np.reshape(name,(1,len(name)))
    name[0,2] = 'status'
    features = csv[1:,:]
    # label = features[:,2]

    sfolder = StratifiedKFold(n_splits=10,random_state=0,shuffle=True)
    i = 1
    for train_index, test_index in sfolder.split(features,features[:,2]):
        features_train, features_test = features[train_index], features[test_index]
        features_train_h = np.concatenate((name,features_train), axis=0)
        features_test_h = np.concatenate((name,features_test), axis=0)
        features_train_h_pd = pd.DataFrame(features_train_h)
        features_test_h_pd = pd.DataFrame(features_test_h)
        new_fold_path = new_path+str(i)+'/'
        if not os.path.exists(new_fold_path):
            os.makedirs(new_fold_path)
        features_train_h_pd.to_csv(new_fold_path+'ov_DCAS_train_f'+str(i)+'.csv',header=None,index=False)
        features_test_h_pd.to_csv(new_fold_path+'ov_DCAS_test_f'+str(i)+'.csv',header=None,index=False)
        i+=1
make_10folds('./ov_DCAS_weighted_features.csv','./folds/')


