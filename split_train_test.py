import os
import numpy as np
import pandas as pd

def get_patch_path_withlabel(path,label_path):
    label = pd.read_csv(label_path,header = None)
    label = np.array(label)
    label = label[1:,:]

    for first_path_name in os.listdir(path):
        list = []
        first_path = path+first_path_name
        for image_name in os.listdir(first_path):
            name_list = []
            image_path = first_path+'/'+image_name
            name_list.append(image_path)

            name = str(image_name.split('-')[0])+'-'+str(image_name.split('-')[1])+'-'+str(image_name.split('-')[2])+'-'+str(image_name.split('-')[3])
            for i in range(label.shape[0]):
                if name == label[i][0]:
                    name_list.append(label[i,1])
                    name_list.append(label[i, 2])
                    break
            name_list = np.reshape(name_list,(1,3))

            list.append(name_list)
        list = np.array(list)
        list = np.reshape(list,(len(list),3))


        list = pd.DataFrame(list)
        list.to_csv('./label/'+first_path_name+'.csv',header=None,index=None)
get_patch_withlabel('J:\Dling\Ovarian Cancer\cancer_cluster/','./Clinical_OV_label.csv')


def split_train_test(path):
    for csv_path_name in os.listdir(path):
        name = csv_path_name.split('.')[0]
        csv_path = path+csv_path_name
        csv = pd.read_csv(csv_path,header=None)
        csv = np.array(csv)

        np.random.shuffle(csv)
        num = len(csv)
        train_num = int(0.8*num)
        train_csv = csv[:train_num,:]
        test_csv = csv[train_num:,:]
        train_csv = pd.DataFrame(train_csv)
        test_csv = pd.DataFrame(test_csv)
        new_folder = './split_label/'+name+'/'
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        train_csv.to_csv(new_folder + name + '_train.csv', header=None, index=False)
        test_csv.to_csv(new_folder + name + '_test.csv', header=None, index=False)
# split('./label/')