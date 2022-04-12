import pandas as pd
import numpy as np
import os

# def get_number(path):
#     images = []
#     for csv_path in os.listdir(path):
#         csv = pd.read_csv(path+csv_path,header=None)
#         csv = np.array(csv)
#         names = csv[:,0]
#         for name in names:
#             patient_name = name.split('-')[0]+'-'+name.split('-')[1]+'-'+name.split('-')[2]
#             images.append(patient_name)
#     patients = set(images)
#     # print(type(patients))
#     nums = []
#     for patient in patients:
#         num = images.count(patient)
#         nums.append(num)
#     patients = list(patients)
#     patients = np.array(patients)
#     patients = np.reshape(patients,(patients.shape[0],1))
#     nums = np.array(nums)
#     nums = np.reshape(nums,(nums.shape[0],1))
#
#     results = np.concatenate((patients,nums),axis=1)
#     results = pd.DataFrame(results)
#     results.to_csv('./features/patients_patch_num.csv',header=None,index=False)
# # get_number('./features/cluster_features/')
#
#
# def get_all_num(patient):
#     csv = pd.read_csv('./features/patients_patch_features.csv')
#     csv = np.array(csv)
#     num = 0
#     for i in range(len(csv)):
#         if csv[i,0].split('-')[0]+'-'+csv[i,0].split('-')[1]+'-'+csv[i,0].split('-')[2] == patient:
#             num = int(csv[i,1])
#             break
#     return num


def make_cluster_patient(in_path,out_path):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for feature_path_name in os.listdir(in_path):
        feature_path = in_path+feature_path_name
        feature = pd.read_csv(feature_path,header=None)
        feature = np.array(feature)
        for i in range(int(feature.shape[0])):
            name = feature[i,0].split('-')[0]+'-'+feature[i,0].split('-')[1]+'-'+feature[i,0].split('-')[2]
            feature[i,0] = name

        all_patient_names = feature[:, 0]
        patient_names = set(all_patient_names)
        patient_names = list(patient_names)
        result_features = []
        for patient_name in patient_names:
            patient_feature = []
            patient_num = 0
            for i in range(len(feature)):
                if feature[i, 0] == patient_name:
                    patient_feature.append(feature[i, 1:])
                    patient_num += 1
            patient_feature = np.reshape(patient_feature, (patient_num, 32))
            result_feature = np.mean(patient_feature, axis=0)
            result_features.append(result_feature)
        result_features = np.array(result_features)
        patient_names_array = np.reshape(patient_names, (len(patient_names), 1))
        patient_features = np.concatenate((patient_names_array, result_features), axis=1)
        patient_features_pd = pd.DataFrame(patient_features)
        patient_features_pd.to_csv(out_path+feature_path_name.split('_')[0]+'_patient.csv', header=False, index=None)

# make_cluster_patient('./DCAS_weighted/cluster_image_features/','./DCAS_weighted/cluster_patient_features/')

def make_patient(path):
    all_features = []
    features_num = 0
    for csv_path_name in os.listdir(path):
        csv_path = path + csv_path_name
        csv = pd.read_csv(csv_path, header=None)
        csv = np.array(csv)
        features_num += len(csv)
        all_features.append(csv)
    all_features_patient = np.concatenate(all_features, axis=0)
    all_features_patient_pd = pd.DataFrame(all_features_patient)
    all_features_patient_pd.to_csv('./DCAS_weighted/all_features_patient.csv', header=False, index=False)

    all_patient_names = all_features_patient[:, 0]
    patient_names = set(all_patient_names)
    patient_names = list(patient_names)
    result_features = []
    for patient_name in patient_names:
        patient_feature = []
        patient_num = 0
        for i in range(len(all_features_patient)):
            if all_features_patient[i, 0] == patient_name:
                patient_feature.append(all_features_patient[i, 1:])
                patient_num += 1
        patient_feature = np.reshape(patient_feature, (patient_num, 32))
        result_feature = np.mean(patient_feature, axis=0)
        result_features.append(result_feature)
    result_features = np.array(result_features)
    patient_names_array = np.reshape(patient_names, (len(patient_names), 1))
    patient_features = np.concatenate((patient_names_array, result_features), axis=1)

    patient_features_pd = pd.DataFrame(patient_features)
    patient_features_pd.to_csv('./DCAS_weighted/patient.csv', header=False, index=None)

# make_patient('./DCAS_weighted/cluster_patient_features/')

def make_patient_features_withlabel(features_path,labels_path):
    features = pd.read_csv(features_path,header=None)
    labels = pd.read_csv(labels_path,header=None)
    features = np.array(features)
    labels = np.array(labels)
    labels = labels[1:,:]

    header = []
    header.append('name')
    header.append('time')
    header.append('status')
    for i in range(int(features.shape[1])-1):
        header.append('ov_image_'+str(i+1))
    header = np.array(header)
    header = np.reshape(header,(1,len(header)))
    time = []
    event = []

    for i in range(int(features.shape[0])):
        name1 = features[i,0]
        for j in range(int(labels.shape[0])):
            name2 = labels[j,0]
            if name1==name2:
                time.append(labels[j,1])
                event.append(labels[j,2])
                break
    time = np.array(time)
    time = np.reshape(time,(len(time),1))
    event = np.array(event)
    event = np.reshape(event,(len(event),1))
    name = np.reshape(features[:, 0], (len(features[:, 0]), 1))
    features_withlabel = np.concatenate((name,time,event,features[:,1:]),axis=1)
    features_withlabelheader = np.concatenate((header,features_withlabel),axis=0)
    features_withlabelheader_pd = pd.DataFrame(features_withlabelheader)
    features_withlabelheader_pd.to_csv('./DCAS_weighted/DCAS_patient_features.csv',index=None,header=False)

# make_patient_features_withlabel('./DCAS_weighted/patient.csv','./DCAS_weighted/Clinical_OV_label.csv')

def make_weighted_patient_features(mean_features_path, patient_weight_path,median_cluster_num):
    mean_features = pd.read_csv(mean_features_path,header=None)
    mean_features = np.array(mean_features)
    patient_weight = pd.read_csv(patient_weight_path, header=None)
    patient_weight = np.array(patient_weight)

    weighted_features = []
    # weighted_features.append(mean_features[0])
    for i in range(len(mean_features[1:])):
        weighted_feature = []
        pname = mean_features[i+1][0]
        for j in range(len(patient_weight)):
            if pname == patient_weight[j][0]:
                weight = float(int(patient_weight[j][1]) / median_cluster_num)
                # weight = float(cluster_num / int(patient_weight[j][1]))
                row_feature = []
                for k in range(len(mean_features[i+1,3:])):
                    row_feature.append(float(mean_features[i+1,k+3])*weight)
                row_feature = np.reshape(row_feature,(1,len(row_feature)))
                weighted_feature = row_feature
                break
        weighted_features.append(weighted_feature)
    weighted_features = np.array(weighted_features)
    weighted_features = np.reshape(weighted_features,(len(weighted_features),32))
    weighted_features_withlabel = np.concatenate((mean_features[1:,:3],weighted_features),axis=1)
    header = mean_features[0,:]
    header = np.reshape(header,(1,len(header)))
    weighted_features_withlabelheader = np.concatenate((header,weighted_features_withlabel),axis=0)
    weighted_features_withlabelheader_pd = pd.DataFrame(weighted_features_withlabelheader)
    weighted_features_withlabelheader_pd.to_csv('./DCAS_weighted/DCAS_weighted_patient_features.csv',index=None,header=False)
make_weighted_patient_features('./DCAS_weighted/DCAS_patient_features.csv','./DCAS_weighted/pwithnum.csv',4)
