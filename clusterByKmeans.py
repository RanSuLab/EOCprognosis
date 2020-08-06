from PIL import Image
import Image
import os
import numpy as np
from sklearn.decomposition import PCA
import xlsxwriter
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def make_thumbnail(path,new_path):
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    for first_path_name in os.listdir(path):
        first_path = path+first_path_name
        for image_name in os.listdir(first_path):
            image_path = first_path+'/'+image_name
            image = Image.open(image_path)
            image.thumbnail((50,50))
            image.save(new_path+image_name)
# make_thumbnail('J:/Dling/Ovarian Cancer/patches/cancer/','J:/Dling/Ovarian Cancer/cancer_thumb/')

def pca_feature(path,pca_feature_size):
    features = []
    image_names = []
    ori_features_size = 0
    for image_name in os.listdir(path):
        image_path = path+image_name
        image = Image.open(image_path)
        image = np.array(image)
        new_image = np.zeros((image.shape[0], image.shape[1]))
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                new_image[i, j] = np.mean((image[i, j, 0], image[i, j, 1], image[i, j, 2]))
        ori_features_size = image.shape[0]*image.shape[1]
        new_image_array = new_image.reshape((ori_features_size,1))
        features.append(new_image_array)
        image_names.append(image_name)
    image_names = np.array(image_names)
    image_names = image_names.reshape((image_names.shape[0],1))
    features = np.array(features)
    features = features.reshape((features.shape[0],ori_features_size))
    pca = PCA(n_components=pca_feature_size)
    pca.fit(features)
    pca_features = pca.fit_transform(features)
    pca_features_with_label = np.concatenate((image_names,pca_features),axis=1)
    pca_features_with_label = pd.DataFrame(pca_features_with_label)
    pca_features_with_label.to_csv('./pca_features.csv',index=False,header=False)
# pca_feature('J:/Dling/Ovarian Cancer/cancer_thumb/',50)

def k_means(csv_path,cancer_image_path):
    pca_features_with_label = pd.read_csv(csv_path,header=None)
    pca_features_with_label = np.array(pca_features_with_label)
    pca_features = pca_features_with_label[:,1:]
    labels = pca_features_with_label[:,0]
    clf_KMeans = KMeans(n_clusters=10)
    cluster = clf_KMeans.fit_predict(pca_features)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=cluster)
    plt.show()
    cluster_results = []
    for i in range(len(labels)):
        line = []
        image_path = labels[i]
        last_image_path = image_path.split('_')[0]
        path = cancer_image_path+last_image_path+'/'+labels[i]
        line.append(path)
        line.append(cluster[i])
        line = np.array(line)
        cluster_results.append(line)
    # cluster_results = np.array(cluster_results)
    cluster_results = pd.DataFrame(cluster_results)
    cluster_results.to_csv('./cluster_results.csv',index=False,header=False)
# k_means('./pca_features.csv','J:/Dling/Ovarian Cancer/patches/cancer/')

def cluster(cav_path,new_path):
    cluster_result = pd.read_csv(cav_path,header=None)
    cluster_result = np.array(cluster_result)
    for i in range(cluster_result.shape[0]):
        image_path = cluster_result[i][0]
        image_name = image_path.split('/')[-1]
        cluster_num = cluster_result[i][1]
        cluster_path = new_path+str(cluster_num)+'/'
        if not os.path.exists(cluster_path):
            os.makedirs(cluster_path)
        image = Image.open(image_path)
        new_image_path = cluster_path+image_name
        image.save(new_image_path)
# cluster('./cluster_results.csv','J:/Dling/Ovarian Cancer/cancer_cluster/')




