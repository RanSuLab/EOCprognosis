Files introduction:
extract_patches.py is for extracting patches from WSIs.
clusterByKmeans.py is for thumbails generation and clustering by K-means.
split_train_test.py is for train set and test set splitting.
DCAS.py is for training in cluster and patch-level feature extraction.
DCAS_weighted.py is for generating weighted patient-level features.
make10folds.py is for making 10-fold files.
cox_lasso_10folds.R is for final survival analysis with LASSO-Cox model.
Clinical_OV_label.csv contains id of patients and their survival time and status .  

You need download the WSIs and clinical data from TCGA-OV. You need obtain survival status(0 for alive and 1 for dead) and time of patients in the clinical data.  
And you need execute files in the order as follows to finish EOCSA framework.
(extract_patches.py->clusterByKmeans.py->split_train_test.py->DCAS.py->DCAS_weighted.py->make10folds.py->cox_lasso_10folds.R)

Details:
In extract_patches.py, you can use function 'write_path_to_csv' to write the path of WSIs to a csv file and use function 'get_patches' to extract patches from WSIs.
In clusterByKmeans.py, you can use function 'make_thumbnail' to make thumbails of patches, and then, use function 'pca_feature' to reduce feature dimension, and at last, use function 'k_means' and 'cluster' to obtain result after cluster.
In split_train_test.py, you can use function 'get_patch_path_withlabel' to obtain pathes and labels of patches in each cluster and then, use function 'split_train_test' to split train set and test set for DCAS model training.
In DCAS.py, you can use function 'train' to train model with train set and use function 'evaluate' to evaluste the model in both train set and test set and use function 'extract_features' to extract patch-level features from selected clusters.
In DCAS_weighted.py, you can obtain weighted patient-level features by executing functions in the order(make_cluster_patient,make_patientmake_patient_features_withlabel,make_weighted_patient_features).
In make10folds.py, you can use function 'make10folds'  to obtain 10-fold files for final survival analysis.
In cox_lasso_10folds.R, you can obtain final survival results.

 
Environment: Python 3.5.2 and R 3.5.3.

If you have any questions, please contact dling@tju.edu.cn!
