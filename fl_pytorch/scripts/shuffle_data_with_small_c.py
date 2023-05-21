#!/usr/bin/env python3

import math, sys, pickle, os
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "./../"))

from utils import utils
from data_preprocess import libsvm_dataset

def indepentFeaturesForTrain(dataset):
    n, d = dataset.shape
    eps = 1e-9
    result = []

    for j in range(d):
        vec_norm = np.linalg.norm(trainX[:,j], np.inf)
        if vec_norm < eps:
            result.append(j)
    return result

def bestSplitForDataset(dataset, not_used_features_list):
    n, d = dataset.shape
    eps = 1e-9

    best_split_fraction = 1000.0
    best_split_index = 0

    for j in range(d):
        if j in not_used_features_list:
            continue

        samples = (np.abs(trainX[:,j]) > eps)
        samples_count_with_non_zero_feature = samples.sum()

        # fraction with zero feature
        fraction_of_examples = (n - samples_count_with_non_zero_feature)/n

        if fraction_of_examples < 1.0 - eps:
            if abs(fraction_of_examples - 1.0) < abs(best_split_fraction - 1.0):
                best_split_fraction = fraction_of_examples
                best_split_index = j

    return best_split_index, best_split_fraction

def splitDataset(datasetX, datasetY, split_feature):
    n, d = datasetX.shape

    eps = 1e-9
    samples = (np.abs(datasetX[:, split_feature]) > eps)

    samples_count_with_non_zero_feature = samples.sum()
    samples_count_with_zero_feature = n - samples_count_with_non_zero_feature

    dataset_with_non_zero_feature = np.zeros( (samples_count_with_non_zero_feature, d), datasetX.dtype)
    dataset_with_zero_feature     = np.zeros( (samples_count_with_zero_feature, d), datasetX.dtype)

    dataset_with_non_zero_feature_Y = np.zeros( (samples_count_with_non_zero_feature, 1), datasetY.dtype)
    dataset_with_zero_feature_Y     = np.zeros( (samples_count_with_zero_feature, 1), datasetY.dtype)

    non_zero_write_head = 0
    zero_write_head = 0

    for i in range(n):
        if abs(datasetX[i, split_feature]) > eps:
            dataset_with_non_zero_feature[non_zero_write_head, :] = datasetX[i, :]
            dataset_with_non_zero_feature_Y[non_zero_write_head, :] = datasetY[i, :]

            non_zero_write_head += 1
        else:
            dataset_with_zero_feature[zero_write_head, :]   = datasetX[i, :]
            dataset_with_zero_feature_Y[zero_write_head, :] = datasetY[i, :]

            zero_write_head += 1

    return dataset_with_non_zero_feature, dataset_with_non_zero_feature_Y, dataset_with_zero_feature, dataset_with_zero_feature_Y

def zeroOutFeaures(datasetX, features):
    for feature_index in features:
        datasetX[:, feature_index] = 0.0

def concatDatasets(datasets, labels):
    assert len(datasets) == len(labels)

    samples_total = sum([datasets[i].shape[0] for i in range(len(datasets))])
    d = datasets[0].shape[1]

    dataset_union = np.zeros( (samples_total, d), datasets[0].dtype)
    labels_union  = np.zeros( (samples_total, labels[0].shape[1]), labels[0].dtype)

    write_pos = 0

    for i in range(len(datasets)):
        ds = datasets[i]
        lbl = labels[i]

        for j in range(ds.shape[0]):
            dataset_union[write_pos, :] = ds[j, :]
            labels_union[write_pos, :] = lbl[j, :]
            write_pos += 1

    return dataset_union, labels_union


#0 1:0.18257 2:0.18257 3:0.18257 4:0.18257 5:0.18257 6:0.18257 7:0.18257 8:0.18257 9:0.18257 10:0.18257 11:0.18257 12:0.18257 13:0.18257 14:0.18257 15:0.18257 16:0.18257 17:0.18257 18:0.18257 19:0.18257 20:0.18257 21:0.18257 22:0.18257 23:0.18257 24:0.18257 25:0.18257 26:0.18257 27:0.18257 28:0.18257 29:0.18257 30:0.18257
def writeDatasetToFile(datasetX, datasetY, fname, ignore_features_for_write):
    samples = datasetX.shape[0]
    D = datasetX.shape[1]
    eps = 1e-9

    with open(fname, "w") as f_out:
        for i in range(samples):
            # Write target
            f_out.write(str(int(datasetY[i,0])))

            for d in range(D):
                if d in ignore_features_for_write:
                    continue

                featureValue = datasetX[i, d]
                if abs(featureValue) < eps:
                    continue
                else:
                    f_out.write(f' {d}:{featureValue}')
            f_out.write('\n')


def compute_sigma_for_Z_prime(datasetX, datasetY, clients, ignore_features, plot_c = True):
    samplesTotal = datasetX.shape[0]
    D = datasetX.shape[1]
    samplesPerClient = samplesTotal // clients

    not_Z_prime = np.zeros((clients, D))
    eps = 1e-9

    for c in range(clients):
        subdata = datasetX[int(c) * samplesPerClient: int(c+1) * samplesPerClient, ...]

        for j in range(D):
            j_feature_Linf_norm = np.abs(subdata[:, j]).max()

            if j_feature_Linf_norm < eps:
                # For all samples j_features is close to Zero
                # => (c,j) \in Z => (c,j) not in not(Z)
                not_Z_prime[c, j] = 0.0
            else:
                # For some samples j_features is not Zero
                # We assume it's possibly imply that (c,j) is not in Z.
                #  (c,j) \in not(Z)
                not_Z_prime[c, j] = 1.0

    #================================================================================================================
    # Maximum cardinality by columns via looking into not Z'
    max_card = 0.0
    for j in range(D):
        cardinality_j = not_Z_prime[:, j].sum()
        if cardinality_j > max_card:
            max_card = cardinality_j

    sigma_For_Zprime = max_card
    if plot_c:
        figsize = (12, 12)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.spy(not_Z_prime, precision=0.1, markersize=5)

        fig.tight_layout()
        ax.set_title("test")
        ax.grid(True)
        plt.show()

        pass

if __name__ == "__main__":
    
    fname = R"C:\projects\test_\fl_pytorch\fl_pytorch\data\phishing"
    fname_out = R"C:\projects\test_\fl_pytorch\fl_pytorch\data\phishing_new"

    includeBias = True

    print("Read data from: ", fname)

    total_features_train, targets_train = libsvm_dataset.analyzeDataset(fname)
    total_samples_train = sum(targets_train.values())
    features = total_features_train

    # Extra offset for maybe features are indexed from #0
    features += 1  

    if includeBias:
        features += 1

    print("  Number of samples: ", total_samples_train)
    print("  Number of features: ", features)
    print("  Add bias: ", includeBias)

    keys = list(targets_train.keys())
    reinterpretate_labels = {keys[0]: 0.0, keys[1]: 1.0}

    # Read data from file
    trainX, trainY = libsvm_dataset.readSparseInput(fname, features, total_samples_train, reinterpretate_labels, fieldSeparator='\t', featureNameValueSeparator=':', includeBias = includeBias)
    # ind = indepentFeaturesForTrain(trainX)

    dataset_list = []
    dataset_Y_list = []

    not_used_features_list = []

    residual_X = trainX
    residual_Y = trainY

    for i in range(trainX.shape[1]):
        splitFeature, splitFraction = bestSplitForDataset(residual_X, not_used_features_list)
        dataset_with_non_zero_feature, dataset_with_non_zero_feature_Y, dataset_with_zero_feature, dataset_with_zero_feature_Y = splitDataset(residual_X, residual_Y, splitFeature)

        if dataset_with_non_zero_feature.shape[0] == 0 or dataset_with_zero_feature.shape[0] == 0:
            break

        not_used_features_list.append(splitFeature)

        dataset_list.append(dataset_with_non_zero_feature)
        dataset_Y_list.append(dataset_with_non_zero_feature_Y)

        residual_X = dataset_with_zero_feature
        residual_Y = dataset_with_zero_feature_Y

    #===================================================================================================================
    last_features = []
    last_dataset_list = [residual_X]
    last_dataset_Y_list = [residual_Y]

    for i in range(trainX.shape[1]):
        if i not in not_used_features_list:
            last_features.append(i)

    length_a_x = sum([dataset_list[i].shape[0] for i in range(len(dataset_list))])
    length_a_y = sum([dataset_Y_list[i].shape[0] for i in range(len(dataset_Y_list))])

    length_b_x = sum([last_dataset_list[i].shape[0] for i in range(len(last_dataset_list))])
    length_b_y = sum([last_dataset_Y_list[i].shape[0] for i in range(len(last_dataset_Y_list))])

    assert length_a_x == length_a_y
    assert length_b_x == length_b_y
    assert length_a_x + length_b_x == trainX.shape[0]
    #===================================================================================================================
    # Concatenate dataset
    # ===================================================================================================================
    dataset_num = len(dataset_list)
    # writeDatasetToFile(union_x, union_y, fname_out, ignore_features_for_write = [0])
    print("Features")
    print(not_used_features_list)

    for i in range(1, dataset_num):
        union_x, union_y = concatDatasets(dataset_list[:i], dataset_Y_list[:i])

        zeroOutFeaures(union_x, last_features)
        zeroOutFeaures(union_x, not_used_features_list[i:])

        print("Feature split")
        print(not_used_features_list[i-1])

        compute_sigma_for_Z_prime(union_x, union_y, clients=10, ignore_features=[], plot_c=True)
