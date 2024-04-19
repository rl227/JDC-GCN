import numpy as np
import math
from utils.loadData import loadMatData
import os
import torch
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale


def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2) ** 2)
    return cost


def frobenius_loss(y_pred, true_label):
    cov1 = torch.matmul(y_pred, y_pred.t())
    cov2 = torch.matmul(true_label, true_label.t())
    cost = torch.mean((cov1 - cov2) ** 2)
    return cost


def Adj_normalize(Adj):
    """
    Symmetrically normalize adjacency matrix.
    """
    rowsum = np.array(Adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    A_hat = Adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return A_hat


def preprocess_adj(Adj):
    """
    Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation.
    """
    adj_normalized = Adj_normalize(Adj + np.eye(Adj.shape[0]))
    return adj_normalized


def count_each_class_num(gnd):
    '''
    Count the number of samples in each class
    '''
    count_dict = {}
    for label in gnd:
        if label in count_dict.keys():
            count_dict[label] += 1
        else:
            count_dict[label] = 1
    return count_dict


def generate_permutation(features, gnd, ratio):
    '''
    Generate permutation for training (labeled) and testing (unlabeled) data.
    '''

    N = len(gnd)

    each_class_num = count_each_class_num(gnd)
    labeled_each_class_num = {}  ## number of labeled samples for each class
    for label in each_class_num.keys():
        labeled_each_class_num[label] = max(round(each_class_num[label] * ratio), 1)  # min is 1

    # shuffle
    perm = np.random.permutation(N)
    # print(perm)
    gnd = gnd[perm]
    for idx, fea in enumerate(features[0]):
        features[0][idx] = fea[perm]

    # index of labeled and unlabeled samples
    p_labeled = []
    p_unlabeled = []
    for idx, label in enumerate(gnd):
        if (labeled_each_class_num[label] > 0):
            labeled_each_class_num[label] -= 1
            p_labeled.append(idx)
        else:
            p_unlabeled.append(idx)

    return features, gnd, p_labeled, p_unlabeled


def feature_normalization(features, normalization_type='normalize'):
    for idx, fea in enumerate(features[0]):
        if normalization_type == 'minmax_scale':
            features[0][idx] = minmax_scale(fea)
        elif normalization_type == 'maxabs_scale':
            features[0][idx] = maxabs_scale(fea)
        elif normalization_type == 'normalize':
            features[0][idx] = normalize(fea)
        elif normalization_type == 'robust_scale':
            features[0][idx] = robust_scale(fea)
        elif normalization_type == 'scale':
            features[0][idx] = scale(fea)
        elif normalization_type == '255':
            features[0][idx] = np.divide(fea, 255.)
        elif normalization_type == '50':
            features[0][idx] = np.divide(fea, 50.)
        else:
            print("Please enter a correct normalization type!")
            pdb.set_trace()
    return features

