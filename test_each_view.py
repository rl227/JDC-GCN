import torch
import torch.nn as nn
import torch.nn.functional as F
from model.GCN import JdcGCN, GCN
from model.GCN import *
# from model import GCN
import copy
import os
import numpy as np
import scipy.io as sio
from utils.loadData import loadMatData
from utils.loadAdj import load_adjacency_multiview
from utils.dataProcess import generate_permutation, feature_normalization, common_loss

import math
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from utils.dataProcess import frobenius_loss
import time
import random
import sys


def test_each(dataset_name, direction, k_value, ratio_value, device):
    seed = 2016
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset_path = direction + dataset_name
    rep_num = 1
    max_iter = 500
    random_ratio = ratio_value

    features, gnd = loadMatData(dataset_path)
    features, gnd, p_labeled, p_unlabeled = generate_permutation(features, gnd, random_ratio)
    adjs, adj_hats, com_adj = load_adjacency_multiview(features, dataset_name, k=k_value, rep=1)
    # normalize features
    features = feature_normalization(features)
    num_class = np.unique(gnd).shape[0]  # number of classes
    view_num = len(adjs)
    N = gnd.shape[0]  # number of samples
    labels = torch.zeros((N, num_class)).to(device)
    for index in range(N):
        labels[index, gnd[index]] = 1
    gnd = torch.from_numpy(gnd).long().to(device)

    # GCN settings
    fea = []
    view = []
    sum = 0
    for v in range(view_num):
        fea.append(torch.tensor(features[0, v], dtype=torch.float32).to(device))
        view.append(features[0, v].shape[1])
        sum += features[0, v].shape[1]
    view.append(sum)
    com_fea = fea[0]
    for i in range(view_num - 1):
        com_fea = torch.cat((com_fea, fea[i + 1]), dim=1)

    dim_out = num_class
    adj_hats = torch.tensor(adj_hats, dtype=torch.float32).float().to(device)
    com_adj = torch.tensor(com_adj, dtype=torch.float32).float().to(device)
    com_adj = com_adj.unsqueeze(0)
    adj_hats = torch.cat([adj_hats, com_adj], dim=0)
    fea.append(com_fea)
    for inputI in range(len(view)):
        ratio_perf_list = []
        time_cost = []
        for rep in range(rep_num):
            model = GCN(adj_hats[inputI], view[inputI], dim_out).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            # Training settings
            best_accuracy = 0.0
            # Identity_input = torch.eye(N, dim_in).to(device)
            # Begin training
            start_time = time.time()
            tag = 0
            # com_fea = com_fea.to(device)
            with tqdm(total=max_iter, desc="Training", ncols=120) as pbar:
                for i in range(max_iter):
                    emb = model(fea[inputI])
                    y_pred = emb
                    loss = F.cross_entropy(y_pred[p_labeled], gnd[p_labeled])
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    pred_label = torch.argmax(F.log_softmax(y_pred, 1), 1)
                    accuracy_value = accuracy_score(gnd[p_unlabeled].cpu().detach().numpy(),
                                                    pred_label[p_unlabeled].cpu().detach().numpy())
                    if accuracy_value > best_accuracy:
                        tag = 0
                        best_accuracy = accuracy_value
                    else:
                        tag += 1
                    pbar.set_postfix({'Loss_GCN': '{0:1.5f}'.format(loss),
                                      'Cur Accuracy': '{0:1.5f}'.format(accuracy_value),
                                      'Best Accuracy': '{0:1.5f}'.format(best_accuracy)})
                    pbar.update(1)
            end_time = time.time()
            time_cost.append(end_time - start_time)
            ratio_perf_list.append(accuracy_value)
        avg_perf = np.around(np.mean(ratio_perf_list), decimals=4)
        std_perf = np.around(np.std(ratio_perf_list), decimals=4)
        avg_time = np.around(np.mean(time_cost), decimals=4)
        print(dataset_name, "view=", inputI + 1, "ratio =", str(ratio_value), "Avg perf:", avg_perf, "Std:", std_perf,
              "Time cost:", avg_time)
    # return avg_perf, std_perf, avg_time
