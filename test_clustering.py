import torch
import torch.nn as nn
import torch.nn.functional as F
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
from utils.clusteringPerformance2 import *
from sklearn.cluster import KMeans


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def test_clustering(dataset_name, direction, k_value, ratio_value, device):
    seed = 2016
    kl_gamma = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset_path = direction + dataset_name
    rep_num = 1
    max_iter = 1000
    random_ratio = ratio_value
    ratio_perf_list = []
    time_cost = []
    for rep in range(rep_num):
        features, gnd = loadMatData(dataset_path)
        features, gnd, p_labeled, p_unlabeled = generate_permutation(features, gnd, random_ratio)
        adjs, adj_hats, com_adj = load_adjacency_multiview(features, dataset_name, k=k_value, rep=rep)
        # normalize features
        features = feature_normalization(features)
        view_dims = [features[0, v].shape[1] for v in range(len(features[0]))]
        num_class = np.unique(gnd).shape[0]  # number of classes
        view_num = len(adjs)
        n = gnd.shape[0]  # number of samples
        labels = torch.zeros((n, num_class)).to(device)
        for index in range(n):
            labels[index, gnd[index]] = 1
        gnd = torch.from_numpy(gnd).long().to(device)
        # GCN settings
        fea = []
        view = []
        for v in range(view_num):
            fea.append(torch.tensor(features[0, v], dtype=torch.float32).to(device))
            view.append(features[0, v].shape[1])

        com_fea = fea[0]
        for i in range(view_num - 1):
            com_fea = torch.cat((com_fea, fea[i + 1]), dim=1)

        dim_out = 32
        adj_hats = torch.tensor(adj_hats, dtype=torch.float32).float().to(device)
        com_adj = torch.tensor(com_adj, dtype=torch.float32).float().to(device)
        model = ClusteringJdcGCN(adj_list=adj_hats, com_adj=com_adj, view=view, dim_out=dim_out, n_clusters=num_class,
                                 n_enc_1=50, n_enc_2=50, n_enc_3=100, n_dec_1=100, n_dec_2=50, n_dec_3=50).to(
            device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        start_time = time.time()
        com_fea = com_fea.to(device)
        mse_loss = nn.MSELoss()
        alpha = 0.95
        with tqdm(total=max_iter, desc="Training", ncols=120) as pbar:
            for i in range(max_iter):
                # emb_list: Diverse-GCN和Com-GCN输出的特征
                # emb     : attention之后的特征即输入编码器的特征
                # rec_emb : 解码器输出的特征
                # z       : 自编码器的聚类层
                # att     : 注意力机制的权重的值
                emb_list, emb, rec_emb, z, att = model(fea, com_fea)
                loss = alpha * mse_loss(emb, rec_emb)
                for view in range(len(emb_list)):
                    loss += (1 - alpha) * mse_loss(emb_list[view], rec_emb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                kmeans = KMeans(n_clusters=num_class, n_init=20)
                y_pred = kmeans.fit_predict(z.data.cpu().numpy())
                # y_pred = z
                # pred_label = torch.argmax(F.log_softmax(y_pred, 1), 1)
                accuracy_value, nmi, purity, ari, f_score, precision, recall = clusteringMetrics(
                    trueLabel=gnd.cpu().numpy(), predictiveLabel=y_pred)
                pbar.set_postfix({'Loss_GCN': '{0:1.5f}'.format(loss),
                                  'ACC': '{0:1.5f}'.format(accuracy_value),
                                  'NMI': '{0:1.5f}'.format(nmi),
                                  'ARI': '{0:1.5f}'.format(ari)})
                pbar.update(1)
        # print("Att:", att.shape)
        # model.visualization_loss()
        end_time = time.time()
        time_cost.append(end_time - start_time)
        ratio_perf_list.append(accuracy_value)
    avg_perf = np.around(np.mean(ratio_perf_list), decimals=4)
    std_perf = np.around(np.std(ratio_perf_list), decimals=4)
    avg_time = np.around(np.mean(time_cost), decimals=4)
    print(dataset_name, "ratio =", str(ratio_value), "Avg perf:", avg_perf, "Std:", std_perf, "Time cost:", avg_time)
    return avg_perf, std_perf, avg_time
