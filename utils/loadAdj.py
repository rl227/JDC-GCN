import os
import pdb
import time
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale, maxabs_scale, normalize, robust_scale, scale
from utils.loadData import loadMatData


def load_adj(features, normalization=True, normalization_type='normalize',
              k_nearest_neighobrs=10, prunning_one=False, prunning_two=True , common_neighbors=2):
    if normalization:
        if normalization_type == 'minmax_scale':
            features = minmax_scale(features)
        elif normalization_type == 'maxabs_scale':
            features = maxabs_scale(features)
        elif normalization_type == 'normalize':
            features = normalize(features)
        elif normalization_type == 'robust_scale':
            features = robust_scale(features)
        elif normalization_type == 'scale':
            features = scale(features)
        elif normalization_type == '255':
            features = np.divide(features, 255.)
        elif normalization_type == '50':
            features = np.divide(features, 50.)
        else:
            print("Please enter a correct normalization type!")
            pdb.set_trace()

    # construct three kinds of adjacency matrix

    adj, adj_wave, adj_hat = construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one,
                                                        prunning_two, common_neighbors)
    return adj, adj_hat


def construct_adjacency_matrix(features, k_nearest_neighobrs, prunning_one, prunning_two, common_neighbors):
    start_time = time.time()
    nbrs = NearestNeighbors(n_neighbors=k_nearest_neighobrs+1, algorithm='ball_tree').fit(features)
    adj_wave = nbrs.kneighbors_graph(features)  # <class 'scipy.sparse.csr.csr_matrix'>

    if prunning_one:
        # Pruning strategy 1
        original_adj_wave = adj_wave.A
        judges_matrix = original_adj_wave == original_adj_wave.T
        np_adj_wave = original_adj_wave * judges_matrix
        adj_wave = sp.csc_matrix(np_adj_wave)
    else:
        # transform the matrix to be symmetric (Instead of Pruning strategy 1)
        np_adj_wave = construct_symmetric_matrix(adj_wave.A)
        adj_wave = sp.csc_matrix(np_adj_wave)

    # obtain the adjacency matrix without self-connection
    # np_adj_wave = np.ones(np_adj_wave.shape[0]) - np_adj_wave
    adj = sp.csc_matrix(np_adj_wave)
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()

    if prunning_two:
        # Pruning strategy 2
        adj = adj.A
        b = np.nonzero(adj)
        rows = b[0]
        cols = b[1]
        dic = {}
        for row, col in zip(rows, cols):
            if row in dic.keys():
                dic[row].append(col)
            else:
                dic[row] = []
                dic[row].append(col)
        for row, col in zip(rows, cols):
            if len(set(dic[row]) & set(dic[col])) < common_neighbors:
                adj[row][col] = 0
        adj = sp.csc_matrix(adj)
        adj.eliminate_zeros()

    # construct the adjacency hat matrix
    adj_hat = construct_adjacency_hat(adj)  # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    # print("The construction of adjacency matrix is finished!")
    # print("The time cost of construction: ", time.time() - start_time)

    return adj, adj_wave, adj_hat


def construct_adjacency_hat(adj):
    """
    :param adj: original adjacency matrix  <class 'scipy.sparse.csr.csr_matrix'>
    :return:
    """
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))  # <class 'numpy.ndarray'> (n_samples, 1)
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    # <class 'scipy.sparse.coo.coo_matrix'>  (n_samples, n_samples)
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return adj_normalized


def construct_symmetric_matrix(original_matrix):
    """
        transform a matrix (n*n) to be symmetric
    :param np_matrix: <class 'numpy.ndarray'>
    :return: result_matrix: <class 'numpy.ndarray'>
    """
    result_matrix = np.zeros(original_matrix.shape, dtype=float)
    num = original_matrix.shape[0]
    for i in range(num):
        for j in range(num):
            if original_matrix[i][j] == 0:
                continue
            elif original_matrix[i][j] == 1:
                result_matrix[i][j] = 1
                result_matrix[j][i] = 1
            else:
                print("The value in the original matrix is illegal!")
                pdb.set_trace()
    assert (result_matrix == result_matrix.T).all() == True

    if ~(np.sum(result_matrix, axis=1) > 1).all():
        print("There existing a outlier!")
        pdb.set_trace()

    return result_matrix


def construct_sparse_float_tensor(np_matrix):
    """
        construct a sparse float tensor according a numpy matrix
    :param np_matrix: <class 'numpy.ndarray'>
    :return: torch.sparse.FloatTensor
    """
    sp_matrix = sp.csc_matrix(np_matrix)
    three_tuple = sparse_to_tuple(sp_matrix)
    sparse_tensor = torch.sparse.FloatTensor(torch.LongTensor(three_tuple[0].T),
                                        torch.FloatTensor(three_tuple[1]),
                                        torch.Size(three_tuple[2]))
    return sparse_tensor


def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()

    # sparse_mx.row/sparse_mx.col  <class 'numpy.ndarray'> [   0    0    0 ... 2687 2694 2706]
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()  # <class 'numpy.ndarray'> (n_edges, 2)
    values = sparse_mx.data  # <class 'numpy.ndarray'> (n_edges,) [1 1 1 ... 1 1 1]
    shape = sparse_mx.shape  # <class 'tuple'>  (n_samples, n_samples)
    return coords, values, shape

def load_adjacency_multiview(multi_view_features, dataset_name, k, rep):
    adj_list = []
    adj_hat_list = []
    # dataset_direction = os.path.join("/data/yaoj/mvc_datasets/adj_matrix", dataset_name)
    # if not os.path.exists(dataset_direction):
    #     os.mkdir(dataset_direction)
    # knn_direction = os.path.join(dataset_direction, str(k)+'-nn')
    # if not os.path.exists(knn_direction):
    #     os.mkdir(knn_direction)
    # save_direction = os.path.join(knn_direction, 'Rand-' + str(rep + 1))
    # if not os.path.exists(save_direction):
    #     os.mkdir(save_direction)


    # construct three kinds of adjacency matrix
    print("Constructing the adjacency matrix of " + dataset_name)
    for idx, features in enumerate(multi_view_features[0]):
        adj, adj_hat = load_adj(features, k_nearest_neighobrs=k)
        adj_list.append(adj.todense())
        adj_hat_list.append(adj_hat.todense())

    adj_list = np.array(adj_list)
    adj_hat_list = np.array(adj_hat_list)
    com_adj = None
    for i in range(len(adj_list)):
        if com_adj is None:
            com_adj = adj_list[i]
        else:
            com_adj = np.maximum(com_adj, adj_list[i])
    com_adj = construct_adjacency_hat(com_adj).todense()
    # save these scale and matrix
    # print("Saving the adjacency matrix to " + save_direction)
    # np.save(os.path.join(save_direction, 'adj'), adj_list)
    # np.save(os.path.join(save_direction, 'adj_hat'), adj_hat_list)
    # np.save(os.path.join(save_direction, 'com_adj'), com_adj)
    return adj_list, adj_hat_list, com_adj

if __name__ == '__main__':
    start_time = time.time()
    direction_path = './data'
    dataset_name = 'ALOI'
    features, gnd = loadMatData(os.path.join(direction_path, dataset_name))
    # print(features[0][0].shape, type(features[0][0])) # features[0][view_num]
    # adj = load_adj(features[0][0], dataset_name, load_saved = False)
    load_adjacency_multiview(features, dataset_name)
    # print(type(adj))
    print(time.time() - start_time)