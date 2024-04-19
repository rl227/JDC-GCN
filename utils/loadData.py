'''
Author: Zhaoliang Chen
Date: 2020-09-01 16:47:29
LastEditTime: 2020-09-04 15:18:30
LastEditors: Zhaoliang Chen
Description: Data loading tools
chenzl23@outlook.com
'''


import scipy.io
import os
import numpy as np

### Load feature and ground truth data with .mat
def loadMatData(data_name):
    data = scipy.io.loadmat(data_name) 
    features = data['X']#.dtype = 'float32'
    gnd = data['Y']
    gnd = gnd.flatten()
    if min(gnd) == 1:
        gnd = gnd - 1
    return features, gnd

### Load similarity data with .mat
def loadSIM(data_name):
    data = scipy.io.loadmat(data_name)
    #print(data.keys())
    similaritis = data['W']
    return similaritis

def loadData_combinedWeight(dataset_name):
    '''
    Load multi-view data with a pre-calculated combined weight for all views.
    (Weight is computed by matlab codes)
    '''
    features, gnd = loadMatData(os.path.join("data",dataset_name+".mat"))
    W = loadSIM(os.path.join("data",dataset_name+"W.mat"))
    return features, gnd, W




