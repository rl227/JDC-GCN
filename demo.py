from test_classification import test_ratio
from test_clustering import *
from test_each_view import test_each
from xlutils.copy import copy
# import matplotlib.pyplot as plt
# import numpy as np
import xlrd
import torch

# workbook = xlrd.open_workbook('/data/yaoj/mvc_datasets/results.xls')
# copybook = copy(workbook)
# bookSheet = copybook.get_sheet(0)
# bookSheet.write(0, 0, "Dataset")
# bookSheet.write(1, 0, "ACC")
# bookSheet.write(2, 0, "Std")
# bookSheet.write(3, 0, "Time(s)")

USE_CUDA = True
device = torch.device("cpu")

if USE_CUDA and torch.cuda.is_available():
    USE_CUDA = True
    device = torch.device("cuda:2")
    print("CUDA activated!")
else:
    USE_CUDA = False
    print("CPU activated!")
direction = '/data/yaoj/mvc_datasets/'
datasets = {1: '3Source', 2: 'ALOI', 3: 'Caltech101-7', 4: 'Caltech101-20', 5: 'Caltech101-all', 6: 'Caltech101-all',
            7: 'COIL_4view', 8: 'flower17', 9: 'HW', 10: 'HW2', 11: 'MITIndoor', 12: 'MSRC-v1', 13: 'NUS-WIDE',
            14: 'scene15', 15: 'Wikipedia', 16: 'MNIST10k', 17: 'Youtube', 18: 'MNIST30K', }
test_list = [13]
# k of KNN-graph
k = 1

for i in test_list:
    dataset_id = i
    # for k in range(1, 11):
    # print("Testing:", datasets[i], "k =", str(k))
    for ratio in range(2, 3):
        avg_perf, std_perf, TCost = test_ratio(datasets[dataset_id], direction, k, ratio_value=0.05 * ratio,
                                                    device=device)  # ratio_value: ratio of labeled samples
        # test_each(datasets[dataset_id], direction, k, ratio_value=0.05 * ratio, device=device)
        # bookSheet.write(0, k, k)
        # bookSheet.write(1, k, avg_perf)
        # bookSheet.write(2, k, std_perf)
        # bookSheet.write(3, k, TCost)
        # copybook.save('results.xls')
        # bookSheet.write(0, ratio, ratio * 0.05)
        # bookSheet.write(1, ratio, avg_perf)
        # bookSheet.write(2, ratio, std_perf)
        # bookSheet.write(3, ratio, TCost)
        # copybook.save('results.xls')
