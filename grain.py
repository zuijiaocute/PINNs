import torch
from torch.utils.data import Dataset,DataLoader
import random
from collections import OrderedDict

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch.nn as nn
from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings


# the deep neural network
class Grain_Dataset(Dataset):
    """
    :grain_dataset:  数据集文件夹所在位置，文件夹里包含的文件里面是每个时刻的所有空间点以及它们的温度
    """
    def __init__(self,grain_dataset):
        super().__init__()

        #确定文件路径
        self.full_path = []
        for item in os.listdir(grain_dataset):
            self.full_path.append(os.path.join(grain_dataset,item))

        #确定中数据集数量
        self.nums_of_files = len(grain_dataset)
        self.items_in_one_file = len(np.load(self.full_path[0]))
        self.total_num =  self.nums_of_files * self.items_in_one_file
        
    def __getitem__(self, item):
        #item是从0开始的，要对应回去确认每个item所在位置
        item_order = (item + 1) // self.items_in_one_file
        item_order_in_file = (item + 1) % self.items_in_one_file - 1
        item_loacate_path = self.full_path[item_order]
        item = np.load(item_loacate_path)[item_order_in_file]

        X = torch.from_numpy(item[0:4], requires_grad = True)
        h = torch.from_numpy(item[4], requires_grad = True)

        return X,h
  
    def __len__(self):
        return self.total_num