import torch
from torch.utils.data import Dataset,DataLoader
import random
from collections import OrderedDict
from itertools import cycle

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import warnings


def random_choice(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def is_cuda():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device

class Grain_Dataset(Dataset):
    """
    :grain_dataset:  数据集文件夹所在位置，文件夹里包含的文件里面是每个时刻的所有空间点以及它们的温度
    """

    # to do list  对于服务器直接在init中加载好np数据集应该可以提速，可以测试一下
    def __init__(self,grain_dataset):
        super().__init__()

        #确定文件路径
        self.full_path = []
        for item in os.listdir(grain_dataset):
            self.full_path.append(os.path.join(grain_dataset,item))

        #确定中数据集数量
        self.nums_of_files = len(self.full_path)
        self.items_in_one_file = len(np.load(self.full_path[0]))
        self.total_num =  self.nums_of_files * self.items_in_one_file
        
    def __getitem__(self, item):
        #item是从0开始的，要对应回去确认每个item所在位置
        item_order = item // self.items_in_one_file
        item_order_in_file = item % self.items_in_one_file
        item_loacate_path = self.full_path[item_order]
        item = np.load(item_loacate_path)[item_order_in_file]

        x = torch.tensor(item[0:1],requires_grad = True)
        y = torch.tensor(item[1:2],requires_grad = True)
        z = torch.tensor(item[2:3],requires_grad = True)
        t = torch.tensor(item[3:4], requires_grad = True)
        h = torch.tensor(item[4:], requires_grad = True)

        return x, y, z, t, h
  
    def __len__(self):
        return self.total_num

class DNN(nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()
        
        # parameters
        self.depth = len(layers) - 1
        
        # set up layer order dict
        self.activation = torch.nn.Tanh
        
        layer_list = list()
        for i in range(self.depth - 1): 
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)
        
        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        out = self.layers(x)
        return out


# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self,device):

        layers = [4, 100, 200, 200, 100, 1]
        # deep neural networks
        self.dnn = DNN(layers).to(device)
        
        # optimizers: using the same settings        
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(),lr = 1e-3)
        self.scheculer = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_Adam, mode='max', factor=0.5, patience=3)
        self.iter = 0
    
        
    def net_h(self, x,y,z,t): 
        X = torch.cat([x,y,z,t], dim=1) 
        h = self.dnn(X)
        return h 
    
    def net_f(self,x,y,z,t):
        h= self.net_h(x,y,z,t)

        h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h),retain_graph=True,create_graph=True)[0]
        h_y = torch.autograd.grad(h, y, grad_outputs=torch.ones_like(h),retain_graph=True,create_graph=True)[0]
        h_z = torch.autograd.grad(h, z, grad_outputs=torch.ones_like(h),retain_graph=True,create_graph=True)[0]
        h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h),retain_graph=True,create_graph=True)[0]
        
        h_xx = torch.autograd.grad(h_x, x, grad_outputs=torch.ones_like(h_x),retain_graph=True,create_graph=True)[0]
        h_yy = torch.autograd.grad(h_y, y, grad_outputs=torch.ones_like(h_y),retain_graph=True,create_graph=True)[0]
        h_zz = torch.autograd.grad(h_z, z, grad_outputs=torch.ones_like(h_z),retain_graph=True,create_graph=True)[0]       

        f = h_t - 5.56 * 1e-8* h_xx - 5.49 * 1e-8*  h_yy - 14.94 * 1e-8*  h_zz

        return f

    def train_Adam(self,x, y, z, t, h, x2, y2, z2, t2, h2):
        self.dnn.train()
  
        h_pred = self.net_h(x, y, z, t)
        f_pred = self.net_f(x, y, z, t)

        h_base_pred = self.net_h(x2, y2, z2, t2)
        f_base_pred = self.net_f(x2, y2, z2, t2)


        loss = torch.mean((h_pred - h) ** 2) + \
               torch.mean((h_base_pred - h2) ** 2) + \
               torch.mean(f_pred ** 2)       + \
               torch.mean(f_base_pred ** 2)
        
        # Backward and optimize
        self.optimizer_Adam.zero_grad()
        loss.backward()
        self.optimizer_Adam.step()
        self.scheculer.step(loss)

        return loss
        
    
    def predict(self, x,y,z,t):
        self.dnn.eval()
        h = self.net_h(x,y,z,t)
        f = self.net_f(h,x,y,z,t)
        h = h.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return h, f


if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    random_choice(888)
    device = is_cuda()

    grain_train_dataset = "/Volumes/Extreme SSD/train_grain_dataset"
    grain_base_train_dataset = "/Volumes/Extreme SSD/train_base_grain_dataset"
    grain_test_dataset  = "/Volumes/Extreme SSD/test_grain_dataset"
    epochs_num = 100
    
    train_set = Grain_Dataset(grain_train_dataset)
    train_loader = DataLoader(dataset=train_set, batch_size = 1024 ,shuffle=True,pin_memory=False)

    train_base_set = Grain_Dataset(grain_base_train_dataset)
    train_base_loader = DataLoader(dataset=train_base_set, batch_size = 1024 ,shuffle=True,pin_memory=False)

    test_set = Grain_Dataset(grain_test_dataset)
    test_loader = DataLoader(dataset=test_set, batch_size = 1024 ,shuffle=True,pin_memory=False)
    
    model = PhysicsInformedNN(device)

    for m in model.dnn.modules():
        if isinstance(m,torch.nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)
 
    train_base_loader = cycle(train_base_loader)

    #train
    for epoch in range(epochs_num):
        for item,inputs in enumerate(train_loader):
            x,  y,  z,  t , h = inputs
            x = x.to(device); y = y.to(device);z = z.to(device);t = t.to(device);h = h.to(device)
            x2, y2, z2, t2, h2 = next(train_base_loader)
            x2 = x2.to(device); y2 = y2.to(device);z2 = z2.to(device);t2 = t2.to(device);h2 = h2.to(device)
            t0 = time.time()
            loss = model.train_Adam(x,y,z,t,h,x2,y2,z2,t2,h2)
            print('[Epoch %d,It: %d] Loss: %.5f,Time: %.3f'% (epoch+1,item+1,loss.item(),time.time()-t0))