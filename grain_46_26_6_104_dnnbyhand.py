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

# the deep neural network
class Grain_Dataset(Dataset):
    def __init__(self,grain_dataset_add):
        super().__init__()
        self.x_raw = np.linspace(0,3,4)[:,None].astype(np.float32) * 6.0 / 3.0
        self.y_raw = np.linspace(0,9,10)[:,None].astype(np.float32) * 4.6 / 9.0
        self.z_raw = np.linspace(0,5,6)[:,None].astype(np.float32) * 2.6 / 5.0
        self.t_raw = np.linspace(0,103,104)[:,None].astype(np.float32)
        self.x, self.y, self.z, self.t = np.meshgrid(self.x_raw,self.y_raw,self.z_raw,self.t_raw,indexing='ij')
        self.x = self.x.flatten(); self.y = self.y.flatten(); self.z = self.z.flatten(); self.t = self.t.flatten()
        self.h = np.load(grain_dataset_add).astype(np.float32).flatten()
        self.total_num = len(self.h)
        
    def __getitem__(self, item):

        x = torch.tensor(self.x[item:item + 1],requires_grad = True)
        y = torch.tensor(self.y[item:item + 1],requires_grad = True)
        z = torch.tensor(self.z[item:item + 1],requires_grad = True)
        t = torch.tensor(self.t[item:item + 1], requires_grad = True)
        h = torch.tensor(self.h[item:item + 1], requires_grad = True)

        return x,y,z,t,h
  
    def __len__(self):
        return self.total_num

# the physics-guided neural network
class PhysicsInformedNN():
    def __init__(self,device):        
        self.iter = 0

        self.layers = [4, 100, 200, 200, 100, 1]
        self.weights, self.biases = self.initialize_NN(self.layers, device)
        self.parameter_dict  = [{"params":self.weights}, {"params":self.biases}]

        # optimizers: using the same settings
        #self.optimizer_SGD = torch.optim.SGD(self.dnn.parameters(),lr = 1e-3)        
        self.optimizer_Adam = torch.optim.Adam(self.parameter_dict,lr = 1e-3, betas = (0.9,0.999),eps = 1e-8)
        #self.scheculer = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_Adam, mode='min', factor=0.5, patience=5)

    def initialize_NN(self, layers, device):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = torch.zeros([layers[l], layers[l+1]]).to(device).requires_grad_(True)
            nn.init.xavier_normal_(W)
            b = torch.zeros([1,layers[l+1]]).to(device).requires_grad_(True)
            weights.append(W)
            biases.append(b)   

        return weights, biases

    def dnn(self, X):
        num_layers = len(self.weights) + 1

        for l in range(0,num_layers-2):
            W = self.weights[l]
            b = self.biases[l]
            X = torch.tanh(torch.add(torch.matmul(X, W), b))
        W = self.weights[-1]
        b = self.biases[-1]
        Y = torch.add(torch.matmul(X, W), b)

        return Y
        
    def net_h(self, x, y, z, t): 
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

    def train_Adam(self,x, y, z, t, h):

        h_pred = self.net_h(x, y, z, t)
        f_pred = self.net_f(x, y, z, t)

        self.iter += 1
        if(self.iter % 20 == 1):
            h_print = torch.cat([h_pred,h], dim = 1)
            print(h_print)

        # h_base_pred = self.net_h(x2, y2, z2, t2)
        # f_base_pred = self.net_f(x2, y2, z2, t2)


        loss = torch.mean((h_pred - h) ** 2) + \
               torch.mean(f_pred ** 2)       
            #    torch.mean((h_base_pred - h2) ** 2) + \
            #    torch.mean(f_base_pred ** 2)
        
        # Backward and optimize
        self.optimizer_Adam.zero_grad()
        loss.backward(retain_graph=True)
        self.optimizer_Adam.step()
        #self.scheculer.step(loss)

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

    grain_dataset_add = "/data/xlj/git_repo/PINNs_raw/main_dir/Data/np_4_10_6_104.npy"

    epochs_num = 2000
    
    train_set = Grain_Dataset(grain_dataset_add)
    train_loader = DataLoader(dataset=train_set, batch_size = 16384 ,shuffle=True,pin_memory=False)
    
    model = PhysicsInformedNN(device)

    #train
    for epoch in range(epochs_num):
        for item,inputs in enumerate(train_loader):
            x,  y,  z,  t , h = inputs
            x = x.to(device); y = y.to(device);z = z.to(device);t = t.to(device);h = h.to(device)
            t0 = time.time()
            loss = model.train_Adam(x,y,z,t,h)
            print('[Epoch %d,It: %d] Loss: %.5f,Time: %.3f'% (epoch+1,item+1,loss.item(),time.time()-t0))
