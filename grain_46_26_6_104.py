import torch
from torch.utils.data import Dataset,DataLoader
import random
from collections import OrderedDict

import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torch.nn as nn
from scipy.interpolate import griddata
#from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
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
        self.grain_dataset = np.load(grain_dataset_add)
        self.total_num = len(self.grain_dataset)
        
    def __getitem__(self, item):
        X = torch.from_numpy(self.grain_dataset[item][0:4], requires_grad = True)
        h = torch.from_numpy(self.grain_dataset[item][4], requires_grad = True)

        return X,h
  
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
    def __init__(self,device,h_value):

        layers = [4, 100, 200, 200, 100, 1]
        # deep neural networks
        self.dnn = DNN(layers).to(device)
        
        # optimizers: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(), 
            lr=1.0, 
            max_iter=100, 
            max_eval=50000, 
            history_size=50,
            tolerance_grad=1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"       # can be "strong_wolfe"
        )
        
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters(),lr = 1e-3)
        self.scheculer = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_Adam, mode='max', factor=0.5, patience=3)
        self.iter = 0
      
        ####################################################
        #######        初始条件及边界条件初值            #######
        ####################################################
        x = np.linspace(0,3,4)[:,None].astype(np.float32) * 6.0 / 3.0
        y = np.linspace(0,9,10)[:,None].astype(np.float32) * 4.6 / 9.0
        z = np.linspace(0,5,6)[:,None].astype(np.float32) * 2.6 / 5.0
        t = np.linspace(0,103,104)[:,None].astype(np.float32)
        
        #使值均匀至-1到1
        self.x0 = 2.0*x/3.0 - 1.0
        self.y0 = 2.0*y/9.0 - 1.0
        self.z0 = 2.0*z/5.0 - 1.0
        self.t0 = 2.0*t/79.0 - 1.0
        h = h_value


        self.x_0yzt, self.y_0yzt, self.z_0yzt, self.t_0yzt = np.meshgrid([-1],self.y0,self.z0,self.t0,indexing='ij')
        self.x_0yzt = torch.tensor(self.x_0yzt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.y_0yzt = torch.tensor(self.y_0yzt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.z_0yzt = torch.tensor(self.z_0yzt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.t_0yzt = torch.tensor(self.t_0yzt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.h_0yzt = torch.tensor(h[[0]][:,y][:,:,z][:,:,:,t].flatten(),requires_grad=True).to(device)

        self.x_50yzt, self.y_50yzt, self.z_50yzt, self.t_50yzt = np.meshgrid([1],self.y0,self.z0,self.t0,indexing='ij')
        self.x_50yzt = torch.tensor(self.x_50yzt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.y_50yzt = torch.tensor(self.y_50yzt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.z_50yzt = torch.tensor(self.z_50yzt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.t_50yzt = torch.tensor(self.t_50yzt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.h_50yzt = torch.tensor(h[[3]][:,y][:,:,z][:,:,:,t].flatten(),requires_grad=True).to(device)

        self.x_x0zt, self.y_x0zt, self.z_x0zt, self.t_x0zt = np.meshgrid(self.x0,[-1],self.z0,self.t0,indexing='ij')
        self.x_x0zt = torch.tensor(self.x_x0zt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.y_x0zt = torch.tensor(self.y_x0zt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.z_x0zt = torch.tensor(self.z_x0zt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.t_x0zt = torch.tensor(self.t_x0zt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.h_x0zt = torch.tensor(h[x][:,[0]][:,:,z][:,:,:,t].flatten(),requires_grad=True).to(device)

        self.x_x90zt, self.y_x90zt, self.z_x90zt, self.t_x90zt = np.meshgrid(self.x0,[1],self.z0,self.t0,indexing='ij')
        self.x_x90zt = torch.tensor(self.x_x90zt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.y_x90zt = torch.tensor(self.y_x90zt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.z_x90zt = torch.tensor(self.z_x90zt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.t_x90zt = torch.tensor(self.t_x90zt.flatten().astype(np.float32), requires_grad=True).to(device)
        self.h_x90zt = torch.tensor(h[x][:,[9]][:,:,z][:,:,:,t].flatten(),requires_grad=True).to(device)

        self.x_xy0t, self.y_xy0t, self.z_xy0t, self.t_xy0t = np.meshgrid(self.x0,self.y0,[-1],self.t0,indexing='ij')
        self.x_xy0t = torch.tensor(self.x_xy0t.flatten().astype(np.float32), requires_grad=True).to(device)
        self.y_xy0t = torch.tensor(self.y_xy0t.flatten().astype(np.float32), requires_grad=True).to(device)
        self.z_xy0t = torch.tensor(self.z_xy0t.flatten().astype(np.float32), requires_grad=True).to(device)
        self.t_xy0t = torch.tensor(self.t_xy0t.flatten().astype(np.float32), requires_grad=True).to(device)
        self.h_xy0t = torch.tensor(h[x][:,y][:,:,[0]][:,:,:,t].flatten(),requires_grad=True).to(device)

        self.x_xy18t, self.y_xy18t, self.z_xy18t, self.t_xy18t = np.meshgrid(self.x0,self.y0,[1],self.t0,indexing='ij')
        self.x_xy18t = torch.tensor(self.x_xy18t.flatten().astype(np.float32), requires_grad=True).to(device)
        self.y_xy18t = torch.tensor(self.y_xy18t.flatten().astype(np.float32), requires_grad=True).to(device)
        self.z_xy18t = torch.tensor(self.z_xy18t.flatten().astype(np.float32), requires_grad=True).to(device)
        self.t_xy18t = torch.tensor(self.t_xy18t.flatten().astype(np.float32), requires_grad=True).to(device)
        self.h_xy18t = torch.tensor(h[x][:,y][:,:,[5]][:,:,:,t].flatten(),requires_grad=True).to(device)

        self.x_xyz0, self.y_xyz0, self.z_xyz0, self.t_xyz0 = np.meshgrid(self.x0,self.y0,self.z0,[-1],indexing='ij')
        self.x_xyz0 = torch.tensor(self.x_xyz0.flatten().astype(np.float32), requires_grad=True).to(device)
        self.y_xyz0 = torch.tensor(self.y_xyz0.flatten().astype(np.float32), requires_grad=True).to(device)
        self.z_xyz0 = torch.tensor(self.z_xyz0.flatten().astype(np.float32), requires_grad=True).to(device)
        self.t_xyz0 = torch.tensor(self.t_xyz0.flatten().astype(np.float32), requires_grad=True).to(device)
        self.h_xyz0 = torch.tensor(h[x][:,y][:,:,z][:,:,:,[0]].flatten(),requires_grad=True).to(device)
        
    def net_h(self, x,y,z,t):  
        X = torch.cat([x[:,None],y[:,None],z[:,None],t[:,None]], dim=1)
        h = self.dnn(X)
        return h 
    
    def net_f(self, h,x,y,z,t):
        """ The pytorch autograd version of calculating residual """
        h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h),retain_graph=True,create_graph=True)[0]
        h_y = torch.autograd.grad(h, y, grad_outputs=torch.ones_like(h),retain_graph=True,create_graph=True)[0]
        h_z = torch.autograd.grad(h, z, grad_outputs=torch.ones_like(h),retain_graph=True,create_graph=True)[0]
        h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h),retain_graph=True,create_graph=True)[0]
        
        h_xx = torch.autograd.grad(h_x, x, grad_outputs=torch.ones_like(h_x),retain_graph=True,create_graph=True)[0]
        h_yy = torch.autograd.grad(h_y, y, grad_outputs=torch.ones_like(h_y),retain_graph=True,create_graph=True)[0]
        h_zz = torch.autograd.grad(h_z, z, grad_outputs=torch.ones_like(h_z),retain_graph=True,create_graph=True)[0]       

        f = h_t - 5.56 * 1e-8* h_xx - 5.49 * 1e-8*  h_yy - 14.94 * 1e-8*  h_zz

        return f

    def train_Adam(self,x,y,z,t,h):
        self.dnn.train()
        # #使值均匀至-1到1
        # x = 2.0*x/3.0 - 1.0
        # y = 2.0*y/9.0 - 1.0
        # z = 2.0*z/5.0 - 1.0
        # t = 2.0*t/79.0 - 1.0
  
        h_pred = self.net_h(x, y, z, t)
        f_pred = self.net_f(h_pred, x, y, z, t)

        h_0yzt_pred = self.net_h(self.x_0yzt,self.y_0yzt,self.z_0yzt,self.t_0yzt)
        h_50yzt_pred = self.net_h(self.x_50yzt,self.y_50yzt,self.z_50yzt,self.t_50yzt)
        h_x0zt_pred = self.net_h(self.x_x0zt,self.y_x0zt,self.z_x0zt,self.t_x0zt)
        h_x90zt_pred = self.net_h(self.x_x90zt,self.y_x90zt,self.z_x90zt,self.t_x90zt)
        h_xy0t_pred = self.net_h(self.x_xy0t,self.y_xy0t,self.z_xy0t,self.t_xy0t)
        h_xy18t_pred = self.net_h(self.x_xy18t,self.y_xy18t,self.z_xy18t,self.t_xy18t)
        h_xyz0_pred = self.net_h(self.x_xyz0,self.y_xyz0,self.z_xyz0,self.t_xyz0)

        loss = torch.mean((h_0yzt_pred - self.h_0yzt) ** 2) + \
               torch.mean((h_50yzt_pred - self.h_50yzt) ** 2) + \
               torch.mean((h_x0zt_pred - self.h_x0zt) ** 2) + \
               torch.mean((h_x90zt_pred - self.h_x90zt) ** 2) + \
               torch.mean((h_xy0t_pred - self.h_xy0t) ** 2) + \
               torch.mean((h_xy18t_pred - self.h_xy18t) ** 2) + \
               torch.mean((h_xyz0_pred - self.h_xyz0) ** 2) + \
               torch.mean(f_pred ** 2)
        
        # Backward and optimize
        self.optimizer_Adam.zero_grad()
        loss.backward()
        self.optimizer_Adam.step()
        #self.scheculer.step(loss)

        return loss
        
    def loss_fun(self):
        h_pred = self.net_h(self.L_x, self.L_y, self.L_z, self.L_t)
        f_pred = self.net_f(h_pred,self.L_x, self.L_y, self.L_z, self.L_t)

        h_0yzt_pred = self.net_h(self.x_0yzt,self.y_0yzt,self.z_0yzt,self.t_0yzt)
        h_50yzt_pred = self.net_h(self.x_50yzt,self.y_50yzt,self.z_50yzt,self.t_50yzt)
        h_x0zt_pred = self.net_h(self.x_x0zt,self.y_x0zt,self.z_x0zt,self.t_x0zt)
        h_x90zt_pred = self.net_h(self.x_x90zt,self.y_x90zt,self.z_x90zt,self.t_x90zt)
        h_xy0t_pred = self.net_h(self.x_xy0t,self.y_xy0t,self.z_xy0t,self.t_xy0t)
        h_xy18t_pred = self.net_h(self.x_xy18t,self.y_xy18t,self.z_xy18t,self.t_xy18t)
        h_xyz0_pred = self.net_h(self.x_xyz0,self.y_xyz0,self.z_xyz0,self.t_xyz0)

        loss = torch.mean((h_0yzt_pred - self.h_0yzt) ** 2) + \
               torch.mean((h_50yzt_pred - self.h_50yzt) ** 2) + \
               torch.mean((h_x0zt_pred - self.h_x0zt) ** 2) + \
               torch.mean((h_x90zt_pred - self.h_x90zt) ** 2) + \
               torch.mean((h_xy0t_pred - self.h_xy0t) ** 2) + \
               torch.mean((h_xy18t_pred - self.h_xy18t) ** 2) + \
               torch.mean((h_xyz0_pred - self.h_xyz0) ** 2) + \
               torch.mean(f_pred ** 2)

        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        return loss

    def train_LBFGS(self,x,y,z,t,h):
        # Backward and optimize

        #使值均匀至-1到1
        self.dnn.train()
        self.L_x = 2.0*x/3.0 - 1.0
        self.L_y = 2.0*y/9.0 - 1.0
        self.L_z = 2.0*z/5.0 - 1.0
        self.L_t = 2.0*t/79.0 - 1.0
        self.L_h = h
                
        loss = self.optimizer.step(self.loss_fun)

        return loss
    
    def predict(self, x,y,z,t):
        self.dnn.eval()
        h = self.net_h(x,y,z,t)
        f = self.net_f(h,x,y,z,t)
        h = h.detach().cpu().numpy()
        f = f.detach().cpu().numpy()
        return h, f

# train
def train(x,y,z,t,h):
    pass


# evaluations
def evaluations(x,y,z,t,h_value,best_path=None):
    model = PhysicsInformedNN(device,h_value)
    best_para = best_path
    if best_para:
        model.dnn.load_state_dict(torch.load(best_para))

    h_pred, f_pred = model.predict(x,y,z,t)

    return h_pred,f_pred

if __name__ == '__main__':

    warnings.filterwarnings('ignore')
    random_choice(1234)
    device = is_cuda()

    grain_dataset_add = "./data/grain_dataset_46_26_6_104.npy"
    
    train_set = Grain_Dataset(grain_dataset_add)
    train_loader = DataLoader(dataset=train_set, batch_size = 4096 ,shuffle=True,pin_memory=False)

    model = PhysicsInformedNN(device,h_value)

    for m in model.dnn.modules():
        if isinstance(m,torch.nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0.0)

    epochs_Adam = 10
    
    best_loss_Adam = 99999
    

    for epoch in range(epochs_Adam):
        for item,inputs in enumerate(train_loader):
            X, h = inputs
            t0 = time.time()
            loss = model.train_Adam(X,h)
            print('[Epoch %d,It: %d] Loss: %.5f,Time: %.3f'% (epoch+1,item+1,loss.item(),time.time()-t0))
        if(loss < best_loss_Adam):
            best_loss_Adam = loss
            torch.save(model.dnn.state_dict(),'./stage1_epoch%d.pth'%(epoch))




    # epochs_LBFGS = 1    
    # best_loss_LBFGS = 99999  
    # best_para = '/data/xlj/git_repo/PINNs_pytorch/stage1_epoch8.pth'
    # for epoch in range(epochs_LBFGS):
    #     for item,inputs in enumerate(train_loader):
    #         t0 = time.time()
    #         x,y,z,t,h = inputs
    #         if best_para:
    #             model.dnn.load_state_dict(torch.load(best_para))
    #         loss = model.train_LBFGS(x,y,z,y,h)
    #         print('[Epoch %d,It: %d] Loss: %.5f,Time: %.3f'% (epoch+1,item+1,loss.item(),time.time()-t0))
    #         if(loss < best_loss_LBFGS):
    #             best_loss_LBFGS = loss
    #             torch.save(model.dnn.state_dict(),'./stage2_epoch%ditem%d.pth'%(epoch,item))

    # x = torch.tensor(np.array([0]*73).astype(np.float32),requires_grad=True).to(device)
    # y = torch.tensor(np.array([0]*73).astype(np.float32),requires_grad=True).to(device)
    # z = torch.tensor(np.array([0]*73).astype(np.float32),requires_grad=True).to(device)
    # t = torch.tensor(np.linspace(0,72,73).astype(np.float32),requires_grad=True).to(device)

    # print(h_value[0,0,0,:].mean())
    #h_pred,f_pred = evaluations(x,y,z,t,h_value,best_path='/data/xlj/git_repo/PINNs_pytorch/stage2_epoch0item81.pth')
    #print(h_pred,'\n',f_pred)
    # error_h = np.linalg.norm(h_value-h_pred,2)
    # error_h_percentage = np.linalg.norm(h_value-h_pred,2)/np.linalg.norm(h_value,2)
    # error_f = np.linalg.norm(f_pred,2)
    # print('Error h_percentage: %f,Error h: %f,Error f: %f' % (error_h_percentage,error_h,error_f)) 
    


   