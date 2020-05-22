"""
A python file containing various NN models and their util functions for primarilu DQN
"""
import numpy as np
import pdb
from collections import defaultdict
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms

def _getoutkernel(invol,kernel):
    K,F,S,P = kernel
    Din,Win,Hin = invol

    Wout = 1+ ((Win - F + 2*P)/S)
    Hout = 1+ ((Hin - F + 2*P)/S)
    Dout = K
    outvol = [int(Dout),int(Wout),int(Hout)]

    return outvol

def _getoutpool(invol,pool):
    """
    Computes output volume after a maxpool layer without padding and dilation=1
    """
    F,S = pool
    Din,Win,Hin = invol
    Wout = np.floor(1+(Win-(F-1)-1)/S)
    Hout = np.floor(1+(Hin-(F-1)-1)/S)
    Dout = Din

    outpool = [int(Dout),int(Wout),int(Hout)]

    return outpool


class dqndata(Dataset):
    def __init__(self, D):
        self.samples = D

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        return sample
    
# For vector inputs    
class NeuralNet(nn.Module):
    """
    A simple feed forward neural network models 
    
    Inputs
    ------
        env   : OpenAI Gym environment initalized class.
    """
    def __init__(self, env):
        input_size = env.observation_space.shape[0]
        hidden_size = 100
        num_actions = env.action_space.n
        
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 10)
        self.fc3 = nn.Linear(10,num_actions)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    
# For Image inputs
class Convnet(nn.Module):
    """
    A simple convolutional neural network model for Image based modelling.
    Uses a gray scale image of the image observation from OpenAI Gym
    
    Inputs
    ------
        env  : Environment with observation space as a Box.
        num_channels : Number of stacked images as observation
        res : Resolution of each image
    TODOs:
    ------
        1. Probably change the intitialization of models
    """
    def __init__(self, env, num_channels, res):
        super(Convnet, self).__init__()
        num_actions = env.action_space.n
        self.num_channels = num_channels
        outchannels, kernelsize = 3, 8
        self.layer1 = nn.Sequential(nn.Conv2d(self.num_channels,outchannels,kernelsize,stride = 1,
                                              padding = 1),nn.BatchNorm2d(outchannels),nn.MaxPool2d(2,2),nn.ReLU())
        outvol = self._getoutkernel((self.num_channels,res,res),(outchannels,kernelsize,1,1))
        outpool = self._getoutpool(outvol,(2,2))

        outchannels, kernelsize = 3, 8
        self.layer2 = nn.Sequential(nn.Conv2d(outpool[0],outchannels,kernelsize,stride = 1,
                                              padding = 1),nn.BatchNorm2d(outchannels),nn.MaxPool2d(2,2),nn.ReLU())
        outvol = self._getoutkernel(outpool,(outchannels, kernelsize,1,1))
        outpool = self._getoutpool(outvol,(2,2)) 

        outchannels, kernelsize = 3, 8
        self.layer3 = nn.Sequential(nn.Conv2d(outpool[0],outchannels,kernelsize,stride = 1,
                                              padding = 1),nn.BatchNorm2d(outchannels),nn.MaxPool2d(2,2),nn.ReLU())
        outvol = self._getoutkernel(outpool,(outchannels, kernelsize,1,1))
        outpool = self._getoutpool(outvol,(2,2))         
        self.fcin = int(np.prod(outpool))
        self.fc = nn.Linear(self.fcin,num_actions)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.fcin)
        x = self.fc(x)
        return x
    
    def _getoutkernel(self,invol,kernel):
        K,F,S,P = kernel
        Din,Win,Hin = invol
        
        Wout = 1+ ((Win - F + 2*P)/S)
        Hout = 1+ ((Hin - F + 2*P)/S)
        Dout = K
        outvol = [int(Dout),int(Wout),int(Hout)]
        
        return outvol
    
    def _getoutpool(self,invol,pool):
        """
        Computes output volume after a maxpool layer without padding and dilation=1
        """
        F,S = pool
        Din,Win,Hin = invol
        Wout = np.floor(1+(Win-(F-1)-1)/S)
        Hout = np.floor(1+(Hin-(F-1)-1)/S)
        Dout = Din
        
        outpool = [int(Dout),int(Wout),int(Hout)]
        
        return outpool
        
from groupy.gconv.pytorch_gconv import P4MConvP4M,P4ConvZ2,P4ConvP4,P4MConvZ2

class gMaxPool2D(nn.Module):
    def __init__(self,ksize,stride=0,padding=0):
        self.kernel = ksize
        self.stride = stride
        self.padding = padding
        super(gMaxPool2D, self).__init__()
    def forward(self,x):
        xs = x.size()
        x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
        x = F.max_pool2d(input=x, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
        
        return x
                
class Gconvnet(nn.Module):
    """
    Implementation of a Gconvnet using P4m group
    """
    def __init__(self, env, num_channels, res):
        self.num_actions = env.action_space.n
        super(Gconvnet, self).__init__()
        num_actions = env.action_space.n
        self.num_channels = num_channels
        outchannels, kernelsize = 4, 5
        outchannels, kernelsize, stride, padding = 32, 7, 4, 0
        self.layer1 = nn.Sequential(P4MConvZ2(self.num_channels,outchannels,kernelsize,stride = stride,padding = padding),
                                   nn.BatchNorm3d(outchannels),gMaxPool2D(2,2), nn.ReLU())
        outchannels, kernelsize, stride, padding = 64, 5, 2, 0
        self.layer2 = nn.Sequential(P4MConvP4M(32,outchannels,kernelsize,stride = stride,padding = padding),
                                   nn.BatchNorm3d(outchannels),gMaxPool2D(2,2), nn.ReLU())
        
        outshape = [64,8,1,1]
        self.out = np.prod(outshape)
        self.fc = nn.Linear(self.out, self.num_actions)

    def forward(self, x):
        x = self.layer1(x) 
        x = self.layer2(x)
        x = x.view(-1, self.out)
        x = self.fc(x)
        return x
    
class deepmind(nn.Module):
    """
    Convnet architecture used in deepmind paper.
    Model is defined for a 64x64 image only.
    """
    def __init__(self, env, num_channels, res):
        super(deepmind, self).__init__()
        num_actions = env.action_space.n
        self.num_channels = num_channels
        outchannels, kernelsize, stride, padding = 32, 8, 4, 0
        self.layer1 = nn.Sequential(nn.Conv2d(self.num_channels,outchannels,kernelsize,stride = stride,
                                              padding = padding),nn.ReLU())

        outchannels, kernelsize, stride, padding = 64, 4, 2, 0
        self.layer2 = nn.Sequential(nn.Conv2d(32,outchannels,kernelsize,stride = stride,
                                              padding = padding),nn.ReLU())

        outchannels, kernelsize, stride, padding = 64, 3, 1, 0
        self.layer3 = nn.Sequential(nn.Conv2d(64,outchannels,kernelsize,stride = stride,
                                              padding = padding),nn.ReLU())
        self.fcin = 1024
        self.fc1 = nn.Sequential(nn.Linear(self.fcin,512),nn.ReLU())
        self.fc2 = nn.Linear(512,num_actions)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.fcin)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    
""" for Policy Gradient Methods"""    

class pgmodel(nn.Module):
    """
    A simple feed forward neural network models for Policy gradient algorithms 
    
    Inputs
    ------
        env   : OpenAI Gym environment initalized class.
    """
    def __init__(self, env):
        input_size = env.observation_space.shape[0]
        hidden_size = 128
        num_actions = env.action_space.n
        
        super(pgmodel, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size),nn.Dropout(p=0.5),nn.ReLU()) 
        self.fc2 = nn.Linear(hidden_size,num_actions)
        self.softmax = nn.Softmax(dim=1)
            
    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        out = self.softmax(out)
        
        return out
    
class a2cnet(nn.Module):
    """
    A simple feed forward neural network models for A2C Algorithm 
    Works with environments where observation is vector like
    
    Inputs
    ------
        env   : OpenAI Gym environment initalized class.
    """
    def __init__(self, env):
        input_size = env.observation_space.shape[0]
        hidden_size = 128
        num_actions = env.action_space.n
        
        super(a2cnet, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(input_size, hidden_size),nn.Dropout(p=0.5),nn.ReLU()) 
        self.fc2 = nn.Linear(hidden_size,num_actions)
        self.fc3 = nn.Linear(hidden_size,1)
        self.softmax = nn.Softmax(dim=1)
            
    def forward(self, x):
        out = self.fc1(x)
        
        pitheta = self.softmax(self.fc2(out))
        vw = self.fc3(out)
        
        return pitheta, vw
    
    
###########################################################################
# Actor Critic Model tuned for Atari Games
###########################################################################

class a2cCNN(nn.Module):
    """
    Convnet architecture used in deepmind paper.
    Model is defined for a 64x64 image only.
    """
    def __init__(self, env, num_channels, res):
        super(a2cCNN, self).__init__()
        num_actions = env.action_space.n
        self.num_channels = num_channels
        outchannels, kernelsize, stride, padding = 32, 8, 4, 0
        self.layer1 = nn.Sequential(nn.Conv2d(self.num_channels,outchannels,kernelsize,stride = stride,
                                              padding = padding),nn.ReLU())

        outchannels, kernelsize, stride, padding = 64, 4, 2, 0
        self.layer2 = nn.Sequential(nn.Conv2d(32,outchannels,kernelsize,stride = stride,
                                              padding = padding),nn.ReLU())

        outchannels, kernelsize, stride, padding = 64, 3, 1, 0
        self.layer3 = nn.Sequential(nn.Conv2d(64,outchannels,kernelsize,stride = stride,
                                              padding = padding),nn.ReLU())
        self.fcin = 1024
        self.fc1 = nn.Sequential(nn.Linear(self.fcin,512),nn.ReLU())
        
        self.fc2 = nn.Linear(512,num_actions) # action values
        self.fc3 = nn.Linear(512,1) # state values
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(-1, self.fcin)
        x = self.fc1(x)
        
        pitheta = self.softmax(self.fc2(x))
        vw = self.fc3(x)
        
        return pitheta.reshape(-1), vw.reshape(-1)
    
   
    
