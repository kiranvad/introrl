"""
Atari DQN on any selected game using a CNN model.
"""
import math, random
import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')

import pdb
import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')
    
from solvers.deepqlearn import DQN
from solvers.models import deepmind
import gym
import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

from solvers.wrappers import make_atari, wrap_deepmind, wrap_pytorch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env_id = "PongNoFrameskip-v4"
print(env_id)
env    = make_atari(env_id)
env    = wrap_deepmind(env,episode_life=True,frame_stack=True)
env    = wrap_pytorch(env)

model = deepmind(env.observation_space.shape[0], env.action_space.n)              
optimizer = optim.Adam(model.parameters(), lr=0.00001)
                
num_frames = 1500000
N = 100000
batch_size = 32
C = 10000
eps_decay = 30000
path = env_id+ '_deepmind.pth'

dqn = DQN(env,model,optimizer,N=N, C = C, \
          num_frames = num_frames, batch_size = batch_size, eps_decay = eps_decay, path=path, verbose='text')
output = dqn.solve()

with open(env_id +'_deepmind.pkl', 'wb') as f: 
    pickle.dump(output['trace'], f)