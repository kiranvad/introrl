"""
Deep Q-learning in Pytorch.
Main structure of the code is as follows:
    1. Class is initialized using an environment, model, optimizer and other parameters
    2. Update function involves computing TD loss for Q-learning
    3. Solved for a fixed nummber of frames with in which the learning is assumed to be complete
    4. Model at regular intervals of episodes is saved for comparision.
    
    An example can be seen in the notebooks folder
"""
import math, random
import pdb
import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)
# this throws a warning but it is also allows the code to be less clumsy with requires_grad parameter

print('Using CUDA: {}'.format(USE_CUDA))

from collections import deque
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from IPython.display import clear_output
import matplotlib.pyplot as plt

seed = 123
torch.manual_seed(seed)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
            
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        sample = self.buffer[idx]

        return sample
        
class DQN(object):
    def __init__(self,env,model,optimizer, batch_size = 32, N = 100000,\
                 C=10000, eps_decay = 30000, num_frames = 1500000, gamma=0.99, path=None, verbose='text' ):
        self.env = env
        self.model = model
        self.optimizer = optimizer
        
        self.batch_size = batch_size
        self.N = N
        self.C = C
        self.eps_decay = eps_decay
        self.num_frames = num_frames
        self.gamma = gamma
        self.replay_buffer = ReplayBuffer(self.N)
        if path is None:
            self.path = 'dqnmodel.pth'
        else:
            self.path = path
            
        self.verbose = verbose
            
        if USE_CUDA:
            self.model = self.model.to(device)   
            
    def _get_epsilon(self,frame_idx):
        eps = 0.01 + (1.0 - 0.01) * math.exp(-1. * frame_idx / self.eps_decay)
        return eps
    
    def _plot_trace(self, frame_idx, rewards, losses):
        clear_output(True)
        plt.figure(figsize=(20,5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        plt.show()
        
    def select_action(self,state,epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state   = torch.Tensor(state).unsqueeze(0)
                q = self.model(state.to(device))
                action  = q.max(1)[1].data[0]
        else:
            action = random.randrange(self.env.action_space.n)
        return action
    
    def update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state      = torch.tensor(np.float32(state)).to(device)
        next_state = torch.tensor(np.float32(next_state)).to(device)
        action     = torch.LongTensor(action).to(device)
        reward     = torch.Tensor(reward).to(device)
        done       = torch.Tensor(done).to(device)

        q_values      = self.model(state)
        next_q_values = self.model(next_state)
        
        q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value     = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        target = expected_q_value.data.clone().detach().requires_grad_(False).to(device)

        loss = (q_value.to(device) - target).pow(2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
        
    def solve(self):
        losses = [0]
        all_rewards = []
        episode_reward, episodes = 0,1

        state = self.env.reset()
        for frame_idx in range(1, self.num_frames+1):
            epsilon = self._get_epsilon(frame_idx)
            if frame_idx == self.eps_decay:
                print('Exploration schedule ended...')
            action = self.select_action(state, epsilon)
            if type(action) is torch.Tensor:
                action = action.item()
            next_state, reward, done, _ = self.env.step(action)
            self.replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                state = self.env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                episodes += 1
                
                if episodes%500 ==0:
                    print('Finished {} episodes'.format(episodes))

            if len(self.replay_buffer) > self.C:
                loss = self.update()
                losses.append(loss.item())

            if frame_idx%10000==0:
                print('Frame: {} Reward: {} and loss: {:.2f}'.format(frame_idx,all_rewards[-1],losses[-1]))
                if self.verbose is 'plot':
                    self._plot_trace(frame_idx, all_rewards, losses)
  
            if frame_idx % 1000000 == 0:
                torch.save({
                    'episode': episodes,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, self.path)
        
        output = {'model':self.model, 'trace':[all_rewards,losses]}
        
        return output

        
    
    
    
    
    