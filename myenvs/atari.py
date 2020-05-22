import gym
from gym.envs.atari.atari_env import AtariEnv
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import pdb
import time
import sys

import torch
from torchvision import transforms
from PIL import Image
from IPython import display

GAME = ['space_invaders','breakout']

import sys
if '../' not in sys.path:
    sys.path.insert(0,'../')

from myenvs.helpers import make_atari, wrap_deepmind, wrap_pytorch

def get_deepmind_atari(env, mode='train'):
    env = make_atari(env)
    env = wrap_pytorch(wrap_deepmind(env),mode=mode)   
    return env

class ATARI(AtariEnv):
    def __init__(self,game,frameskip=4):
        super(ATARI,self).__init__(game=game,obs_type='image',frameskip=frameskip)
        
        if isinstance(self.frameskip, int):
            self.num_steps = self.frameskip*4
        else:
            raise NotImplementedError()
            
    def step(self, a,**kwargs):
        if not len(kwargs)==0:
            max_steps = kwargs['max_steps']
            steps = kwargs['steps']
            stepreward = kwargs['stepreward']
        else:
            steps = 0
            stepreward = 1
            max_steps = 10000
            
        ob = []
        action = self._action_set[a]
        reward = self.ale.act(action)

        for s in range(self.num_steps):
            reward += self.ale.act(0)
            if s%self.frameskip==0:
                ob.append(self._get_obs())
        done = self.ale.game_over()
        if not done:
            reward += stepreward
        
        done = bool(done or steps>max_steps)    
        
        return ob, reward, done, {"ale.lives": self.ale.lives()}
    
    def reset(self):
        ob = []
        self.ale.reset_game()
        for s in range(self.num_steps):
            if s%self.frameskip==0:
                ob.append(self._get_obs())      
        return ob
    
    def render(self, ob, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1,1,1)
        for i,frame in enumerate(ob):
            ax.imshow(frame)
            ax.set_title('frame%d'%i)
            ax.axis('off')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.1)
              
class atariwrap(gym.Wrapper):
    def __init__(self,env,num_channels=4):
        super().__init__(env)
        self.num_repeats = num_channels
        
    def step(self, a,**kwargs):
        if not len(kwargs)==0:
            max_steps = kwargs['max_steps']
            steps = kwargs['steps']
            stepreward = kwargs['stepreward']
        else:
            steps = 0
            stepreward = 0
            max_steps = 10000
    
        ob = []
        st,reward,done,info = self.env.step(a)
        ob.append(st)
        for s in range(self.num_repeats-1):
            st,rt,done,info = self.env.step(1)
            ob.append(st)
            reward += rt
        if not done:
            reward += stepreward
        
        done = bool(done or steps>max_steps)
        ob = self.phi(ob)
        return ob, reward, done, info
    
    def reset(self):
        ob = []
        frame = self.env.reset()
        ob.append(frame)   
        for s in range(self.num_repeats-1):
            st,rt,done,info = self.env.step(0)
            ob.append(st)  
        ob = self.phi(ob)
        
        return ob
    
    def phi(self,ob):
        res = 64
        grayim = transforms.Compose([transforms.Resize((res,res)),
                                     transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
        X = torch.zeros(len(ob),res,res)
        croparea = (0,20,160,200)
        for frame in range(len(ob)):
            img = Image.fromarray((ob[frame] * 255).astype(np.uint8))
            img = img.crop(croparea)
            X[frame,:,:]  = grayim(img)
            
        return X
    
    def render(self, ob, ax=None):
        if ax is None:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1,1,1)
        for i,frame in enumerate(ob):
            ax.imshow(frame)
            ax.set_title('frame%d'%i)
            ax.axis('off')
            display.display(plt.gcf())
            display.clear_output(wait=True)
            time.sleep(0.1)

        
            
        
    
    
    
        
        
        
    