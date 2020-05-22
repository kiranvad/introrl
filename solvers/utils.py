"""
various utility functions that may be required across different classes
"""

import numpy as np
from collections import defaultdict
import pdb
np.random.seed()

def eps_greedy(env,qvalues, rate):
    """
    Episilon greedy policy
    Inputs:
    ------
        (uses) qvalues      : qvalues at the state for all possible actions
        (uses) rate : rate of the form : 5*(episode/num_episodes)

    Output:
    ------
        Action : Action to be take at `state` using `Q(s,a)`
    """    
    epsilon = np.exp(-rate)
    coin = np.random.rand()
    
    if coin<epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(qvalues)

    return action 


def greedy_policy(env,qvalues):
    """
    Greedy policy based on qvalues
    Inputs:
    -------
        env  : Environment
        qvalues  :  qvalues at a given state
        
    Outputs:
    --------
        action : Action to take at the state given Q(s,a)
    
    """
    
    action = np.argmax(qvalues)
    
    return action

import torch
from torchvision import transforms
from PIL import Image

def phi_image(T, res = 64):
    grayim = transforms.Compose(
        [transforms.Resize((res,res)),
         transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),
         ])
    X = torch.zeros(len(T[0]),res,res)
    Xn = X
    croparea = (0,20,160,200)
    for frame in range(len(T[0])):
        img = Image.fromarray((T[0][frame] * 255).astype(np.uint8))
        #img = img.crop(croparea)
        X[frame,:,:]  = grayim(img)
        img = Image.fromarray((T[2][frame] * 255).astype(np.uint8))
        #img = img.crop(croparea)
        Xn[frame,:,:]  = grayim(img) 
    
    X.requires_grad = True
    Xn.requires_grad = True
    R = torch.tensor(T[3]).float()
    
    return X, R, Xn  

from IPython import display
import matplotlib.pyplot as plt
import time

import gym

def game_render(ob, ax, show_all_frames=False):
    for i,frame in enumerate(ob):
        frame = frame.transpose()
        ax.imshow(frame)
        ax.set_title('frame%d'%i)
        ax.axis('off')

        if not show_all_frames:
            break
        else:
            time.sleep(0.01)
    
    return ax
        
def play(env,policy = None,render=True,save={'save_fig':True,'path':'./play.png'}):
    ob = env.reset()
    actions_str = env.get_action_meanings()
    score = 0
    step = 1
    output = {'axis_list':  []}
    while True:
        if policy is None:
            action = env.action_space.sample()
        else:
            action = policy(ob)

        ob,reward,done,info = env.step(action) # take a random action
        
        score += reward
        if not render:
            sys.stdout.write("\r" + "Step: "+ str(step)+ "\t Action: "+
                             actions_str[action] + "\t Rewards: " +  str(score) + "\t Lives: " + str(info)
                            )
            sys.stdout.flush()
            time.sleep(0.1)
            if done:
                break
        step+=1
        if render:
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(1,1,1)
            fig.suptitle('Time Step {} Score {} Lives {}'.format(step,score,info['ale.lives']), fontsize=12)
            ax = game_render(ob,ax)
            output['axis_list'].append(ax)
            if done :
                if save['save_fig']:
                    plt.savefig(save['path'],dpi=500,bbox_inches='tight')
                break
            else:
                plt.show()
                display.clear_output(wait=True)
                time.sleep(0.01)
                plt.close()
    print('\n Total reward: {} in {} steps'.format(score, step))
    
    return output


    
    
    
    
    
    
    
    
    
    
    