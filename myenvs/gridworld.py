# -*- coding: utf-8 -*-
"""
A python class that defines a basic gridworld environment
Written to be compatible with OpenAI Gym environments

Example usage:

gw = gridworld.GridWorld()
gw.reset()
gw.ax.scatter(gw.flatgrid[gw.initialstate][0],gw.flatgrid[gw.initialstate][1],s=100)
actions = ['right', 'left', 'up', 'down']
for _ in range(10):
    act = np.random.randint(4, size=1)[0]
    gw.step(act) # take a random action
    print(actions[act],gw.oldstate,gw.newstate)
    gw.render()
print('Total Reward: {}'.format(gw.cumlreward))
plt.show()

Created on Fri Jan 31 15:24:57 2020

@author: kiranvad
"""
import numpy as np
import pdb
import matplotlib.pyplot as plt
from itertools import product
np.random.seed()

from gym import spaces

class GridWorld:
    """
    A Simple grid world environment with a determinstic transition probability
    """
    def __init__(self,size=[6,6]):
        """
        Initiate your grid world with the following:
            size : Size of the square grid world
            actions : Different actions that you can take
            rewards : Various rewards possible in the environment
        """
        self.nA = 4
        self.nS  = np.prod(size)
        self.size = size
        # Just to be in sync with GYM environments
        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)
        self.actions = ['R', 'L', 'U', 'D']
                
        self._grid = list(product(np.arange(self.size[0]), np.arange(self.size[0])))
        self._terminal_states = [self._grid[0],self._grid[-1]]
        self.Tstates = [0,self.observation_space.n-1]
        self._det_next_state = [[1,0],[-1,0],[0,1],[0,-1]] # in the order (right, left, up, down)

    # Define reset for environment
    def reset(self, init_state = None):
        # initiate it to a random seed
        if init_state is None:
            self.oldstate = np.random.choice(np.arange(1,self.nS-1), 1)[0]
        else:
            self.oldstate = init_state
        self.initialstate = self.oldstate
        self.s = self.oldstate

    # Define how to take a step
    def step(self, action):
        state = self.s
        prob = np.zeros(self.action_space.n)
        current_trans_probs = self.P[self.s][action]
        for action,tp in enumerate(current_trans_probs):
            prob[action] = tp[0][0]

        coin = np.random.choice([0,1,2,3], 1, p=prob)[0]
        tpms = self.P[self.s][action][coin][0]
        self.newstate = tpms[1]
        self.recentaction = coin
        self.cumlreward += tpms[2]
        self.s = tpms[1]
        return tpms[1], tpms[2], tpms[3]

    # Define a render to plot grid world
    def render(self):
        '''
        This function does the following:
            0. Use the environment plot axis to add arrow of trjectories
            1. Shows the current state and cumulative reward on top of it
        '''
        diff = tuple(i-j for i,j in zip(self.flatgrid[self.newstate],self.flatgrid[self.oldstate]))
        self.ax.arrow(self.flatgrid[self.oldstate][0], self.flatgrid[self.oldstate][1], diff[0],diff[1],\
                      head_width=0.2, head_length=0.2, fc='lightblue', ec='black')
        self.oldstate = self.newstate

    def _plotenv(self, showgridids=False):

        #fig = plt.figure()
        ax = plt.gca()

        for i in range(self.size[0] + 1):
            ax.plot(np.arange(self.size[0] + 1) - 0.5, np.ones(self.size[0] + 1) * i - 0.5, color='k')

        for i in range(self.size[1] + 1):
            ax.plot(np.ones(self.size[1] + 1) * i - 0.5, np.arange(self.size[1] + 1) - 0.5, color='k')
        for tstate in self.Tstates:
            ax.scatter(self.flatgrid[tstate][0],self.flatgrid[tstate][1],marker="8",c='red',s=250)
        if showgridids:
            for ind,state in enumerate(self.flatgrid):
                ax.text(state[0],state[1],ind)

        ax.set_aspect('equal')

        ax.grid(False)
        self.ax = ax
        plt.axis('off')

        return ax
    
    def _get_stepin_state(self,state,index,action):
        """
        A function that can be used to get possible state, given a state and an action.
        Inputs:
        ------
            state :  A state tupple
            index : state index
            action : action index
        Outputs:
        -------
            next_state : next state as a tuple, useful for grid plotting
            next_state_index :  next state as an index
        """
        if state in self._terminal_states:
            next_state = state
            next_state_index = index
            done = True
        else:
            done = False
            stepinto = np.asarray(state) + self._det_next_state[action]
            stepinto_index = np.where((np.asarray(self._grid) == stepinto).all(axis=1))
            if stepinto_index[0].size ==0:
                next_state = state
                next_state_index = index
            else:
                next_state = tuple(stepinto)
                next_state_index = stepinto_index[0][0]
        
        return next_state, next_state_index
    

class DetermGridWorld(GridWorld):
    """
    A simple deterministic gridworld class.
    Sets the probability of each action to one.
    """
    def __init__(self,size=[6,6],plotgrid=True):
        super(DetermGridWorld, self).__init__(size=size)
             
        self._get_grid() 
        if plotgrid:
            self.ax = self._plotenv()

    def _get_grid(self):
        tpm = {} # transition probability matrix with (probab, next_state, reward)

        for index, state in enumerate(self._grid): # get a state
            tpm[index] = {a: [] for a in range(self.action_space.n)}
            for action in range(self.action_space.n): # take an action                  
                probs = np.zeros(self.action_space.n)
                probs[action] = 1.0
                for ap in range(self.action_space.n):
                    next_state, next_state_index = self._get_stepin_state(state,index,ap)
                    # define a reward for the state and action pair
                    if next_state is self._terminal_states[0]:
                        current_reward = 5
                        done = True
                    elif next_state is self._terminal_states[1]:
                        current_reward = 7
                        done = True
                    else:
                        current_reward = -1
                        done = False
                    temp_tpm = [(probs[ap],next_state_index,current_reward, done)]

                    tpm[index][action].append(temp_tpm)

        self.P = tpm
        self.flatgrid = self._grid
        self.newstate = []
        self.cumlreward = 0


class StochasticGridWorld(GridWorld):
    """
    Stochastic grid world built based on the base class.

    """
    def __init__(self,size=[6,6],plotgrid=True):
        super(StochasticGridWorld, self).__init__(size=size)
             
        self._get_grid() 
        if plotgrid:
            self.ax = self._plotenv()
        
    def _get_grid(self):
        tpm = {} # transition probability matrix with (probab, next_state, reward)

        for index, state in enumerate(self._grid): # get a state
            tpm[index] = {a: [] for a in range(self.action_space.n)}
            for action in range(self.action_space.n): # take an action
                for ap in range(self.action_space.n):
                    if action == ap:
                        prob = 0.5
                    else:
                        prob = 0.5/3
                    next_state, next_state_index = self._get_stepin_state(state,index,ap)
                    # define a reward for the state and action pair
                    if next_state is self._terminal_states[0]:
                        current_reward = 5
                        done = True
                    elif next_state is self._terminal_states[1]:
                        current_reward = 7
                        done = True
                    else:
                        current_reward = -1
                        done = False
                    temp_tpm = [(prob,next_state_index,current_reward, done)]

                    tpm[index][action].append(temp_tpm)

        self.P = tpm
        self.flatgrid = self._grid
        self.newstate = []
        self.cumlreward = 0

