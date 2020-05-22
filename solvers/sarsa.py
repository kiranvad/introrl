"""
Implementation of Temporal Difference Learning: On-policy Learning: SARSA algorithm
This was purely implemented based on Prof.David Silver course slides which provide 
exact algorithm for the two methods:
    1. SARSA(0) -- Forward View SARSA algorithm  (Lecture 5 - Slide 22)
    2. SARSA(\lambda) -- Backward view SARSA(lambda) algorithm (Lecture5- Slide 29)
"""
import sys
sys.path.insert(0,'../')
import numpy as np
from collections import defaultdict, namedtuple
import pdb
np.random.seed()

from solvers.utils import eps_greedy

class SARSA():
    def __init__(self,env,num_episodes=100,alpha=1.0, gamma = 1.0, verbose = False, eps_decay = None):
        """
        Initiate SARSA class with an environment
        """
        self.env = env
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose
        self.trace = namedtuple("trace",["lengths", "rewards","epsilon"])
        self.trace = self.trace(lengths=[],rewards=[],epsilon = [])
        
        if eps_decay is None:
            self.eps_decay = 10/self.num_episodes
        else:
            self.eps_decay = eps_decay
    
    
    def Zero(self):
        """
        SARSA Algorithm as presented in Prof.David Silver course: Lecture5-Slide 22
        """
        self.Qsa = defaultdict(float)
        
        for e in range(self.num_episodes):
            self.epsilon = np.exp(-self.eps_decay*e)
            self.env.reset()
            is_done = False
            steps = 0
            rewards = 0
            while not is_done:
                state = self.env.s
                # select an action using eps-greedy policy
                action = eps_greedy(self.env,self.Qsa,self.epsilon, state)
                sp,r,is_done = self.env.step(action)
                ap = eps_greedy(self.env,self.Qsa,self.epsilon, sp)
                
                self.Qsa[state,action] += self.alpha*(r + self.gamma*self.Qsa[sp,ap] -self.Qsa[state,action])
                steps += 1
                rewards += r
                
            self.trace.rewards.append(rewards)
            self.trace.lengths.append(steps)
            self.trace.epsilon.append(self.epsilon)
            
            if self.verbose:
                perc_episodes = 100*(e/self.num_episodes)
                if perc_episodes%10==0:
                    print('{}/{} episodes finished'.format(e,self.num_episodes))
        
        return self.Qsa
            
            
    def Lambda(self, Lambda = 1.0):
        """
        SARSA(lambda) as presented in Prof.David Silver course: Lecture5-Slide 29
        """
        self.Qsa = defaultdict(float)
        self.Lambda = Lambda
        
        nS = self.env.observation_space.n
        nA = self.env.action_space.n
        
        for e in range(self.num_episodes):
            self.E = defaultdict(float)
            self.epsilon = np.exp(-self.eps_decay*e)
            self.env.reset()
            is_done = False
            
            steps = 0
            rewards = 0
            while not is_done:
                state = self.env.s
                action = eps_greedy(self.env,self.Qsa,self.epsilon, state)
                sp,r,is_done = self.env.step(action)
                ap = eps_greedy(self.env,self.Qsa,self.epsilon, sp)
                
                delta = r + self.gamma*self.Qsa[sp,ap] - self.Qsa[state,action]
                self.E[state,action] += 1
                #pdb.set_trace()
                for s in range(nS):
                    for a in range(nA):
                        self.E[s,a] = self.E[s,a]*self.gamma*self.Lambda
                        self.Qsa[s,a] += self.alpha*delta*self.E[s,a]
                steps += 1
                rewards += r
                
            self.trace.rewards.append(rewards)
            self.trace.lengths.append(steps)
            self.trace.epsilon.append(self.epsilon)
            
            if self.verbose:
                perc_episodes = 100*(e/self.num_episodes)
                if perc_episodes%10==0:
                    print('{}/{} episodes finished'.format(e,self.num_episodes))
            
        return self.Qsa
        
        
        
        
        
        
        
        
        
        
        