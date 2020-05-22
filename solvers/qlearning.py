"""
Q-learning algorithm for off-policy control.
The aim is to learn a policy by observing a behaviour policy.
This algorithm is completely based on Prof.David Silver lecture slides : Lecture 5 - Slide 38
"""
import numpy as np
from collections import defaultdict, namedtuple
from solvers.utils import eps_greedy
import sys
sys.path.insert(0,'../')
import pdb
np.random.seed()

class Qlearning():
    """
    Main class for Qlearning algorithm. Initiate the class with an environment
    """
    def __init__(self,env,num_episodes = 100,alpha=1.0, gamma = 1.0, verbose = False,\
                 eps_decay = None):
        self.env = env
        self.num_episodes = num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose
        self.trace = namedtuple("trace",["lengths", "rewards","epsilon"])
        # based on https://github.com/dennybritz/reinforcement-learning/
        # blob/master/TD/SARSA%20Solution.ipynb
        self.trace = self.trace(lengths=[],rewards=[],epsilon = [])
        
    def tabular(self):

        self.Qsa = defaultdict(float)
        
        nA = self.env.action_space.n
        
        for e in range(self.num_episodes):
            self.env.reset()
            is_done = False
            steps = 0
            rewards = 0
            while not is_done:
                state = self.env.s
                
                nA = self.env.action_space.n
                qvalues = np.zeros(nA)
                for a in range(nA):
                    qvalues[a] = Qsa[state,a]
                    
                action = eps_greedy(self.env,qvalues,e*(5/self.num_episodes), state)
                sp,r,is_done = self.env.step(action)
                qvalues = np.zeros(nA)
                for a in range(nA):
                    qvalues[a] = self.Qsa[sp,a]
                self.Qsa[state,action] += self.alpha*(r + self.gamma*np.max(qvalues) - self.Qsa[state,action])
                
                rewards += r
                steps += 1
            self.trace.rewards.append(rewards)
            self.trace.lengths.append(steps)
            self.trace.epsilon.append(self.epsilon)    
            if self.verbose:
                perc_episodes = 100*(e/self.num_episodes)
                if perc_episodes%10==0:
                    print('{}/{} episodes finished'.format(e,self.num_episodes))     
        return self.Qsa
        
    