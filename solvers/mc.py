"""
Monte-Carlo Methods for prediction and control in reinforcement learning.
Following techniques are implemented:
    1. first-visit monte carlo policy evaluation
    2. every-visit monte carlo policy evaluation
    3. incremental monte carlo
"""
import numpy as np
import pdb
import sys
from collections import defaultdict, namedtuple


class MC:
    def __init__(self,env, verbose = False):
        self.env = env
        self.verbose = verbose
        
    def _collect_samples(self):
        """
        collects multiple samples of experiences from the environment.
        Inputs:
        -------
            policy       : A policy under which to sample. 
                           a function that takes `state` as an input and returns an action
            num_episodes : Number of episodes to run (default = 100)
            
        Output:
        -------
            V : Value function approximated under policy [nS X 1] numpy matrix
            
        """
        samples = {}
        for e in range(self.num_episodes):
            samples[e] = {'episode':[],'returns':[]} 
            self.env.reset()
            episode = []
            is_done = False
            while not is_done:
                state = self.env.s
                action = self.policy(state)
                sp,r,is_done = self.env.step(action)
                episode.append((state,r,sp))
            G = 0
            states_and_returns = []
            for state, reward,_ in reversed(episode):
                G = reward + self.gamma*G
                states_and_returns.append((state,G))
            samples[e].update([('episode',episode), ('returns',states_and_returns[::-1])])
        
        self.samples = samples
        
        return samples
    
    def _compute_Gtreturns(self, episode):
        """
        Computes cumulative reward for any given sample of experience.
        """
        G = 0
        Gt = []
        for state,_, reward,_ in reversed(episode):
            G = reward + self.gamma*G
            Gt.append(G)
        
        Gt = Gt[::-1] # get a correct sequence of Gt wrto visit time
        
        return Gt
    
    def _sample_episode(self):
        """
        collects a samples of experiences from the environment.
        requires the self to contain a policy
        
        the self class should contain env, policy
            
        """
        self.env.reset()
        episode = []
        is_done = False
        while not is_done:
            state = self.env.s
            action = self.policy(state)
            sp,r,is_done = self.env.step(action)
            episode.append((state,action,r,sp))
            
            if len(episode)>1000:
                pdb.set_trace()

        return episode 
        
class policy_evaluation(MC):
    def __init__(self,env,policy,num_episodes = 100, gamma = 1.0):
        """
        A subclass to perform policy evaluation using Monte-Carlo methods.
        This class contians the following types of MC policy evaluations:
            1. First Visit MC updates
            2. Every Visit MC updates
            3. Incremental MC updates
        
        Inputs:
        -------
            env  : Open AI format GYM environment
            policy : A policy to sample the experiences from
            num_episodes : Number of samples to draw from the environment (default, 100)
            gamma : Discount factor to compute returns (default, 1.0)
        """
        super(policy_evaluation, self).__init__(env)
        self.policy = policy
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.samples = self._collect_samples()
        self.V = defaultdict(float)
        self.N = defaultdict(int)
        self.S = defaultdict(int)
        
        
    def first_visit(self):
        """
        Performs first visit Monte-Carlo Policy Evaluation
        
        Inputs:
        ------
            should work with the initialized policy_evaluation class

        Output:
        ------
            V : Value function at "visited" states only
        """
        for e in self.samples:
            states = [i[0] for i in self.samples[e]['returns']]
            returns = [i[1] for i in self.samples[e]['returns']]
            for i,state in enumerate(states):
                first_occurance = np.argwhere(np.asarray(states)==state)[0][0]
                self.N[state] += 1
                self.S[state] += returns[first_occurance]
        for state in self.N:
            self.V[state] = self.S[state]/self.N[state]
            
        return self.V
    
    def every_visit(self):
        """
        Performs every visit Monte-Carlo Policy Evaluation
        
        Inputs:
        ------
            should work with the initialized policy_evaluation class

        Output:
        ------
            V : Value function at "visited" states only
        """
        for e in self.samples:
            states = [i[0] for i in self.samples[e]['returns']]
            returns = [i[1] for i in self.samples[e]['returns']]
            for i,state in enumerate(states):
                self.N[state] += 1
                self.S[state] += returns[i]
        for state in self.N:
            self.V[state] = self.S[state]/self.N[state]
            
        return self.V       
                
    def incremental(self, alpha=None):
        """
        Performs every visit Monte-Carlo Policy Evaluation
        
        Inputs:
        ------
            should work with the initialized policy_evaluation class
            alpha : Alpha value c.f. Documents/DavidSilver/Lecture4-Model-Free-Prediction.pdf
                    default is 1/Number of state visits

        Output:
        ------
            V : Value function at "visited" states only
        """
        for e in self.samples:
            states = [i[0] for i in self.samples[e]['returns']]
            returns = [i[1] for i in self.samples[e]['returns']]
            for i,state in enumerate(states):
                self.N[state] += 1
                if not self.V[state]:
                    self.V[state] = 0
                if alpha is None:
                    alpha = 1/self.N[state]
                    
                self.V[state] = self.V[state] + alpha*(returns[i]- self.V[state])
            
        return self.V              
    
class GLIE(MC):
    """
    Monte-Carlo GLIE Control Method.
    """
    def __init__(self,env,num_episodes = 100, gamma = 1.0, eps_decay = None):
        super(GLIE, self).__init__(env)
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.N = defaultdict(int)
        self.S = defaultdict(int)
        
        self.trace = namedtuple("trace",["lengths", "rewards","epsilon"])
        self.trace = self.trace(lengths=[],rewards=[],epsilon = [])
        
        if eps_decay is None:
            self.eps_decay = 10/self.num_episodes
            
    def _eps_greedy(self,state):
        """
        Episilon greedy policy
        Inputs:
        ------
            state  : satate at which an action needs to be taken
            (uses) self.Qsa    : Q(s,a) values as a dictornary with (state, action) keys
            (uses) self.epsilon : epsilon value used for episoln-greedy policy
            
        Output:
        ------
            Action : Action to be take at `state` using `Q(s,a)`
        """
        coin = np.random.rand()
                
        nA = self.env.action_space.n
        qvalues = np.zeros(nA)
        for a in range(nA):
            qvalues[a] = self.Qsa[state,a]
        
        if coin<self.epsilon:
            action = np.random.randint(nA)
        else:
            action = np.argmax(qvalues)
            
        return action

    def solve(self):
        """
        Solve the method upto the number of episodes
        """
        self.Qsa = defaultdict(float)
        self.policy = self._eps_greedy
        
        for e in range(self.num_episodes):
            self.epsilon = np.exp(-self.eps_decay*e)
            episode = self._sample_episode()
            Gt = self._compute_Gtreturns(episode)
            self.trace.rewards.append(Gt[0])
            self.trace.lengths.append(len(episode))
            self.trace.epsilon.append(self.epsilon)
            
            for time,(st,at,rt,sp) in enumerate(episode):
                self.N[st,at] += 1
                self.Qsa[st,at] += (1/self.N[st,at])*(Gt[time] - self.Qsa[st,at])
            
            if self.verbose:
                if e%100==0:
                    print('{}/{} episodes finished'.format(e,self.num_episodes))
                
                
        return self.Qsa
        
class svfa(MC):
    """
    An MC class for State Value Function Approximation (SVFA).
    fa which is a function-approximator, should have the following:
    predict : Given a state, return state-value function approximation
    update : Given set of training data update function approximator.
     
    Inputs:
    -------
        env  : Environment class
        fa. : Function approximator
        policy : policy under which to sample experience from
        num_episodes. : Number of episodes (default, 100)
        gamma  : Discount factor (default, 1.0)
        verbose : To print updates regularly (default, False)
    Attributes:
    -----------

    Methods:
    --------
        solve :  Solves MC value function updates using function approximator
    
    """
    def __init__(self,env,policy,fa,num_episodes = 100, gamma = 1.0, verbose=False):
        super(svfa, self).__init__(env, verbose = verbose)
        self.policy = policy
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.samples = self._collect_samples()
        self.V = defaultdict(float)
        self.N = defaultdict(int)
        self.S = defaultdict(int)  
        self.fa = fa
        
        self.trace = namedtuple("trace",["lengths", "rewards","epsilon"])
        self.trace = self.trace(lengths=[],rewards=[],epsilon = [])
        
        
    def solve(self):
        for e in self.num_episodes:
            episode = self._sample_episode()
            Gt = self._compute_Gtreturns(episode)
            
            self.trace.rewards.append(Gt[0])
            self.trace.lengths.append(len(episode))
            self.trace.epsilon.append(self.epsilon)
            
            states = []
            for time,(st,at,rt,sp) in enumerate(episode):
                states.append(st)
            
            # Update our function approximator with training data as {<St,Gt>}_{t=1..T}
            self.fa.update(states,Gt) 
            
            if self.verbose:
                if e%100==0:
                    print('{}/{} episodes finished'.format(e,self.num_episodes))
            
        return self.fa
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
                
            
                
                
                
            
        