"""
Implements the Temoral Difference Learning algorithm.
This solver contains TD-Lambda methods based on Prof.David Silver Lecture slides.
Note that TD-Lambda can be used as other solver by setting the n-step return and \gamma value accordingly

(c) copyright Kiran Vaddi 02-2020
"""
import numpy as np
import pdb
from collections import defaultdict

class TD:
    def __init__(self,env,policy):
        self.env = env
        self.policy = policy
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.l = l
        self.alpha = alpha
        
    def _collect_samples(self):
        """
        collects multiple samples of experiences from the environment.
            
        """
        samples = {}
        for e in range(self.num_episodes):
            self.env.reset()
            episode = []
            is_done = False
            while not is_done:
                state = self.env.s
                action = self.policy(state)
                sp,r,is_done = self.env.step(action)
                episode.append((state,r,sp))
            samples[e] = episode
        
        self.samples = samples
        
        return samples  
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

        return episode 
    
    def _compute_lambda_return(self,episode,V):
        """
        Computes lamda return according to the following:
        
        lamda-return using:
        \[G_{t}^{\lambda} = (1-\lambda)*\Bigsum_{n=1}^{n=inf}\lambda^{n-1}G_t^n\]
    
        """
        n = len(episode)
        Gtlambda = defaultdict(float)
        for step in range(n):
            Gtn = self._compute_nstep_return(episode,V,n=step)
            for time in Gtn:
                Gtlambda[time] += (1-self.l)*(self.l**step)*Gtn[time] 

        return Gtlambda

             
    def _compute_nstep_return(self,episode,V, n = None):
        """
        Computes n-step return according to the following:
        
        n-step return using:
        \[G_t^n = R_t+1 + \gamma*R_t+2 + ... +\gamma^{n-1}+\gamma^n*V(S_t+n)\]
        """
        if n is None:
            n = len(episode)
        
        E = []
        for state, reward,next_state in episode:
            E.append((state,reward,next_state))
        Gn = defaultdict(float)
        for ind in range(len(E)):
            nsteps = E[ind:ind+n+1] # We use a step morethan what is asked but it is a hack
            Gtn = 0
            for i,(state,reward,next_state) in enumerate(nsteps):
                Gtn += (self.gamma**i)*reward
            tostate = nsteps[-1][2]
            
            Gn[ind] = Gtn + (self.gamma**n)*V[tostate]
            
        
        return Gn
    
    def _compute_eligibility_trace(self,episode):
        """
        Computes eligibility trace of any state using the following:
        
        \[E_t(s) = \gamma*\lambda*E_{t-1}(s) + \delta_{S_t,s}\]
        
        Inputs:
        ------
            episode : An episode from the environment experience
        Outputs:
        -------
            E  : Eligibility trace. A dictornary with E[time,state] keys
        
        """
        E = defaultdict(float)
        states = [i[0] for i in episode]
        for ind,(state,_,_) in enumerate(episode):
            E[ind,state] = self.gamma*self.l*E[ind-1,state] + 1
        
        return E

class tabular(TD):
    def __init__(self,env,policy,gamma=1.0, l = 0.0, alpha=1.0, verbose = False):
        self.gamma = gamma
        self.l = l
        self.alpha = alpha
        self.verbose = verbose
        super(tabular, self).__init__(env,policy)
        
    def forward_view(self):
        """
        Returns a state value function approximation using Forward view TD-lambda update.
        
        Outputs:
        --------
            Vpi : State value function under policy \pi
        """
        samples = self._collect_samples()
        V = defaultdict(float)
        for e in samples:
            episode = samples[e]
            states = [i[0] for i in episode]
            Gtlambda = self._compute_lambda_return(episode,V)
            for time,state in enumerate(states):
                V[state] = V[state] + self.alpha*(Gtlambda[time]-V[state])
                
        return V
    
    def tdn(self,n=0):
        """
        Perform a TD(n) updates using the following:
        Computes TD-error using n-step return:
        \[ \delta_t = G_t^n - V(S_t)\]
        
        Update the state-value function using the following:
        \[V(S_t) = V(S_t) + \alpha*(\delta_t)\]
        
        Inputs:
        -------
            n  : n-step return to be calulcated (default, n=0)
        Outputs:
        -------
            Vpi : State-value function under policy \(\pi\) a dictonary
        
        """
        samples = self._collect_samples()
        V = defaultdict(float)
        for e in samples:
            episode = samples[e]
            states = [i[0] for i in episode]
            Gtn = self._compute_nstep_return(episode,V,n=n)
            
            for time,state in enumerate(states):
                V[state] = V[state] + self.alpha*(Gtn[time]-V[state])
                
        return V
    
    def backward_view(self, n=0):
        """
        Performs backward view TD-lambda using the following:
        
        Compute eligibility trace:
        \[E_t(S) = \gamma*\lambda*E_{t-1}(s) + \delta_{S_t,s}\]
        
        TD Error:
        \[\delta_t = R_{t+1} + \gamma*V(S_{t+1}) - V(S_t)\]
        
        Make the update using:
        \[V(s) = V(s) + \alpha*\delta_t*E_t(s)\]
        """
        samples = self._collect_samples()
        V = defaultdict(float)
        for e in samples:
            episode = samples[e]
            T = len(episode)
            E = self._compute_eligibility_trace(episode)
            
            states = [i[0] for i in episode]
            
            Gtn = self._compute_nstep_return(episode,V,n=n)
            
            for t in range(T):
                current_state,_,_ = episode[t]
                delta_t = Gtn[t]-V[current_state]
                for state in V:
                    V[state] = V[state] + self.alpha*delta_t*E[t,state]
                
        return V            
            
                               
class svfa(TD):
    """
    A TD class for State Value Function Approximation (SVFA).
    fa which is a function-approximator, should have the following methods:
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
                'method' : Two methods of TD solutions available:
                          'TD0' : Updates the target as R+\gamma*\hat{V}
                          'TDlambda' : Updates to target as G_{t}^{\lambda}
    
    """
    def __init__(self,env,policy,fa,num_episodes = 100,\
                 gamma=1.0, l = 0.0, alpha=1.0, verbose = False):
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.l = l
        self.alpha = alpha
        self.verbose = verbose
        self.fa = fa
        super(svfa, self).__init__(env,policy)
        
        self.V = defaultdict(float)
        self.N = defaultdict(int)
        self.S = defaultdict(int)  
                
        self.trace = namedtuple("trace",["lengths", "rewards","epsilon"])
        self.trace = self.trace(lengths=[],rewards=[],epsilon = [])
        
    def solve(self, method = 'TD0'):
        for e in self.num_episodes:
            episode = self._sample_episode()

            states = []
            targets = []
            for time,(st,at,rt,sp) in enumerate(episode):
                states.append(st)
                
                if method is 'TD0':
                    target = rt + self.gamma*self.fa.predict(st)
                elif method is 'TDlambda':
                    self.V[st] = self.fa.predict(st) 
                    Gtlambda = self._compute_lambda_return(episode,self.V)
                    target = Gtlambda[st]
                    
                targets.append(target)

                # Update our function approximator with 
                # training data as {<St,target_t>}_{t=1..T}
                fa.update(state,target) 
            
            self.trace.rewards.append(np.sum(targets))
            self.trace.lengths.append(len(episode))

            if self.verbose:
                if e%100==0:
                    print('{}/{} episodes finished'.format(e,self.num_episodes))
            
        return self.fa           
        
            
        
        
        
        
        
        
        
        
        
        
        
        
        
        