import numpy as np
import pdb
import sys
if '../' not in sys.path:
    sys.path.append('../')
    
import torch
from torch.distributions import Categorical
import torch.optim as optim
import torch.nn.functional as F

from collections import namedtuple

def _pgpolicy(env,model,state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = model(state.to(device))
    dist = Categorical(probs)
    
    action = dist.sample()
    
    return action.item()

class pgbase:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = 123
        torch.manual_seed(self.seed)
        trace = namedtuple("trace",["score", "steps","loss", "e"])
        self.trace = trace(score=[],steps=[],loss = [],e=[])
        
    def sample_episode(self,env,policy):
        state = env.reset()
        rewards,ep_reward,steps = [],0,0
        log_probs, trace, values = [], [], []
        while True:
            action,pinfo = policy(state)
            state, reward, is_done, info = env.step(action)
            ep_reward += reward
            steps += 1
                
            trace.append((state,action))
            if is_done or steps>10000:
                break
            else:
                rewards.append(reward)
                log_probs.append(pinfo['log_prob'])
                if 'value' in pinfo:
                    values.append(pinfo['value'])
                
        info = {'score':ep_reward,'steps':steps,\
                'log_probs':log_probs,'trace':trace,'values':values}
        
        return rewards, info
    
    def _pgpolicy(self,env,model,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            probs = model(state.to(self.device))
        dist = Categorical(probs)

        action = dist.sample()

        return action.item()
    
    def logtrace(self,info):
        """ 
        Give a dictonary in info logs the following about the episode:
            1. score : Episode reward
            2. Steps : Episode length
            3. e: Episode number
            4. loss : loss value of the model
        """
        self.trace.score.append(info['score'])
        self.trace.steps.append(info['steps'])
        self.trace.loss.append(info['loss'])
        self.trace.e.append(info['e'])
        
class reinforce(pgbase):
    """
    Standard REINFORCE algorithm without baseline reduction.
    Inputs:
    -------
        env : environment
        model :  model that takes a state as an input and return probability of the action set
        num_episodes : Number of episodes to run (default,100)
        verbose : logger outputs (boolian: default False)
    """
    
    def __init__(self,env,model, optimizer, verbose=True):
        super(reinforce,self).__init__()
        self.env = env
        self.env.seed(self.seed)
        self.model = model
        self.optimizer = optimizer
        self.verbose = verbose
        self.eps = np.finfo(np.float32).eps.item()
        
    def policy(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(state.to(self.device))
        dist = Categorical(probs)

        action = dist.sample()

        return action.item(), {'log_prob':dist.log_prob(action)}
        
    def update(self,rewards,log_probs):
        gt = 0
        G = []
        for r in rewards[::-1]:
            gt = r + self.gamma * gt
            G.insert(0, gt)
        
        G = torch.tensor(G)
        G = (G - G.mean()) / (G.std() + self.eps)
        loss = []
        for gt,log_prob in zip(G,log_probs):
            loss.append(-log_prob * gt)
        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum().to(self.device)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def solve(self,gamma=0.99, num_episodes=100, reward_threshold=150):
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.thresh = reward_threshold
        
        e = 1
        avgreward = 0
        while True:
            rewards, info = self.sample_episode(self.env,self.policy)
            loss = self.update(rewards,info['log_probs'])
            avgreward += info['score']
            info.update({'e':e,'loss':loss})
            self.logtrace(info)
            
            if e%self.num_episodes==0:
                avr = avgreward/self.num_episodes
                print('{} episodes finished with latest loss: {:.2f}, average reward: {:.2f}'.format(e,loss,avr))
                if avr>self.thresh or e>1000:
                    break
                else:
                    avgreward = 0
            e += 1
        output = {'model':self.model,'trace':self.trace._asdict()}    
        return output
    
# Advantage Actor Critic
from torch.distributions import Multinomial,Normal

class a2c(pgbase):
    """
    Advantage actor-critic algorithm
    Inputs:
    -------
        env : environment
        model :  model that takes a state as an input and return probability of the action set
        num_episodes : Number of episodes to run (default,100)
        verbose : logger outputs (boolian: default False)
        
    Ref:
    ----
        1. https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#actor-critic
        2. https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
    """
    
    def __init__(self,env,model, optimizer, policy = None,checkpoint=None, verbose=True):
        super(a2c,self).__init__()
        self.env = env
        self.env.seed(self.seed)
        self.model = model
        self.optimizer = optimizer
        self.verbose = verbose
        self.eps = np.finfo(np.float32).eps.item()
        
        if policy is 'categorical':
            self.policy = self._select_action_categorical
        elif policy is 'multinormal':
            self.policy = self._select_action_multinormal
        elif policy is None:
            raise KeyError('Please specify a policy')
        else:
            raise KeyError('Policy {} is not implemented'.format(policy))
            
        if checkpoint is None:    
            self.checkpoint = 'a2cmodel.pth'
        else:
            self.checkpoint = checkpoint

        
    def _refine_returns(self,rewards):
        gt = 0
        G = []
        for r in rewards[::-1]:
            gt = r + self.gamma * gt
            G.insert(0, gt)
        
        G = torch.tensor(G)
        G = (G - G.mean()) / (G.std() + self.eps)
        
        return G.float()
    
    def _select_action_categorical(self,state):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state = state.float().unsqueeze(0)

        # model needs to output action probabilities and value funcion given a state
        probs, value = self.model(state.to(self.device))
        dist = Categorical(probs)

        action = dist.sample()

        return action.item(), {'log_prob':dist.log_prob(action),'value':value} 

    def _select_action_multinormal(self,state):
        if not torch.is_tensor(state):
            state = torch.from_numpy(state).float().unsqueeze(0)
        else:
            state = state.float().unsqueeze(0)

        # model needs to output action probabilities and value funcion given a state
        action_values, state_value = self.model(state.to(self.device))
        action = []
        log_probs = torch.zeros(action_values.shape[1])
        for i,a in enumerate(action_values[0]):
            dist = Normal(a,torch.tensor([1.0]))
            at = dist.sample().item()
            log_probs[i] = dist.log_prob(at)
            action.append(at)

        return action, {'log_prob':log_probs.sum(),'value':state_value[0]} 
    
    def update(self, rewards,log_probs,values):
        G = self._refine_returns(rewards)
        loss = []
        for gt,log_prob, value in zip(G,log_probs,values):
            advantage = gt - value
            thetaloss = -log_prob * advantage
            wloss = F.smooth_l1_loss(value, torch.tensor([gt]).to(self.device))
            loss.append(thetaloss + wloss)
        
        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum().to(self.device)
        if loss.item()<-5000:
            pdb.set_trace()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
            
    def solve(self,gamma=0.99, num_episodes=100, reward_threshold=150, episode_threshold = 5000):
        self.gamma = gamma
        self.num_episodes = num_episodes
        self.thresh = reward_threshold
        self.maxeps = episode_threshold
        
        e = 1
        avgreward = 0
        while True:
            rewards, info = self.sample_episode(self.env,self.policy)
            loss = self.update(rewards,info['log_probs'], info['values'])
            avgreward += info['score']
            info.update({'e':e,'loss':loss})
            self.logtrace(info)
            
            if e%self.num_episodes==0:
                torch.save({'episode': e,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'loss': loss,}, self.checkpoint)
                
                checkpoint = torch.load(self.checkpoint)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                avr = avgreward/self.num_episodes
                print('{} episodes finished with latest loss: {:.2f}, average reward: {:.2f}'.format(e,loss,avr))
                if avr>self.thresh or e>self.maxeps:
                    break
                else:
                    avgreward = 0
            e += 1
            
        output = {'model':self.model,'trace':self.trace._asdict()}    
        
        return output
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    