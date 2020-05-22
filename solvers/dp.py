"""
Dynamic Programming tools for solving MDPs.
Implemented algorithms:
    1. Value iteration
    2. Policy itereation
    3. Policy evaluation

ToDos:
-----
Usematrices isn't working with the current style of transisition probabilities


(c) copyright Kiran Vaddi 2020

"""
import numpy as np
from scipy.linalg import norm
import pdb

class dp:
    def __init__(self,env, gamma=1.0):
        """
        Initiate the dynamic programming class with the following:

        Inputs:
        -------
            env  : Environment with the following compulsory fields
                    env.P - Transisition probabilities P(s'|s,a)
                    env.nS - Number of states in the MDP
                    env.nA - Number of actions possible at each state in the MDP
            gamma : Discount factor for rewards computation (default = 0.5)

        """
        self.gamma = gamma
        self.env = env

    def policy_evaluation(self,policy, theta = 1e-2, usematrix = False):
        """
        This function performs the policy evaluation.
        The goal is to evaluate a policy \pi using iterative application of Bellman equation.
        For given number of iterations, we look at each possible succesive states s'
        from any given state s, and update the value function \(v(s)\).

        The updates are as follows:
            (iterative): \[V_{k+1}(s) = \BigSum_{a\in A} \pi(a|s)\Big(P_ss'^a*( R_s^a + V_k(s'))\Big)\]
            (matrix)   : V_{k+1} = R_s^{\pi} + \gamma*P^{\pi}*V_k

        Inputs:
        -------
        policy  : Policy \pi to be evaluated as a [nS X nA] numpy array
        theta   : Theta value to stop itetarations (default = 1e-3)
        usematrix : (True) to use matrix formulation
                    (False, recommended) to use iterative updates over each state

        Ouput:
        ------
        V : Value function V(s) vector of length env.nS

        c.f. Documents/DavidSilver/Lecture3-DP.pdf Slide - Iterative Policy evaluation
        """

        Vold = np.zeros((self.env.nS,1))
        Vnew = np.zeros((self.env.nS,1))
        while True:
            # The Sutton and Barton way c.f.
            delta = 0
            
            # go over all the states
            for state in range(self.env.nS):
                # compute state lookahead
                v = 0
                for action in range(self.env.nA):
                    for tp in self.env.P[state][action]:
                        p,sp,r,d = tp[0]
                        v += policy[state,action]*p*(r + (self.gamma*Vold[sp]))
                Vnew[state] = v
                delta = max(delta, np.abs(Vnew[state]-Vold[state]))

            Vold = Vnew
            if delta<theta:
                break

        return Vnew

    def _get_matrices(self):
        """
        This function is used to tranform MDP into matrices form of MRPs for Dynamic Programming
        This is done using the following formulation:
            \[P^{\pi} = \BigSum_{a\inA} \pi(a|s)*P^{1}_{ss'}\]
            \[R^{\pi}\ = \BigSum_{a\inA} \pi(a|s)*R^a_s]
        Inputs:
        -------
            policy  : Policy under which we want the matrices to be as a
                      [nS X nA] numpy array
        Output:
        -------
            Matrices forms of P^{\pi} and R^{\pi}
            Ppi  : Transiition probability matrix under policy \pi
                   A matrix of size [nS X nS X nA]
            Rpi  : Reqards matrix under policy \pi
                   A matrix of size [nS X nA]
        """
        PaSSp = np.zeros((self.env.nS,self.env.nS,self.env.nS))
        RSa = np.zeros((self.env.nS,self.env.nA))

        for state in self.env.P:
            for action in self.env.P[state]:
                R = []
                for tp in self.env.P[state][action]:
                    p,sp,r,d = tp[0]                    
                    PaSSp[state,sp,action] = p
                    R.append(r)
                RSa[state,action] = R[action]

        return PaSSp, RSa

    def policy_iteration(self):
        """
        Policy iteration implementation.
        This method works by evaluating a greedy policy after each one-step-lookahead.
        \[q_{\pi}(s,a) = R_s^a + \gamma*(P_{ss'}^{a}*V_{\pi}(s'))\]
        Where \( V_{\pi}(s') \) is calculated using current policy \pi(s)
        A greedy maximization is used to change the policy \pi(s) = \argmax_{a} (q_{\pi}(s,a))

        inputs:
        -------
            should work just with the initilalized class

        outputs:
        -------
            Pi*  : Optimal policy [nS X 1]
            Vpi* : Optimal value function [nS X 1]
        """
        # initialization
        piSa = np.ones([self.env.nS, self.env.nA]) / self.env.nA
        Pi = np.argmax(piSa,axis=1)
                
        count = 1
        while True:
            # policy evaluation
            VpiSp = self.policy_evaluation(piSa)
            count = 1
            
            # policy improvement
            policy_stable = True
            for state in range(self.env.nS):
                action_lookahead = self._compute_action_lookahead(state,VpiSp)
                best_state_action = np.argmax(action_lookahead)
                current_state_action = Pi[state]
                if best_state_action != current_state_action:
                    policy_stable = False
                    piSa[state] = np.eye(self.env.nA)[best_state_action]
            
            Pi = np.argmax(piSa,axis=1)
            count += 1
            if policy_stable:
                break
            elif count>100:
                print('Unable to converge... might need debugging :(...')
                pdb.set_trace()
                
        return Pi,VpiSp
    
    def value_iteration(self, thresh = 1e-3):
        """
        Value iteration update the state-value function using bellman optimality equation:
        \[ v_{k+1}(s) = max_a [V_{k}(s')*P_{ss'}^{a} + R_s^a]\] 
        A policy can then be derived again using one-step-lookahead at each action:
        \[ \pi(s) = argmax_a [V_{k}(s')*P_{ss'}^{a} + R_s^a]\]
        
        Inputs:
        -------
            should work with the initialized dp class
        outputs:
        --------
            Pi : A policy vetor of size nS X 1
            V  : Value function a vector of size nS X 1
            
        c.f. Documents/DavidSilver/Lecture3-DP.pdf Value Iteration(2)
        """
        V = np.zeros(self.env.nS)
        Pi = np.zeros(self.env.nS)
        while True:
            delta = 0
            for state in range(self.env.nS):
                oldVS = V[state]
                action_lookahead = self._compute_action_lookahead(state,V)
                V[state] = max(action_lookahead)
                
                delta = max(delta, np.abs(V[state]-oldVS))
            
            if delta<thresh:
                break
        for state in range(self.env.nS):
            action_lookahead = self._compute_action_lookahead(state,V)
            Pi[state] = np.argmax(action_lookahead)
        
        return Pi.astype(int), V.reshape((self.env.nS,1))


    def _compute_action_lookahead(self,state,V):
        action_lookahead = np.zeros(self.env.nA)
        for action in range(self.env.action_space.n):
            a = 0
            for tp in self.env.P[state][action]:
                p,sp,r,d = tp[0]
                a += p*(self.gamma*V[sp] + r)
            action_lookahead[action] = a
        return action_lookahead
    

        
        








