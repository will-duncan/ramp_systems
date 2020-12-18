"""
RampModel class and methods

    Author: William Duncan

"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
from ramp_systems.ramp_function import *

def power_set(iterable):
    pool = tuple(iterable)
    n = len(pool)
    for k in range(n+1):
        for combo in itertools.combinations(pool,k):
            yield combo
    

class RampSystem:

    def __init__(self,Network,L,Delta,theta,gamma):
        """
        Inputs:
            Network - (_dsgrn.Network)
            L,U,theta - (numpy array) Each is an NxN array of ramp function parameters
            gamma - (numpy array) Length N vector of degradation rates 
        """
        self.Network=Network
        N = Network.size()
        self.L = np.array(L)
        self.Delta = np.array(Delta)
        self.theta = np.array(theta)
        self.gamma = np.array(gamma).reshape([N,1])
        
        self._zero = np.zeros([Network.size(),Network.size()])
        self._set_func_array()
        self._set_R()
        
        #self._set_vector_field()

    def __call__(self,x,eps=[]):
        if len(eps) == 0:
            eps = self._zero
        x = np.array(x).reshape([self.Network.size(),1])
        return -self.gamma*x + self.R(x,eps)

    # def _set_vector_field(self):
    #     self.vector_field = lambda x,eps: -self.gamma*x + self.R(x,eps)



    def _set_R(self):
        """
        Creates the 'R' attribute. R is an np array with dimensions [Network.size(),1] defined by 
        x[j]'= -gamma[j]*x[j] + R[j]
        """
        Network = self.Network
        
        def R(x,eps,Network = Network):
            R_array = np.zeros([Network.size(),1])
            for i in range(Network.size()):
                cur_prod = 1
                for source_set in Network.logic(i):
                    cur_sum = 0
                    for j in source_set:
                        cur_sum = cur_sum + self.func_array(x,eps)[i,j]
                    cur_prod =  cur_prod*cur_sum 
                R_array[i] = cur_prod
            return R_array
        N = Network.size()
        self.R = lambda x,eps=self._zero: R(x,eps)


    def _set_func_array(self):
        """
        Creates the func_array attribute.
        """
        Network = self.Network
        N = Network.size()
        def func_array(x, eps,L=self.L, Delta=self.Delta, theta=self.theta,Network = Network):
            x = np.array(x)
            eps = np.array(eps)
            N = Network.size()
            F = np.zeros([N,N])
            for i in range(Network.size()):
                for j in Network.inputs(i):
                    sign = 1 if Network.interaction(j,i) else -1
                    Rij = RampFunction(sign,L[i,j],Delta[i,j],theta[i,j])
                    F[i,j] = Rij(x[j],eps[i,j])
            return F
        
        self.func_array = lambda x,eps=self._zero: func_array(x,eps)


    
    def get_W(self):
        """
        Output:
             W - (list) length Network.size() list of length 
                 Targets(j) sorted lists. W[j] is the output of get_W_j
        """
        W = []
        for j in range(self.Network.size()):
            W_j = self._get_W_j(j)        
            W.append(W_j)    
        return W
                

    def _get_W_j(self,j):
        """
        Input: 
            j - (int) index of a node
        Output: 
            W_j - (list) The set of values of Lambda_j union {0,np.inf}, sorted
        """
        W_j_set = {0,np.inf}
        N = self.Network.size()
        R_j = lambda x: self.R(x)[j,0]
        inputs = self.Network.inputs(j)
        test_point = np.zeros(N)
        x_low = 1/2*self.theta
        x_high = 2*self.theta
        for low_pattern in power_set(inputs):
            low_pattern =  list(low_pattern)
            high_pattern = list(set(inputs) - set(low_pattern))
            test_point[low_pattern] = x_low[j,low_pattern]
            test_point[high_pattern] = x_high[j,high_pattern]
            W_j_set = W_j_set.union({R_j(test_point)})
        W_j = list(W_j_set) 
        W_j.sort()
        return W_j

    def _get_B_jp(self,j,W_j,p):
        """
        Input:
            j - (int) index of a node
            p - (int) requires 0<p<len(W_j)
            W_j - (list) output of get_W_j(j)
        Output:
            B_jp - (list) indices of thresholds that lie between W_j[p] and W_j[p-1] 
        """
        theta = self.theta
        Network = self.Network
        B_jp = [i for i in Network.outputs(j) if (theta[i,j] > W_j[p-1] and theta[i,j]<W_j[p])]
        return B_jp

    def _get_B_j(self,j,W_j):
        """
        Input:
            j - (int) index of a node
            W_j - (list) output of get_W_j(j)
        Output:
            B_j - (list) list of lists of indices. For 0<p<len(W_j), B_j[p] is the output of B_jp
                   B_j[0] = []
        """
        B_j = [[]]
        for p in range(1,len(W_j)):
            B_jp = self._get_B_jp(j,W_j,p)
            B_j.append(B_jp)
        return B_j
            
    def _get_B(self,W):
        B = []
        for j in range(self.Network.size()):
            B.append(self._get_B_j(j,W[j]))
        return B

    def _get_eps_jp(self,j,W_j,B_j,p):
        """
        Requires Network.inputs(j) != []
        Input:
            j - (int) index of a node
            W_j - (list) output of get_W_j(j)
            B_j - (list of lists) output of get_B_j(j,W_j)
            p - (int) requires 0<p<len(W_j) 
        Output:
            eps_jp - (NxN numpy array) Choice of eps which minimizes slope of 
                     m[i,j] for i in B_jp. If i not in B_jp or k != j 
                     then eps_jp[i,k] == 0
        """
        N = self.Network.size()
        eps_jp = np.zeros([N,N])
        B_jp = B_j[p]
        if B_jp == []:
            return eps_jp
        
        Delta = self.Delta
        theta = self.theta
        gamma = self.gamma 
        ##construct D_tilde
        theta_index = {theta[i,j]:i for i in B_jp}
        Theta_j = [theta[i,j] for i in B_jp]
        Theta_j.sort()
        i = lambda q: theta_index[Theta_j[q]]
        theta_distance = \
            lambda q: (theta[i(q),j] - theta[i(q-1),j])/(Delta[i(q),j] + Delta[i(q-1),j])
        D_tilde = min([theta_distance(q) for q in range(1,len(Theta_j))],default=np.inf)
        ##construct D
        k = len(B_jp)-1
        D_list = [D_tilde]
        if p != 1:  #W_j[p-1] != 0
            W_pm1_dist = (gamma[j]*theta[i(0),j] - W_j[p-1])/(gamma[j]*Delta[i(0),j])
            D_list.append(W_pm1_dist)
        if p != (len(W_j)-1):  #W_j[p] != inf
            W_p_dist = (W_j[p] - gamma[j]*theta[i(k),j])/(gamma[j]*Delta[i(k),j])
            D_list.append(W_p_dist)
        D = min(D_list)
        ##construct eps_jp
        for i in B_jp:
            eps_jp[i,j] = Delta[i,j]*D
        return eps_jp

    def _get_eps_j(self,j,W_j,B_j):
        N = self.Network.size()
        eps_j = np.zeros([N,N])
        for p in range(1,len(W_j)):
            eps_j += self._get_eps_jp(j,W_j,B_j,p)
        return eps_j

    def optimal_eps(self):
        """
        Output:
            eps - (numpy array) A choice of eps which minimizes the maximum slope
                  for each j among eps which satisfiy eps'<eps implies (Z,eps')
                  is strongly equivalent to (Z,0). This guarantees all equilibria 
                  of DSGRN(Z) have corresponding equilibria in R(Z,eps'). 
        """
        Network = self.Network
        W = self.get_W()
        B = self._get_B(W)
        N = Network.size()
        eps = np.zeros([N,N])
        for j in range(N):
            eps += self._get_eps_j(j,W[j],B[j])
        return eps
            
    def is_regular(self,eps=None):
        """
        Returns True if the switching parameter is regular and False otherwise.
        This is a strong form of regularity in that it checks if 
        gamma(theta +/- eps) = Lambda_j for all values of Lambda_j, not just 
        Lambda_j(kappa)
        """
        if eps is None:
            eps = self._zero

        N = self.Network.size()
        L = self.L
        Delta = self.Delta
        theta = self.theta
        gamma = self.gamma
        if sum(gamma > 0) < N:
            return False
        W = self.get_W()
        for j in range(N):
            for i in self.Network.outputs(j):
                if L[i,j]<=0 or Delta[i,j]<=0 or theta[i,j]<=0:
                    return False
                W_j = W[j][1:]
                if gamma[j]*(theta[i,j] + eps[i,j]) in W_j or \
                    gamma[j]*(theta[i,j] - eps[i,j]) in W_j:
                    return False
        return True




    


