"""
Ramp system class and methods

    Author: William Duncan

"""

import matplotlib.pyplot as plt
import numpy as np
import itertools

def power_set(iterable):
    pool = tuple(iterable)
    n = len(pool)
    for k in range(n+1):
        for combo in itertools.combinations(pool,k):
            yield combo


class RampFunction:

    def __init__(self, sign, L, Delta, theta):
        self.sign = sign 
        self.Delta = Delta
        self.L = L
        self.theta = theta
        

    def __call__(self,x,eps = None):
        """
        Evaluation method for the ramp function. 
        When eps == 0, returns Delta/2 at x=theta.
        """
        if eps == None:
            eps = 0
        sign = self.sign
        Delta = self.Delta
        L = self.L
        theta = self.theta
        
        if eps != 0:
            m = Delta/(2*eps)
        else: 
            m=0
        H = lambda x: np.heaviside(x,.5)

        return L*(H(sign)*H((theta-eps) - x) + H(-sign)*(H(x-(theta+eps)))) \
            + (L + Delta)*(H(sign)*H(x-(theta+eps)) + H(-sign)*H((theta-eps)-x)) \
            + (L + Delta/2 + sign*m*(x-theta))*H(x-(theta-eps))*H((theta+eps)-x)*int(eps>0)

    def dx(self,x,eps = None):
        """Computes the derivative at x. Returns nan at the corners. """
        if eps == None:
            eps = 0
        
        theta = self.theta
        H = lambda x: np.heaviside(x,0)
        if eps != 0:
            m = self.Delta/(2*eps)
        else: 
            m = 0
        if isinstance(x,np.ndarray):
            out =  m*H((theta+eps)-x)*H(x-(theta-eps)) 
            out[np.logical_or(x == theta+eps, x == theta-eps)] = np.nan
        else:
            if x == (theta-eps) or x == (theta + eps):
                out = np.nan
            else:
                out = m*H((theta+eps)-x)*H(x-(theta-eps)) 
                
        return out 


    def plot(self, eps = None,xlim=None):
        if eps == None:
            eps = 0
        theta = self.theta
        

        if xlim == None:
            xmin = max(0,theta - 2*eps)
            xmax = theta + 2*eps
            xlim = [xmin,xmax]
        
        if xlim[0] < theta - eps:
            xvals = [xlim[0], theta-eps, theta+eps, xlim[1]] 
        elif xlim[0] < theta + eps:
            xvals = [xlim[0], theta+eps, xlim[1]]
        else:
            xvals = [xlim[0], xlim[1]]
        xvals = np.array(xvals)
        yvals = self(xvals,eps)
        if (eps == 0): 
            if (xlim[0] < theta):
                yvals[1] = self(theta-1,eps)
                yvals[2] = self(theta+1,eps)
            else:
                yvals[0] = self(theta+1,eps)

        plt.plot(xvals, yvals)

    def __repr__(self):
        sign = self.sign
        L = self.L
        Delta = self.Delta
        theta = self.theta
        return 'RampFunction(sign={!r},L={!r},Delta={!r},theta={!r})'.format(sign,L,Delta,theta)
            

class RampSystem:

    def __init__(self,Network,L,Delta,theta,gamma):
        """
        Inputs:
            Network - (_dsgrn.Network)
            L,U,theta - (numpy array) Each is an NxN array of ramp function parameters
            gamma - (numpy array) Length N vector of degradation rates 
        """
        self.L = L
        self.Delta = Delta
        self.theta = theta
        self.gamma = gamma
        self.Network=Network
        self._set_func_array()
        self._set_R()
        self._set_vector_field()

    def __call__(self,x,eps):
        return self.vector_field(x,eps)

    def _set_vector_field(self):
        self.vector_field = lambda x,eps: -self.gamma*x + self.R(x,eps)



    def _set_R(self):
        """
        Creates the 'R' attribute. R is a [Network.size()] array defined by 
        x[j]'= -gamma[j]*x[j] + R[j]
        """
        Network = self.Network
        
        def R(x,eps,Network = Network):
            R_array = np.zeros([Network.size()])
            for i in range(Network.size()):
                cur_prod = 1
                for source_set in Network.logic(i):
                    cur_sum = 0
                    for j in source_set:
                        cur_sum = cur_sum + self.func_array(x,eps)[i,j]
                    cur_prod =  cur_prod*cur_sum 
                R_array[i] = cur_prod
            return R_array
        
        self.R = lambda x,eps: R(x,eps)


    def _set_func_array(self):
        """
        Creates the func_array attribute.
        """
        Network = self.Network
    
        def func_array(x, eps,L=self.L, Delta=self.Delta, theta=self.theta,Network = Network):
            N = Network.size()
            F = np.zeros([N,N])
            for i in range(Network.size()):
                for j in Network.inputs(i):
                    sign = 1 if Network.interaction(j,i) else -1
                    Rij = RampFunction(sign,L[i,j],Delta[i,j],theta[i,j])
                    F[i,j] = Rij(x[j],eps[i,j])
            return F

        self.func_array = lambda x,eps: func_array(x,eps)


    
    def get_W(self):
        """
        Creates the 'W' attribute. W is a length Network.size() list of length 
        Targets(j) sorted lists. Each list stores all possible values of Lambda_j.
        """
        W = []
        for j in range(self.Network.size()):
            W_j = self.get_W_j(j)        
            W.append(W_j)    
        self.W = W
                

    def get_W_j(self,j):
        W_j = [0,np.inf]
        N = self.Network.size()
        zero = np.zeros([N,N])
        R_j = lambda x: self.R(x,zero)[j]
        inputs = self.Network.inputs(j)
        test_point = np.zeros(N)
        x_low = 1/2*self.theta
        x_high = 2*self.theta
        for low_pattern in power_set(inputs):
            low_pattern =  list(low_pattern)
            high_pattern = list(set(inputs) - set(low_pattern))
            test_point[low_pattern] = x_low[j,low_pattern]
            test_point[high_pattern] = x_high[j,high_pattern]
            W_j.append(R_j(test_point)) 
        W_j.sort()
        return W_j

    def get_B_jp(self,j,W_j,p):
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

    def get_B_j(self,j,W_j):
        B_j = []
        for p in range(1,len(W_j)):
            B_jp = self.get_B_jp(j,W_j,p)
            B_j.append(B_jp)
        return B_j
            
    def get_B(self,W):
        B = []
        for j in range(0,self.Network.size()):
            B.append(self.get_B_j(j,W[j]))
        return B

    def get_eps_jp(self,j,W_j,B_j,p):
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
            lambda q: (theta[i[q],j] - theta[i[q-1],j])/(Delta[i[q],j] + Delta[i[q-1],j])
        
        D_tilde = min([theta_distance(q) for q in range(1,len(Theta_j))])
        ##construct D
        k = len(B_jp)
        D_list = [D_tilde]
        if p != 1:  #W_j[p-1] != 0
            W_pm1_dist = (gamma[j]*theta[i[1],j] - W_j[p-1])/(gamma[j]*Delta[i[1],j])
            D_list.append(W_pm1_dist)
        if p!=n:  #W_j[p] != inf
            W_p_dist = (W_j[p] - gamma[j]*theta[i[k],j])/(gamma[j]*Delta[i[k],j])
            D_list.append(W_p_dist)
        D = min(D_list)
        ##construct eps_jp
        for i in B_jp:
            eps_jp[i,j] = Delta[i,j]*D
        return eps_jp

    def get_eps_j(self,j,W_j,B_j):
        N = self.Network.size()
        eps_j = np.zeros([N,N])
        for p in range(1,len(W_j)):
            eps_j += self.get_eps_jp(j,W_j,B_j,p)
        return eps_j

    def get_optimal_eps(self):
        Network = self.Network
        W = self.get_W()
        B = self.get_B(W)
        N = Network.size()
        eps = np.zeros([N,N])
        for j in range(N):
            eps += get_eps_j(j,W[j],B[j])
        return eps
            




    


