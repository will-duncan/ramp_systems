"""
Ramp function class and methods

    Author: William Duncan

"""

import matplotlib.pyplot as plt
import numpy as np

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
        Network = self.Network
        
        def R(x,eps,Network = Network):
            R_array = np.zeros([Network.size()])
            for i in range(Network.size()):
                cur_prod = 1
                for source_set in Network.logic(i):
                    print(source_set)
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





        
