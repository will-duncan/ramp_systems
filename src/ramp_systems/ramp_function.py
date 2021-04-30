"""
RampFunction class and methods. 

    Author: Will Duncan
"""
import matplotlib.pyplot as plt
import numpy as np

class RampFunction:

    def __init__(self, sign, L, Delta, theta):
        self.sign = sign 
        self.Delta = Delta
        self.L = L
        self.theta = theta

    def __eq__(self,other):
        if isinstance(other,self.__class__):
            return (self.sign == other.sign)\
                 and (self.Delta == other.Delta) \
                 and (self.L == other.L) \
                 and (self.theta == other.theta)
        else: 
            return False 

    def __repr__(self):
        sign = self.sign
        L = self.L
        Delta = self.Delta
        theta = self.theta
        return 'RampFunction(sign={!r},L={!r},Delta={!r},theta={!r})'.format(sign,L,Delta,theta)
            
        

    def __call__(self,x,eps = 0):
        """
        Evaluation method for the ramp function. 
        When eps == 0, returns np.nan at x==theta
        Input:
            x - Requires x is a list or a scalar
            eps - (scalar, optional)   
        """
        sign = self.sign
        Delta = self.Delta
        L = self.L
        theta = self.theta
        
        # if eps != 0:
        #     m = Delta/(2*eps)
        # else: 
        #     m=np.inf
        if (x < theta - eps and sign == 1) or (x > theta+eps and sign == -1):
            return L
        elif (x > theta+eps and sign == 1) or (x < theta - eps and sign == -1):
            return L + Delta
        else:   
            if eps == 0:
                return np.nan
            else:
                return L + Delta/2 + sign*Delta/(2*eps)*(x-theta)

            
        
    def evaluate(self,x,eps = 0):
        """
        Evaluate the ramp function on a list of values. 
        Input:
            x - list or numpy array
            eps - (optional) scalar value
        """
        x = np.array(x)
        sign = self.sign
        Delta = self.Delta
        L = self.L
        theta = self.theta
        out = np.zeros(x.shape)
        #x in singular domain
        mid_filter = np.logical_and(x >= theta - eps, x <= theta + eps)
        if eps != 0:
            out[mid_filter] = (L + Delta/2 + sign*Delta/(2*eps)*(x[mid_filter]-theta))
        else: 
            out[mid_filter] = np.nan
        #x outside singular domain
        low_filter = x < theta - eps
        high_filter = x > theta + eps       
        if sign == 1:
            out[high_filter] = (L+Delta)
            out[low_filter] = L
        elif sign == -1:
            out[high_filter] = L
            out[low_filter] = (L+Delta)
        return out

    def dx(self,x,eps = 0):
        """
        Computes the derivative at x. Returns nan at the corners.
        """
        x = np.array(x)
        theta = self.theta
        H = lambda x: np.heaviside(x,0)
        if eps != 0:
            m = self.Delta/(2*eps)
        else: 
            m = np.inf
        out = np.zeros(x.shape)
        out[np.logical_and(x>theta-eps, x<theta+eps)] =  self.sign*m 
        out[np.logical_or(x == theta+eps, x == theta-eps)] = np.nan
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
        yvals = self.evaluate(xvals,eps)
        if (eps == 0): 
            if (xlim[0] < theta):
                yvals[1] = self(theta-1,eps)
                yvals[2] = self(theta+1,eps)
            else:
                yvals[0] = self(theta+1,eps)

        plt.plot(xvals, yvals)