"""
Ramp function class and methods

    Author: William Duncan

"""

import matplotlib.pyplot as plt
import numpy as np

class RampFunction:

    def __init__(self, sign, L, Delta, theta, eps):
        self.sign = sign 
        self.Delta = Delta
        self.L = L
        self.theta = theta
        self.eps = eps
        

    def __call__(self,x):
        """
        Evaluation method for the ramp function. 
        When eps == 0, returns Delta/2 at x=theta.
        """
        sign = self.sign
        Delta = self.Delta
        L = self.L
        theta = self.theta
        eps = self.eps
        if eps != 0:
            m = Delta/(2*eps)
        else: 
            m=0
        H = lambda x: np.heaviside(x,.5)

        return L*(H(sign)*H((theta-eps) - x) + H(-sign)*(H(x-(theta+eps)))) \
            + (L + Delta)*(H(sign)*H(x-(theta+eps)) + H(-sign)*H((theta-eps)-x)) \
            + (L + Delta/2 + sign*m*(x-theta))*H(x-(theta-eps))*H((theta+eps)-x)*int(eps>0)

    def dx(self,x):
        """Computes the derivative at x. Returns nan at the corners. """
        eps = self.eps
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


    def plot(self, xlim=None):
        theta = self.theta
        eps = self.eps

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
        yvals = self(xvals)
        if (eps == 0): 
            if (xlim[0] < theta):
                yvals[1] = self(theta-1)
                yvals[2] = self(theta+1)
            else:
                yvals[0] = self(theta+1)

        plt.plot(xvals, yvals)

        
            


    