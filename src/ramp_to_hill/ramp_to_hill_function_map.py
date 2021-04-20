
import numpy as np
from scipy.optimize import bisect
from ramp_to_hill.hill_system import *




class RampToHillFunctionMap:
    """
    Calling an instance of this class returns a HillParameter which corresponds 
    to the given RampFunction. 
    """
    def __init__(self,max_allowed_hill_coefficient = 1e8):
        self.max_allowed_hill_coefficient = max_allowed_hill_coefficient

    

    def get_hill_max_slope_func(self,RF):        
        return lambda n: hill_derivative_magnitude(\
            hill_second_derivative_root(RF.theta,n),RF.sign,RF.Delta,RF.theta,n)


    def __call__(self,RF,eps):
        """
        Get the HillParameter corresponding to RF at eps. This is a bijective map.

        Input: 
            RF - RampFunction class instance
            eps - positive scalar defining the width of the linear regime of a ramp 
                  function
        Output:
            HillParameter instance with
                L = RF.L
                Delta = RF.Delta
                sign = RF.sign
                theta = RF.theta
                n chosen so that the maximum slope of the hill function is the 
                slope of the ramp function plus the maximum slope of a hill function
                with hill coefficient 1
        """
        n_range = [1+1e-1,100] 
        hill_coefficient_1_slope = RF.Delta/RF.theta
        ramp_function_slope = RF.sign*RF.dx(RF.theta,eps)
        target_hill_slope = ramp_function_slope + hill_coefficient_1_slope #chosen so that the map is bijective
        hill_max_slope = self.get_hill_max_slope_func(RF)

        f = lambda n: hill_max_slope(n) - target_hill_slope
        while f(n_range[0]) > 0:
                n_range[0] = 1 + (n_range[0]-1)*1e-1
        while f(n_range[1])<0 and n_range[1] < self.max_allowed_hill_coefficient:
            n_range[1] *= 10
        if n_range[1] < self.max_allowed_hill_coefficient:
            n = bisect(f,n_range[0],n_range[1])
        else:
            n = np.inf
        return HillParameter(RF.sign,RF.L,RF.Delta,RF.theta,n)