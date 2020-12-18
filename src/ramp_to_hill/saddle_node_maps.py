from scipy.optimize import bisect
import numpy as np

class HillParameter:

    def __init__(self,sign,L,Delta,theta,n):
        self.sign = sign
        self.L = L
        self.Delta = Delta
        self.theta = theta
        self.n = n


# class RampSaddleToHillSaddleMap:



class RampToHillFunctionMap:
    """
    Calling an instance of this class returns a HillParameter which corresponds 
    to the given RampFunction. 
    """
    def __init__(self,allowed_hill_coefficient_interval = [1,1e8]):
        self.allowed_hill_coefficient_interval = allowed_hill_coefficient_interval

    def hill_second_derivative_root(self,theta,n):
        return (theta*((n-1)/(n+1)))**(1/n)

    def hill_derivative(self,x,sign,Delta,theta,n):
        return sign*Delta*n/(theta*(theta/x)**(n-1) + 2*x + x*(x/theta)**n)

    def get_hill_max_slope_func(self,RF,eps):        
        return lambda n: RF.sign*self.hill_derivative(\
            self.hill_second_derivative_root(RF.theta,n),RF.sign,RF.Delta,RF.theta,n)


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
        n_range = self.allowed_hill_coefficient_interval
        hill_coefficient_1_slope = RF.Delta/RF.theta
        ramp_function_slope = RF.sign*RF.dx(RF.theta,eps)
        target_hill_slope = ramp_function_slope + hill_coefficient_1_slope #chosen so that the map is bijective
        hill_max_slope = self.get_hill_max_slope_func(RF,eps)
        if hill_max_slope(n_range[1]) > target_hill_slope:
            f = lambda n: hill_max_slope(n) - target_hill_slope
            lower_bound_offset = 1e-2
            while f(n_range[0]+lower_bound_offset) > 0:
                lower_bound_offset *=1e-2
            n = bisect(f,n_range[0]+lower_bound_offset,n_range[1])
        else:
            n = np.inf
        return HillParameter(RF.sign,RF.L,RF.Delta,RF.theta,n)
        

        

