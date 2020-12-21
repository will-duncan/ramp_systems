"""
saddle_node_maps.py
Implements the N to 1 map from ramp function saddles to hill function saddles.
    Author: William Duncan

"""


from scipy.optimize import bisect
import numpy as np
from ramp_systems.cyclic_feedback_system import DEFAULT_TOLERANCE
import sympy

class HillParameter:

    def __init__(self,sign,L,Delta,theta,n):
        """
        Input:
            sign - either 1 or -1
            L,Delta,theta,n - parameters for a hill function
        """
        self.sign = sign
        self.L = L
        self.Delta = Delta
        self.theta = theta
        self.n = n
    
    def __repr__(self):
        sign = self.sign
        L = self.L
        Delta = self.Delta
        theta = self.theta
        n = self.n
        return 'HillParameter({},{},{},{},{})'.format(sign,L,Delta,theta,n)


class HillSystemParameter:

    def __init__(self,Network,sign,L,Delta,theta,n,gamma):
        """
        Input:
            sign, gamma - length N lists
            L,Delta,theta,n - NxN arrays
        """
        self.Network = Network
        N = Network.size()
        self.sign = np.array(sign)
        self.L = np.array(L)
        self.Delta = np.array(Delta)
        self.theta = np.array(theta)
        self.n = np.array(n)
        self.gamma = np.array(gamma).reshape([N,1])
    

def hill_value(x,hill_parameter):
    sign = hill_parameter.sign
    theta = hill_parameter.theta
    Delta = hill_parameter.Delta
    L = hill_parameter.L
    n = hill_parameter.n
    if sign == 1:
        return L + Delta/((theta/x)**n + 1)
    if sign == -1:
        return L + Delta/((x/theta)**n + 1)

def hill_second_derivative_root(*args):
    if len(args) == 1:
        hill_parameter = args[0]
        theta = hill_parameter.theta
        n = hill_parameter.n
    elif len(args) == 2:
        theta = args[0]
        n = args[1]
    else:
        raise TypeError('hill_second_derivative_root() takes 1 or 2 position arguments\
            but {} were given.'.format(len(args)))
    return theta*((n-1)/(n+1))**(1/n)

def hill_derivative_magnitude(x,*args):
    if len(args) == 1:
        hill_parameter = args[0]
        sign = hill_parameter.sign
        Delta = hill_parameter.Delta
        theta = hill_parameter.theta
        n = hill_parameter.n
    elif len(args) == 4:
        sign = args[0]
        Delta = args[1]
        theta = args[2]
        n = args[3]
    else: 
        raise TypeError('hill_derivative() takes 1 or 4 positiional arguments\
            but {} were given.'.format(len(args)))
    return Delta*n/(theta*(theta/x)**(n-1) + 2*x + x*(x/theta)**n)


def default_monotone_function(x):
    return x

class RampToHillSaddleMap:

    def __init__(self, Network, monotone_function = default_monotone_function, \
        max_allowed_hill_coefficient = 1e8):
        """
        Input:
            Network - DSGRN network object
            monotone_function - any strictly increasing function on the closed interval
                                [-1,1]
            max_allowed_hill_coefficient - upper bound on returned hill coefficients
        """
        self.max_allowed_hill_coefficient = max_allowed_hill_coefficient
        self.Network = Network
        self.monotone_function = monotone_function



    def cyclic_feedback_system_map(self,CFS,bifurcation_pt,eps_func,border_crossing_index):
        """
        The map for a cyclic feedback system. Requires CFS has a bifurcation at 
        eps. These can be found via CFS.get_bifurcations().

        Input:
            CFS - CyclicFeedbackSystem instance.
            bifurcation_pt - (list) has the form (s,x). Values so that CFS has a
                             bifurcation at eps_func(s) with equilibrium x
            eps_func - (sympy expression) parameterization of eps
            border_crossing_index - index i so that the singular equilibrium x 
                                    satisfies x[i] = theta[rho[i],i] +/- eps[rho[i],i]
                                    if i = N, then it is not a border crossing bifurcation
        Output:
            hill_system - HillSystemParameter instance
        """
        if CFS.cfs_sign != 1:
            raise ValueError('cyclic_feedback_system_map currently only implemented \
                for positive cyclic feedback systems, but received a negative cyclic \
                feedback system')
        i = border_crossing_index
        s = bifurcation_pt[0]
        s_sym = sympy.symbols('s')
        eps = eps_func.subs(s_sym,s)
        x_ramp = bifurcation_pt[1]
        function_map = RampToHillFunctionMap(self.max_allowed_hill_coefficient)
        rho_inv = CFS.rho_inv
        rho = CFS.rho
        N = CFS.Network.size()
        ramp_array = CFS.ramp_function_object_array()
        theta = CFS.theta
        gamma_product = CFS.gamma.prod()

        hill_list = [function_map(ramp_array[rho[j],j],eps[rho[j],j]) for j in range(N)]
        x_star = [hill_second_derivative_root(hill_list[j]) for j in range(N)]
        max_slope_list = [function_map.get_hill_max_slope_func(ramp_array[rho[j],j])(hill_list[j].n) for j in range(N)]
        max_slope_list = np.array(max_slope_list)

        #compute x_hill, the x value of the saddle node for the hill system
        N = self.Network.size()
        x_hill = np.zeros([N,1])
        g = self.monotone_function
        for j in range(N):
            if j == i:
                continue
            g_val = g((x_ramp[j,0] - theta[rho[j],j])/eps[rho[j],j])
            if x_ramp[j,0] < theta[rho[j],j]:
                x_jk = self._get_x_jk(j,1,max_slope_list,x_star,gamma_product,hill_list[j])
                x_hill[j,0] = x_star[j] + (x_star[j]-x_jk)*g_val
            else:
                x_jk = self._get_x_jk(j,2,max_slope_list,x_star,gamma_product,hill_list[j])
                x_hill[j,0] = x_star[j] + (x_jk - x_star[j])*g_val
        
        slope_product = np.array([hill_derivative_magnitude(x_hill[j],hill_list[j]) for j in range(N) if j != i]).prod()
        f = lambda x: hill_derivative_magnitude(x,hill_list[i])*slope_product - gamma_product
        if x_ramp[i] < theta[rho[i],i]:
            lower_bound = 1e-2
            while f(lower_bound) > 0:
                lower_bound *= 1e-2
            interval = [lower_bound,x_star[i]]
        else:
            upper_bound = 10*x_star[i]
            while f(upper_bound) > 0:
                upper_bound *= 100
            interval = [x_star[i],100*x_star[i]] 
        x_hill[i,0] = bisect(f,interval[0],interval[1])

        L_hill = np.zeros([N,N])
        sign_hill = np.zeros([N,N])
        n_system = np.zeros([N,N])
        for j in range(N):
            L_hill[j,rho_inv[j]] = CFS.L[j,rho_inv[j]] + CFS.gamma[j]*x_hill[j] - hill_value(x_hill[rho_inv[j]],hill_list[rho_inv[j]])
            sign_hill[j,rho_inv[j]] = CFS.edge_sign[j]
            n_system[rho[j],j] = hill_list[j].n
            
        hill_sys = HillSystemParameter(CFS.Network,sign_hill,L_hill,CFS.Delta,theta,n_system,CFS.gamma)
        return hill_sys, x_hill
            



    def _get_x_jk(self,j,k,max_slope_list,x_star,gamma_product,hill_parameter):
        """
        Get x_j^k as defined in the write up. 

        Inputs:
            max_slope_list - (numpy array)
            gamma_product - product of the gamma[j]
            j - node index
            k - either 1 or 2. If k == 1 then x_j^k < x_j^star, if k == 2 then 
                x_j^k > x_j^star
        Outputs:
            x_jk
        """
        max_slope_product = max_slope_list.prod()
        product_not_j = max_slope_product/max_slope_list[j]
        f = lambda x: hill_derivative_magnitude(x,hill_parameter)*product_not_j - gamma_product
        assert(f(x_star[j]) > 0)
        if k == 1:
            lower_bound = 1e-2
            while f(lower_bound) > 0:
                lower_bound *= 1e-2
            interval = [lower_bound,x_star[j]]
        elif k == 2:
            upper_bound = x_star[j]*10
            while f(upper_bound) > 0:
                upper_bound *= 10
            interval = [x_star[j],upper_bound]
        else:
            raise ValueError('k must be 1 or 2 but received k = {}'.format(k))

        
        x_jk = bisect(f,interval[0],interval[1])
        return x_jk

   



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
        

        

