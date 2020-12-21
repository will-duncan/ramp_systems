"""
CyclicFeedbackSystem class and methods. This is a child class of the RampSystem
class (see ramp_system.py).

    Author: William Duncan
"""
from ramp_systems.ramp_system import RampSystem
import itertools
import sympy
from sympy.matrices import zeros as sympy_zeros
import numpy as np

    



DEFAULT_TOLERANCE = 1e-6

class CyclicFeedbackSystem(RampSystem):

    def __init__(self,Network,L,Delta,theta,gamma,tol = DEFAULT_TOLERANCE):
        """
        Requires that the network is a cyclic feedback network.
        Input:
            tol - tolerance used by j_border_crossings()
        """
        RampSystem.__init__(self,Network,L,Delta,theta,gamma)
        self._set_attributes()
        self.tol = tol

    def _set_attributes(self):
        """
        Sets the attributes 'cfs_sign', 'rho', 'rho_inv','edge_sign' 
        Raises an exception if the network is not a cyclic feedback network. 
            cfs_sign - sign of the loop
            rho - (list) target map. j->rho[j] is the unique edge from j 
            rho_inv - (list) source map. rho_inv[j]->j is the unique edge into j
            edge_sign - (list) edge_sign[j] is the sign of the edge rho_inv[j]->j
        """
        Network = self.Network
        node = 0
        next_node = -1
        cfs_sign = 1
        N = Network.size()
        rho = [-1]*N
        rho_inv = [-1]*N
        edge_sign = [1]*N
        while(next_node != 0):
            output = Network.outputs(node)
            rho[node] = output[0]
            if len(output) != 1:
                raise ValueError("CyclicFeedbackSystem requires Network is a cyclic\
                     feedback network but at least one node had number of outputs\
                         different from 1.")
            next_node = output[0]
            rho_inv[next_node] = node
            if not Network.interaction(node,next_node): #node represses next_node
                cfs_sign *= -1
                edge_sign[next_node] = -1
            node = next_node
        if -1 in rho:
            raise ValueError("CyclicFeedbackSystem requires Network is a cyclic\
                feedback network but the nodes don't form a length Network.size() cycle.")
        self.cfs_sign = cfs_sign
        self.rho = rho
        self.rho_inv = rho_inv
        self.edge_sign = edge_sign

    def _get_slope_product(self,eps_func):
        """
        Helper function for pos_loop_bifurcations and neg_loop_bifurcations.

        input:
            eps_func - sympy expression
        output: 
            function which takes a number s and returns the slope product M(eps_func(s)) 
        """
        Delta = sympy.Matrix(self.Delta)
        rho_inv = self.rho_inv
        slope_product_func = 1
        for j in range(self.Network.size()):
            slope_product_func *= Delta[j,rho_inv[j]]/(2*eps_func[j,rho_inv[j]]) 
        s = sympy.symbols('s')
        return sympy.utilities.lambdify(s,slope_product_func,'numpy')


    def get_bifurcations(self,eps_func = None):
        if self.cfs_sign == 1:
            bifurcations, eps_func = self.pos_loop_bifurcations(eps_func)
        else:
            bifurcations, eps_func = self.neg_loop_bifurcations(eps_func)
        return bifurcations, eps_func

    def neg_loop_bifurcations(self,eps_func = None):
        """
        Finds all bifurcations assuming cfs_sign == -1. 
        Input: 
            eps_func - (optional) sympy expression giving parameterization of eps
                        assumes eps_func is of the form A*s where A is a matrix
        Output:
            s_vals - (list[list[list] ] of length N+1) 
                     s_vals[j] is a list with entries [s,x] so that there is a 
                     stability changing border crossing bifurcation at eps_func(s) with 
                     x[j] = theta[rho[j],j]+/-eps[rho[j],j] if j<N. s_vals[N] are 
                     the values of s so that there is a Hopf bifurcation
            eps_func - same as input. Returned here in case eps_func is not specified
                       in the function call. 
        TO DO: 
            implement for gammas not equal
        """
        if self.cfs_sign != -1:
            raise ValueError('neg_loop_bifurcations but the loop is positive')
        N = self.Network.size()
        s_vals = [[] for i in range(N+1)]
        if N <= 2:
            return s_vals
        s = sympy.symbols('s')
        crossings,eps_func = self.border_crossings(eps_func)
        slope_product = self._get_slope_product(eps_func)
        if self.gamma.min() == self.gamma.max():
            gamma = self.gamma[0,0]
            secant_condition_val = gamma*np.cos(np.pi/N)**(-N)
            #border crossing bifurcations
            for j in range(N):
                cur_vals = set([crossing for crossing in crossings[j] \
                    if slope_product(crossing[0])>=secant_condition_val])
                s_vals[j].extend(cur_vals)
            #hopf bifurcation in singular domain
            s_hopf = (slope_product(1)/secant_condition_val)**(1/N) #M(eps(s_hopf)) = gamma*sec(pi/N)**N. 
            x_hopf = self.singular_equilibrium(eps_func)(s_hopf)
            eps_hopf = eps_func.subs(s,s_hopf)
            if self.in_singular_domain(x_hopf,eps_hopf):
                s_vals[N].append( [s_hopf,x_hopf] )         
            return s_vals, eps_func
        else:
            raise ValueError('Inequal gammas not yet implemented for neg_loop_bifurcations.')

                    


    def pos_loop_bifurcations(self,eps_func = None):
        """
        Finds all bifurcations assuming cfs_sign == 1.
        
        Inputs:
            eps_func - (optional) sympy expression describing the parameterization of eps
        Outputs:
            s_vals - (list[list[list] ] length N+1) s_vals[j] are the values of (s,x) so that there is a 
                     bifurcation at eps_func(s) where x[j] = theta[rho[j],j] +/- eps[rho[j],j]
                     s_vals[N] = set(). len(s_vals) = N+1 for consistency with neg_loop_bifurcations()
            eps_func - same as input. Returned in case eps_func is not specified 
                       during function call. 
        """
        if self.cfs_sign != 1:
            raise ValueError('pos_loop_bifurcations called but the loop is negative.')
        crossings, eps_func = self.border_crossings(eps_func)
        slope_product = self._get_slope_product(eps_func)
       
        gamma_product = self.gamma.prod()
        N = self.Network.size()
        s_vals = [[] for i in range(N+1)]
        for j in range(N):
            cur_vals = [crossing for crossing in crossings[j] if slope_product(crossing[0])>= gamma_product]
            s_vals[j].extend(cur_vals)
        return s_vals, eps_func

        


    def border_crossings(self,eps_func=None):
        """
        Finds all values of eps so that the system has a border crossing on the 
        boundary of loop characteristic cell tau(eps)

        Input:
            eps_func - (optional) sympy expression describing the parameterization of eps
                       default is chosen so that all slopes are the same. 
            tol - desired tolerance to pass to the root isolation function
        Output:
            crossings - (list[list]) crossings[j] is the output of
                        j_border_crossings
            eps_func - same as input. Returned here in case eps_func is not specified
                       in function call. 
        """
        eps_func,s = self._handle_eps_func(eps_func)
        x_eq = self.singular_equilibrium(eps_func,lambdify = False)
        N = self.Network.size()
        crossings = [[] for i in range(N)]
        for j in range(N):
            crossings[j] = self.j_border_crossings(j,x_eq,eps_func)
        return crossings, eps_func



    def j_border_crossings(self,j,x_eq,eps_func): 
        """
        Finds all values of eps so that the system has a border crossing
        of type x[j] = theta[rho[j],j] +/- eps[rho[j],j]. 

        Input:
            j - node of the network. 
            x_eq - sympy expression given by singular_equilibrium(eps_func,lambdify=false) 
            eps_func - sympy expression describing the parameterization of eps
            tol - tolerance on width of saddle node containg intervals
            
        Output:
            crossings - (list[list]) inner lists are of the form [s,x]
                        Border crossing occurs approximately at eps = eps_func(s) 
                        where s is within tol of the true value and the value of 
                        the equilibrium at the crossing is x 
        """
        tol = self.tol
        s = sympy.symbols("s")
        rho = self.rho
        xj = x_eq[j]
        xj = xj.cancel()
        num_xj, denom_xj = sympy.fraction(xj)
        candidates = []
        for beta in [-1,1]:
            crossing_poly = num_xj - denom_xj*(self.theta[rho[j],j] + beta*eps_func[rho[j],j])
            crossing_poly = sympy.Poly(crossing_poly,domain = 'QQ')
            candidates.extend(crossing_poly.intervals(eps=tol,inf = 0))
        crossings = []
        while len(candidates) != 0:
            #output of poly.intervals is a tuple of the form ((a,b),number)
            #(a,b) is the interval. I could not find information on what 
            #number is in the sympy documentation, although I assume it is the 
            #number of roots in the interval and should always be 1 unless there 
            # is a double root. 
            root_int, num_roots = candidates.pop()
            a,b = root_int
            if a == b and a == 0:
                continue
            a = float(a)
            b = float(b)
            #check that for i != j, x[i] is in the singular domain
            a_in = self.in_singular_domain(x_eq.subs(s,a),eps_func.subs(s,a),j)
            b_in = self.in_singular_domain(x_eq.subs(s,b),eps_func.subs(s,b),j)
            if a_in or b_in: 
                crossings.append( [(a+b)/2, np.array(x_eq.subs(s,(a+b)/2)).astype(np.float64)] )
        return crossings  
            


    def in_singular_domain(self,x,eps,j=None):
        """
        Returns true if for each i != j, x[i] is in the closure of the projection 
        of the loop characteristic cell, pi_i(tau(eps)), and false otherwise.

        Input:
            x - (Nx1 numpy array) point in phase space
            eps - (NxN numpy array) value of perturbation parameter
        Output:
            bool
        """
        N = self.Network.size()
        rho = self.rho
        x = np.array(x).reshape([N,1])
        
        theta_vec = np.zeros([N,1])
        eps_vec = np.zeros([N,1])
        for i in range(N):
            theta_vec[i] = self.theta[rho[i],i]
            eps_vec[i] = eps[rho[i],i]
            self._theta_vec = theta_vec
            self._eps_vec = eps_vec
        in_domain = np.logical_and(x >= theta_vec - eps_vec,\
                                   x <= theta_vec + eps_vec)
        if j is not None:
            in_domain[j] = True
        if sum(in_domain[:,0]) == N:
            return True
        return False




    def _handle_eps_func(self,eps_func):
        """
        Function for dealing with eps_func argument.

        Input:
            eps_func - (sympy expression or None) 
        Output:
            eps_func - (sympy expression)
            s - (sympy symbol)
        """
        s = sympy.symbols("s")
        if eps_func is None:
            eps_func = sympy.Matrix(self.Delta)*s
        else:
            for sym in eps_func.free_symbols:
                eps_func = eps_func.subs(sym,s)
        return eps_func, s
    


    def singular_equilibrium(self,eps_func=None,lambdify = True):
        """
        Compute the equilibrium that would exist if all ramp functions were
        operating in their linear regime over all of phase space. 
        Input:
            eps_func - (sympy expression) function giving a parameterization of eps
                        Assumes the function of the form A*s where A is a matrix
        Output:
            x - (function) returns value of equilibrium at eps_func(s)         
        """
        eps_func, s = self._handle_eps_func(eps_func)

        N = self.Network.size()
        rho_inv = self.rho_inv
        L_vec = sympy_zeros(N,1)
        Delta_vec = sympy_zeros(N,1)
        theta_vec = sympy_zeros(N,1)
        signed_slope = sympy_zeros(N,N)
        signed_slope_vec = sympy_zeros(N,1)
        
        for j in range(N):
            L_vec[j] = self.L[j,self.rho_inv[j]]
            Delta_vec[j] = self.Delta[j,self.rho_inv[j]]
            theta_vec[j] = self.theta[j,self.rho_inv[j]]
            signed_slope[j,rho_inv[j]] = self.edge_sign[j]*Delta_vec[j]/(2*eps_func[j,rho_inv[j]])
            signed_slope_vec[j] = signed_slope[j,rho_inv[j]]
        Gamma = sympy.Matrix(np.diagflat(self.gamma))
        A = Gamma - signed_slope
        b = L_vec + 1/2*Delta_vec - signed_slope_vec.multiply_elementwise(theta_vec)
        x = A.LUsolve(b)
        if lambdify:
            return sympy.utilities.lambdify(s,x,"numpy")
        else:
            return x


        