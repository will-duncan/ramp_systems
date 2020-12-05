"""
CyclicFeedbackSystem class and methods. This is a child class of the RampSystem
class (see ramp_system.py).

    Author: William Duncan
"""
from ramp_systems.ramp_system import *
import itertools
import sympy
from sympy.matrices import zeros as sympy_zeros


    



        






class CyclicFeedbackSystem(RampSystem):

    def __init__(self,Network,L,Delta,theta,gamma):
        """Requires that the network is a cyclic feedback network."""
        RampSystem.__init__(self,Network,L,Delta,theta,gamma)
        self._set_attributes()

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
        self.cfs_sign = cfs_sign
        self.rho = rho
        self.rho_inv = rho_inv
        self.edge_sign = edge_sign

    def pos_loop_saddles():
        pass

    def border_crossings(self,j,eps_func=None,tol=1e-4): #,min_tol=1e-10):
        """
        Finds all values of eps so that the system has a border crossing
        of type x[j] = theta[rho[j],j] +/- eps[rho[j],j]. 
        Input:
            j - node of the network. 
            eps_func - (function, optional) function giving a parameterization
                        of eps.  Assumes the function is univariate and requires 
                        eps_func(s) > 0 for s>0 and eps_func(0) = sympy_zeros(N,N). 
            tol - tolerance on width of saddle node containg intervals
            
        Output:
            crossings - list of tuples of the form (s,non-degenerate)
                s - Border crossing bifurcation occurs approximately 
                    at eps_func(s) where s is within tol of the true value
                non-degenerate - (bool) False if the saddle node possibly occurs
                                 on a codimension < N-1 boundary
        """
        #min_tol - smallest allowed value of refined_tol
        eps_func, s = self._handle_eps_func(eps_func)    
        x_eq = self.singular_equilibrium(eps_func,lambdify=False)
        for sym in x_eq.free_symbols:
            x_eq = x_eq.subs(sym,s)
        rho = self.rho
        #refinement_factor = 1e-2
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
            #number is in the sympy documentation, although typically number == 1
            root_int, unknown = candidates.pop()
            assert(unknown == 1)
            a,b = root_int
            if a == b and a == 0:
                continue
            a = float(a)
            b = float(b)
            #check that for i != j, x[i] is in the singular domain
            a_in = self.in_singular_domain(x_eq.subs(s,a),eps_func.subs(s,a),j)
            b_in = self.in_singular_domain(x_eq.subs(s,b),eps_func.subs(s,b),j)
            if a_in and b_in:
                    crossings.append(((a+b)/2,True))
            elif a_in ^ b_in:
                # refined_tol = (b-a)*refinement_factor
                # if refined_tol < min_tol:
                crossings.append(((a+b)/2,False))
                # else:
                    #candidates.append(crossing_poly.intervals(eps=refined_tol,inf=a,sup = b))
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
        try:
            theta_vec = self._theta_vec
            eps_vec = self._eps_vec
        except AttributeError:
            theta_vec = np.zeros([N,1])
            eps_vec = np.zeros([N,1])
            for j in range(N):
                theta_vec[j] = self.theta[rho[j],j]
                eps_vec[j] = eps[rho[j],j]
                self._theta_vec = theta_vec
                self._eps_vec = eps_vec
        in_domain = np.logical_and(x >= self.gamma*(theta_vec - eps_vec),\
                                   x <= self.gamma*(theta_vec + eps_vec))
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
                        Assumes the function is univariate and requires 
                        eps_func(s) >= 0 and eps_func(0) = np.zeros([N,N]). 
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


        