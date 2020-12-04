"""
CyclicFeedbackSystem class and methods. This is a child class of the RampSystem
class (see ramp_system.py).

    Author: William Duncan
"""
from ramp_systems.ramp_system import *
import itertools
import sympy
from sympy.matrices import zeros as sympy_zeros
#from sympy.matrices.dense import matrix_multiply_elementwise

    

## Considering completing this code if the sympy symbolic solver is slow
# def solve_one_cycle_lin_sys(A,b):
#     """
#     Solves Ax = b where the non-zero entries of A are the diagonal and entries
#     of the form A[j+1,j] where rho defines a cycle. Uses Cramer's rule, which
#     is O(N^2) in this case. 
#     """
#     N = A.shape[1]
#     #compute det(A) using Liebniz formula
#     diag_A = A.diagonal()
#     cycle_A = A.diagonal(-1)
#     diag_prod_A = diag_A.prod()
#     det_A = diag_prod_A + (-1)**(N-1)*cycle_A.prod() 
#     x = np.zeros([N])
    
#     #compute A_j using Liebniz formula for each j
#     for i,j in itertools.product(range(N),2):
#         if i <= j:
#             parity = j-i
#         else:
#             parity = N-(i-j)


        






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

    def pos_loop_saddle_eps(self,j,eps_func=None):
        """
        Finds all values of eps so that the system has a saddle node of type 
        x[j] = theta[rho[j],j] +/- eps[rho[j],j]. Requires a positive cyclic feedback system.
        Input:
            j - node of the network. 
            eps_func - (function, optional) function which maps a positive 
                        scalar to a numpy array of eps values. Requires 
                        eps_func(s) >= 0 and eps_func(0) = np.zeros([N,N]). 
        """
        

        
        
        #solve Ax = b where A = gamma + edge_sign_matrix*

    
    def singular_equilibrium(self,eps_func=[]):
        """
        Compute the equilibrium that would exist if all ramp functions were
        operating in their linear regime over all of phase space. 
        Input:
            eps_func - (sympy expression) function giving a parameterization of eps
                        Assumes the function is scalar
        Output:
            x - (function) returns value of equilibrium at eps_func(s)         
        """
        s = sympy.symbols("s")
        if len(eps_func) == 0:
            eps_func = sympy.Matrix(self.Delta)*s
        else:
            for sym in eps_func.free_symbols:
                eps_func = eps_func.subs(sym,s)

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
        Gamma = sympy.Matrix(np.diag(self.gamma))
        print(L_vec,Delta_vec,signed_slope,Gamma)
        A = Gamma - signed_slope
        b = L_vec + 1/2*Delta_vec - signed_slope_vec.multiply_elementwise(theta_vec)
        x = A.QRsolve(b)
        return sympy.utilities.lambdify(s,x,"numpy")


        