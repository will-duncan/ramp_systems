"""
RampModel class and methods

    Author: William Duncan

"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
from ramp_systems.ramp_function import RampFunction
from ramp_systems.cell import Cell

def power_set(iterable):
    pool = tuple(iterable)
    n = len(pool)
    for k in range(n+1):
        for combo in itertools.combinations(pool,k):
            yield combo
    

def get_ramp_system_from_parameter_string(pstring,Network):
    """
    Parses the output of a DSGRN.ParameterSampler instance. 

    :param pstring: Parameter string. Output of DSGRN.ParameterSampler instance.
    :param Network: DSGRN.Network object. 
    :return: RampSystem object. 
    """
    parameter_symbols = {'L','U','T'}
    N = Network.size()
    L = np.zeros([N,N])
    U = np.zeros([N,N])
    theta = np.zeros([N,N])
    gamma = np.ones([N,1])
    k = 0
    while k < len(pstring):
        #locate a parameter representation
        if pstring[k] not in parameter_symbols:
            k += 1
            continue
        cur_symbol = pstring[k]
        #get out node name
        out_name_start = k + 2
        while pstring[k] != '-':
            k += 1
        out_name = pstring[out_name_start:k]
        out_index = Network.index(out_name)
        #get in node name
        in_name_start = k + 2
        while pstring[k] != ']':
            k += 1
        in_name = pstring[in_name_start:k]
        in_index = Network.index(in_name)
        #get the value
        while pstring[k] != ':':
            k += 1
        val_start = k+2
        while pstring[k] != ',' and pstring[k] != '}':
            k += 1
        val = float(pstring[val_start:k])
        if cur_symbol == 'L':
            L[in_index,out_index] = val
        elif cur_symbol == 'U':
            U[in_index,out_index] = val
        else: #cur_symbol == 'T'
            theta[in_index,out_index] = val
    Delta = U - L
    return RampSystem(Network,L,Delta,theta,gamma)
        





class RampSystem:

    def __init__(self,Network,L,Delta,theta,gamma):
        """
        Inputs:
            Network - (_dsgrn.Network)
            L,Delta,theta - (numpy array) Each is an NxN array of ramp function parameters
            gamma - (numpy array) Length N vector of degradation rates 
        """
        self.Network=Network
        N = Network.size()
        self.L = np.array(L,dtype = 'float')
        self.Delta = np.array(Delta,dtype = 'float')
        self.theta = np.array(theta,dtype = 'float')
        self.gamma = np.array(gamma,dtype = 'float').reshape([N,1])
        
        self._zero = np.zeros([Network.size(),Network.size()])
        self._set_func_array()
        self._set_R()
        
   
    def __eq__(self,other):
        if not np.array_equal(self.gamma,other.gamma) or not np.array_equal(self.L,other.L) \
            or not np.array_equal(self.Delta,other.Delta) or not np.array_equal(self.theta,other.theta):
            return False
        for j in range(self.Network.size()):
            try:
                if self.Network.inputs(j) != other.Network.inputs(j):
                    return False
            except AttributeError:
                return False
        return True  

    def __call__(self,x,eps=[]):
        if len(eps) == 0:
            eps = self._zero
        x = np.array(x).reshape([self.Network.size(),1])
        return -self.gamma*x + self.R(x,eps)

    def Lambda(self,kappa,neighbor_index = None,neighbor_direction = None):
        """
        Get value of Lambda on a cell or neighbor of a cell.

        :param r: index of regular direction of cell
        :param cell: Cell object representing a cell in the cell complex. 
        :param neighbor_index: (optional) node index. if passed, neighbor_direction must also
        be specified. Evaluates Lambda on kappa_{neighbor_index}^{neighbor_direction}
        :param neighbor_direction: (optional) one of -1 or 1. if passed, neighbor_index must also 
        be specified. 
        :return: Network.size() x 1 numpy array 
        """
        N = self.Network.size()
        test_point = self.get_cell_test_point(kappa,neighbor_index,neighbor_direction)
        return self.R(test_point)

    def get_cell_test_point(self,kappa,neighbor_index = None,neighbor_direction = None):
        N = self.Network.size()
        test_point = np.zeros([N,1])
        if neighbor_index is not None:
            pi = kappa.pi.copy()
            if neighbor_index in kappa.singular_directions():
                if neighbor_direction == -1:
                    pi[neighbor_index] = (kappa.rho_minus(neighbor_index), kappa(neighbor_index)[0])
                else:
                    pi[neighbor_index] = (kappa(neighbor_index)[0],kappa.rho_plus(neighbor_index))
            else:
                if neighbor_direction == -1:
                    pi[neighbor_index] = (kappa(neighbor_index[0]),)
                else:
                    pi[neighbor_index] = (kappa(neighbor_index[1]),)
            kappa = Cell(kappa.theta,*pi)
        for r in kappa.regular_directions():
            pi_r = kappa(r)
            left = 0 if pi_r[0] == -np.inf else self.theta[pi_r[0],r]
            right = 2*left if pi_r[1] == np.inf else self.theta[pi_r[1],r]
            test_point[r] = (left + right)/2
        for s in kappa.singular_directions():
            test_point[s] = self.theta[kappa(s)[0],s]
        return test_point

    def _set_R(self):
        """
        Creates the 'R' attribute. R is an np array with dimensions [Network.size(),1] defined by 
        x[j]'= -gamma[j]*x[j] + R[j]
        """
        Network = self.Network
        func_array = self.func_array
        def R(x,eps,Network = Network,func_array = func_array):
            R_array = np.zeros([Network.size(),1])
            for i in range(Network.size()):
                cur_prod = 1
                for source_set in Network.logic(i):
                    cur_sum = 0
                    for j in source_set:
                        cur_sum = cur_sum + func_array(x,eps)[i,j]
                    cur_prod =  cur_prod*cur_sum 
                R_array[i] = cur_prod
            return R_array
        self.R = lambda x,eps=self._zero: R(x,eps)


    def _set_func_array(self):
        """
        Creates the func_array attribute.
        """
        Network = self.Network
        def func_array(x, eps,Network = Network,ramp_func_array = self.ramp_function_object_array()):
            x = np.array(x)
            eps = np.array(eps)
            N = Network.size()
            F = np.zeros([N,N])
            for i in range(N):
                for j in Network.inputs(i):
                    Rij = ramp_func_array[i,j]
                    F[i,j] = Rij(x[j],eps[i,j])
            return F
        
        self.func_array = lambda x,eps=self._zero: func_array(x,eps)

    def ramp_function_object_array(self):
        N = self.Network.size()
        array = np.empty([N,N],dtype = RampFunction)
        L = self.L
        Delta = self.Delta
        theta = self.theta
        for i in range(N):
            for j in self.Network.inputs(i):
                sign = 1 if self.Network.interaction(j,i) else -1
                array[i,j] = RampFunction(sign,L[i,j],Delta[i,j],theta[i,j])
        return array

    
    def get_W(self):
        """
        Output:
             W - (list) length Network.size() list of length 
                 Targets(j) sorted lists. W[j] is the output of get_W_j
        """
        W = []

        for j in range(self.Network.size()):
            W_j = self._get_W_j(j)        
            W.append(W_j)    
        return W
                

    def _get_W_j(self,j):
        """
        Input: 
            j - (int) index of a node
        Output: 
            W_j - (list) The set of values of Lambda_j union {0,np.inf}, sorted
        """
        W_j_set = {0,np.inf}
        N = self.Network.size()
        R_j = lambda x: self.R(x)[j,0]
        inputs = self.Network.inputs(j)
        test_point = np.zeros(N)
        x_low = 1/2*self.theta
        x_high = 2*self.theta
        for low_pattern in power_set(inputs):
            low_pattern =  list(low_pattern)
            high_pattern = list(set(inputs) - set(low_pattern))
            test_point[low_pattern] = x_low[j,low_pattern]
            test_point[high_pattern] = x_high[j,high_pattern]
            W_j_set = W_j_set.union({R_j(test_point)})
        W_j = list(W_j_set) 
        W_j.sort()
        return W_j

    def _get_B_jp(self,j,W_j,p):
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

    def _get_B_j(self,j,W_j):
        """
        Input:
            j - (int) index of a node
            W_j - (list) output of get_W_j(j)
        Output:
            B_j - (list) list of lists of indices. For 0<p<len(W_j), B_j[p] is the output of B_jp
                   B_j[0] = []
        """
        B_j = [[]]
        for p in range(1,len(W_j)):
            B_jp = self._get_B_jp(j,W_j,p)
            B_j.append(B_jp)
        return B_j
            
    def _get_B(self,W):
        B = []
        for j in range(self.Network.size()):
            B.append(self._get_B_j(j,W[j]))
        return B

    def is_weakly_equivalent(self,eps):
        """
        Return True if the ramp paramaeter (Z,eps) is weakly equivalent to (Z,0)
        and False otherwise. Assumes self.is_regular().

        Input: 
            eps - (numpy array) choice of perturbation parameter
        """
        N = self.Network.size()
        for j in range(N):
            theta_out = self.theta[:,j]
            out_index = {theta_out[i]:i for i in range(N)}
            theta_out = theta_out[theta_out != 0]
            theta_out.sort()
            for i in range(len(theta_out)-1):
                if theta_out[i] + eps[out_index[theta_out[i]],j] >= theta_out[i+1] - eps[out_index[theta_out[i+1]],j]:
                    return False
        return True


    def is_strongly_equivalent(self,eps):
        """
        Returns True if the ramp parameter (Z,eps) is strongly equivalent to (Z,0)
        and False otherwise. 

        Input:
            eps - (numpy array) choice of perturbation parameter
        """
        if not self.is_weakly_equivalent(eps):
            return False
        W = self.get_W()
        B = self._get_B(W)
        N = self.Network.size()
        for j in range(N):
            B_j = B[j]
            W_j = W[j]
            for p in range(len(B_j)):
                B_jp = B_j[p]
                if len(B_jp) == 0:
                    continue
                if p != 0 and self.theta[B_jp[0],j] - eps[B_jp[0],j] <= W_j[p-1]:
                    return False
                if p != len(B_j) - 1 and self.theta[B_jp[-1],j] - eps[B_jp[-1],j] >= W_j[p]:
                    return False
        return True


    def _get_eps_jp(self,j,W_j,B_j,p):
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
            lambda q: (theta[i(q),j] - theta[i(q-1),j])/(Delta[i(q),j] + Delta[i(q-1),j])
        D_tilde = min([theta_distance(q) for q in range(1,len(Theta_j))],default=np.inf)
        ##construct D
        k = len(B_jp)-1
        D_list = [D_tilde]
        if p != 1:  #W_j[p-1] != 0
            W_pm1_dist = (gamma[j]*theta[i(0),j] - W_j[p-1])/(gamma[j]*Delta[i(0),j])
            D_list.append(W_pm1_dist)
        if p != (len(W_j)-1):  #W_j[p] != inf
            W_p_dist = (W_j[p] - gamma[j]*theta[i(k),j])/(gamma[j]*Delta[i(k),j])
            D_list.append(W_p_dist)
        D = min(D_list)
        ##construct eps_jp
        for i in B_jp:
            eps_jp[i,j] = Delta[i,j]*D
        return eps_jp

    def _get_eps_j(self,j,W_j,B_j):
        N = self.Network.size()
        eps_j = np.zeros([N,N])
        for p in range(1,len(W_j)):
            eps_j += self._get_eps_jp(j,W_j,B_j,p)
        return eps_j

    def optimal_eps(self):
        """
        Output:
            eps - (numpy array) A choice of eps which minimizes the maximum slope
                  for each j among eps which satisfiy eps'<eps implies (Z,eps')
                  is strongly equivalent to (Z,0). This guarantees all equilibria 
                  of DSGRN(Z) have corresponding equilibria in R(Z,eps'). 
        """
        Network = self.Network
        W = self.get_W()
        B = self._get_B(W)
        N = Network.size()
        eps = np.zeros([N,N])
        for j in range(N):
            eps += self._get_eps_j(j,W[j],B[j])
        return eps
            
    def is_regular(self,eps=None):
        """
        Returns True if the switching parameter is regular and False otherwise.
        This is a strong form of regularity in that it checks if 
        gamma(theta +/- eps) = Lambda_j for all values of Lambda_j, not just 
        Lambda_j(kappa)
        """
        if eps is None:
            eps = self._zero
        N = self.Network.size()
        L = self.L
        Delta = self.Delta
        theta = self.theta
        gamma = self.gamma
        if sum(gamma > 0) < N:
            return False
        W = self.get_W()
        for j in range(N):
            for i in self.Network.outputs(j):
                if L[i,j]<=0 or Delta[i,j]<=0 or theta[i,j]<=0:
                    return False
                if sum(theta[i,:] == theta[i,j]) > 1:
                    return False
                W_j = W[j][1:]
                if gamma[j]*(theta[i,j] + eps[i,j]) in W_j or \
                    gamma[j]*(theta[i,j] - eps[i,j]) in W_j:
                    return False
        return True

    
    def theta_order(self,j):
        """
        Get a list representing the orders of the thresholds. 

        :param j: index of a node
        :return: list of integers of the form [i_1,i_2,...,i_k] such that
        theta[i_1,j]<theta[i_2,j]<...theta[i_k,j] are consecutive thresholds
        """
        theta_j = self.theta[:,j].copy()
        theta_j[theta_j == 0] = np.inf
        index_list = []
        while(sum(theta_j == np.inf) < self.Network.size()):
            i = np.argmin(theta_j)
            index_list.append(i)
            theta_j[i] = np.inf
        return index_list

    def all_theta_orders(self):
        orders = []
        for j in self.Network.size():
            orders.append(self.theta_order(j))
        return orders


    def is_opaque(self,kappa):
        """
        Returns true if the regular directions of the cell are invariant under the flow of RS

        :param kappa: Cell object
        """
        Lambda = self.Lambda(kappa)
        gamma = self.gamma
        theta = self.theta
        for r in kappa.regular_directions():
            pi_r = kappa(r)
            left = 0 if pi_r[0] == -np.inf else gamma[r,0]*theta[pi_r[0],r]
            right = 2*left if pi_r[1] == np.inf else self.gamma[r,0]*self.theta[pi_r[1],r]
            if Lambda[r] < left or Lambda[r] > right:
                return False
        return True

    def equilibria_regular_directions(self,kappa):
        """
        Gives the entries of the equilibrium values for the regular directions of a cell. 

        :param cell: Cell object which is assumed to be opaque: RS.is_opaque(kappa). 
        :return: N x 1 array with equilibrium of x_r' on kappa for regular directions r. 
        Entries corresponding to singular directions s are 0
        """
        Lambda = self.Lambda(kappa)
        eq = np.zeros([self.Network.size(),1])
        for r in kappa.regular_directions():
            gamma_r = self.gamma[r,0]
            eq[r,0] = Lambda[r]/gamma_r
        return eq

    def __repr__(self):
        return 'RampSystem(Network = {},L = {},Delta = {},theta = {},gamma = {})'.format(self.Network.specification(),self.L,self.Delta,self.theta,self.gamma)
