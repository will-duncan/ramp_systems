"""
RampSystem class and methods. 

    Author: William Duncan

"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
from ramp_systems.ramp_function import RampFunction
from ramp_systems.cell import Cell
from ramp_to_hill.ramp_to_hill_function_map import RampToHillFunctionMap
from dsgrn_utilities.graphtranslation import netspec2nxgraph
import networkx as nx


def power_set(iterable):
    pool = tuple(iterable)
    n = len(pool)
    for k in range(n+1):
        for combo in itertools.combinations(pool,k):
            yield combo

def get_FPs_from_MG(MG):
    MG_string = MG.stringify()
    last_FP_index = 0
    FPs = []
    while True:
        cur_FP_index = MG_string.find('FP',last_FP_index)
        if cur_FP_index == -1:
            #no remaining FPs so break out of loop
            break
        cur_FP_coord_string = MG_string[cur_FP_index + 5 : MG_string.find('}',cur_FP_index+5)-1]
        last_FP_index = cur_FP_index + 5 + len(cur_FP_coord_string)
        FPs.append([int(c) for c in cur_FP_coord_string.split(',')])
    return FPs
    

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
        

def get_all_LCCs(RS):
    Network = RS.Network
    nxgraph = netspec2nxgraph(Network.specification())
    cycles = list(nx.simple_cycles(nxgraph))
    permutations = get_permutations_from_cycles(cycles)
    LCCs = []
    for perm in permutations:
        cell_projections = [[] for j in range(Network.size())]
        singular_directions = set()
        for cycle in perm:
            for j in range(len(cycle)):
                if j < len(cycle) - 1:
                    jplus1 = j+1
                else:
                    jplus1 = 0
                cell_projections[cycle[j]] = [cycle[jplus1]]
                singular_directions.add(cycle[j])
        for j in range(Network.size()):
            if j in singular_directions:
                continue
            cell_projections[j] = get_all_regular_projections(RS.theta,j)
        for projection in itertools.product(*cell_projections):
            LCCs.append(Cell(RS.theta,*projection))
    return LCCs

def get_all_regular_projections(theta,j):
    Theta_j = theta[:,j].copy()
    Theta_j[Theta_j == 0] = np.inf
    out_index_order = []
    while sum(Theta_j == np.inf) < len(Theta_j):
        out_index_order.append(np.argmin(Theta_j))
        Theta_j[out_index_order[-1]] = np.inf
    projections = [(-np.inf,out_index_order[0]),(out_index_order[-1],np.inf)]
    for i in range(len(out_index_order)-1):
        projections.append((out_index_order[i],out_index_order[i+1]))
    return projections


def get_permutations_from_cycles(cycles):
    """
    Return permutations of singular directions generated by cycles. 

    :param cycles: list of cycles. Each cycle is represented by a list. The cycle 
    is defined by cycle[0]->cycle[1]->...->cycle[-1]->cycle[0]
    :return: list of permutations. Each permutation is represented by a list of non-overlapping
    cycles. 
    """
    permutations = []
    for cycle_combo in power_set(cycles):
        if len(cycle_combo) == 0:
            continue
        if len(cycle_combo) == 1:
            permutations.append(cycle_combo)
            continue
        nodes_seen = set()
        cycles_overlap = False
        for cycle in cycle_combo:
            for node in cycle:
                if node in nodes_seen:
                    cycles_overlap = True
                    break
                else:
                    nodes_seen.add(node)
            if cycles_overlap:
                break
        if not cycles_overlap:
            permutations.append(cycle_combo)
    return permutations






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
        self.set_ramp_object_array()
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
        """
        Gets a point x contained in a cell. 

        Input:
            kappa - Cell object
            neighbor_index - (optional) index of a network node
            neighbor_direction - (optional, required if neighbor_index is passed) one of 1 or -1
        Output:
            numpy array with shape (N,1) whose jth entry is the midpoint of of
            pi_j(kappa) if neighbor_index and neighbor_direction are not passed and 
            the midpoint of pi_j(kappa_{neighbor_index}^{neighbor_direction}) if it is. 
        """
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
        x[j]'= -gamma[j,0]*x[j] + R[j]
        """
        Network = self.Network
        ramp_array = self.ramp_object_array
        N = Network.size()
        logic = Network.logic

        def R(x,eps = self._zero):#N= Network.size(),logic = Network.logic,func_array = func_array):
            R_array = np.zeros([N,1])
            for i in range(N):
                cur_prod = 1
                for source_set in logic(i):
                    cur_sum = 0
                    for j in source_set:
                        cur_sum = cur_sum + ramp_array[i,j](x[j],eps[i,j])
                    cur_prod =  cur_prod*cur_sum 
                R_array[i] = cur_prod
            return R_array
        self.R = R


    def _set_func_array(self):
        """
        Creates the func_array attribute.
        """
        Network = self.Network
        ramp_array = self.ramp_object_array
        N = Network.size()
        inputs = Network.inputs
        def func_array(x, eps = self._zero):
            F = np.zeros([N,N])
            for i in range(N):
                for j in inputs(i):
                    Rij = ramp_array[i,j]
                    F[i,j] = Rij(x[j],eps[i,j])
            return F
        
        self.func_array = func_array

    def set_ramp_object_array(self):
        """
        Creates a numpy array containing a RampFunction object in each entry [i,j]
        such that j->i is an edge in the network. 
        """
        N = self.Network.size()
        array = np.empty([N,N],dtype = RampFunction)
        L = self.L
        Delta = self.Delta
        theta = self.theta
        for i in range(N):
            for j in self.Network.inputs(i):
                sign = 1 if self.Network.interaction(j,i) else -1
                array[i,j] = RampFunction(sign,L[i,j],Delta[i,j],theta[i,j])
        self.ramp_object_array = array

    


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
        theta = self.theta
        gamma = self.gamma
        for j in range(N):
            B_j = B[j]
            W_j = W[j]
            for p in range(len(B_j)):
                B_jp = B_j[p]
                if len(B_jp) == 0:
                    continue
                if p != 0 and gamma[j,0]*(theta[B_jp[0],j] - eps[B_jp[0],j]) <= W_j[p]:
                    return False
                if p != len(B_j) - 1 and gamma[j,0]*(theta[B_jp[-1],j] + eps[B_jp[-1],j]) >= W_j[p+1]:
                    return False
        return True

    ##################################
    # Optimal eps
    ##################################

    def get_W(self):
        """
        Create a list which stores they values attained by Lambda. The jth entry 
        is a sorted list with the values of Lambda_j union {0,np.inf}. 
        Output:
             W - (list) length Network.size() list of length 
                 Targets(j) sorted lists. W[j] is the output of get_W_j
        """
        try:
            W = self._W
        except AttributeError:
            W = []
            for j in range(self.Network.size()):
                W_j = self._get_W_j(j)        
                W.append(W_j)    
            self._W = W
        return W
                

    def _get_W_j(self,j):
        """
        Input: 
            j - (int) index of a node
        Output: 
            W_j - (list) The values of Lambda_j union {0,np.inf}, sorted
        """
        W_j_set = {0,np.inf}
        N = self.Network.size()
        #R_j = lambda x: self.R(x)[j,0]
        R = self.R
        inputs = self.Network.inputs(j)
        test_point = np.zeros(N)
        x_low = 1/2*self.theta
        x_high = 2*self.theta
        for low_pattern in power_set(inputs):
            low_pattern =  list(low_pattern)
            high_pattern = list(set(inputs) - set(low_pattern))
            test_point[low_pattern] = x_low[j,low_pattern]
            test_point[high_pattern] = x_high[j,high_pattern]
            W_j_set = W_j_set.union({R(test_point)[j,0]})
        W_j = list(W_j_set) 
        W_j.sort()
        return W_j

    def _get_B_jp(self,j,W_j,p,theta_order):
        """
        Input:
            j - (int) index of a node
            p - (int) requires 0<p<len(W_j)
            W_j - (list) output of get_W_j(j)
            theta_order - list of target indices of node j sorted by size of theta[i,j]
        Output:
            B_jp - (list) indices of thresholds that lie between W_j[p] and W_j[p+1] 
                   ordered by the threshold order. 
        """
        theta = self.theta
        gamma = self.gamma
        B_jp = [i for i in theta_order if (gamma[j,0]*theta[i,j] > W_j[p] and gamma[j,0]*theta[i,j]<W_j[p+1])]
        return B_jp

    def _get_B_j(self,j,W_j,theta_order):
        """
        Input:
            j - (int) index of a node
            W_j - (list) output of get_W_j(j)
            theta_order - list of target indices for node j sorted by size of theta[i,j]
        Output:
            B_j - (list) list of lists of indices. B_j[p] is the output of B_jp

        """
        B_j = []
        for p in range(len(W_j)-1):
            B_jp = self._get_B_jp(j,W_j,p,theta_order)
            B_j.append(B_jp)
        return B_j
            
    def _get_B(self,W):
        """
        Create a list of target indices i for each node j sorted by ordering of theta[i,j]. 
        
        Input:
            W - output of get_W()
        Output: 
            B - (list) The jth entry is a list of length len(W[j]) - 1 where W = get_W(). 
            The pth entry of B[j] is a list of target indices i for node j for which 
            theta[i,j] is between W[j][p] and W[j][p+1]. 
        """
        try:
            B = self._B 
        except AttributeError:
            B = []
            theta_orders = self.all_theta_orders()
            for j in range(self.Network.size()):
                B.append(self._get_B_j(j,W[j],theta_orders[j]))
            self._B = B
        return B

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
        k = len(B_jp) - 1
        D_list = [D_tilde]
        if p != 0:  #W_j[p] != 0
            W_p_dist = (gamma[j,0]*theta[i(0),j] - W_j[p])/(gamma[j,0]*Delta[i(0),j])
            D_list.append(W_p_dist)
        if p != (len(W_j)-2):  #W_j[p+1] != inf
            W_pp1_dist = (W_j[p+1] - gamma[j,0]*theta[i(k),j])/(gamma[j,0]*Delta[i(k),j])
            D_list.append(W_pp1_dist)
        D = min(D_list)
        ##construct eps_jp
        for i in B_jp:
            eps_jp[i,j] = Delta[i,j]*D
        return eps_jp

    def _get_eps_j(self,j,W_j,B_j):
        N = self.Network.size()
        eps_j = np.zeros([N,N])
        for p in range(len(B_j)):
            eps_j += self._get_eps_jp(j,W_j,B_j,p)
        return eps_j

    def optimal_eps(self):
        """
        Get a choice of eps which minimizes the maximum slope for each j among eps
        which satisfy eps'<eps implies (Z,eps') is strongly equivalent to Z. This guarantees
        all equilibrium cells of SWITCH(Z) have corresponding equilibria in R(Z,eps').
        Output:
            eps - numpy array with eps.shape = [N,N]. If j->i is not an edge then 
                  eps[i,j] = 0. 
        """
        Network = self.Network
        W = self.get_W()
        B = self._get_B(W)
        N = Network.size()
        eps = np.zeros([N,N])
        for j in range(N):
            eps += self._get_eps_j(j,W[j],B[j])
        return eps
            
    ###################################
    # optimal theta
    ###################################
    def get_D_jp(self,j,W_j,B_j,p):
        """
        Input:
            j - (int) index of a node
            W_j - (list) output of get_W_j(j)
            B_j - (list of lists) output of get_B_j(j,W_j)
            p - (int) requires 0<p<len(W_j)-1
        """
        if B_j[p] == []:
            return np.inf
        Delta = self.Delta
        gamma = self.gamma
        if p == 0:
            Delta_sum = sum([Delta[i,j] for i in B_j[p][1:]])
            D_jp = W_j[p+1]/(gamma[j,0]*(Delta[B_j[p][0],j] + 2*Delta_sum))
        else:
            Delta_sum = sum([Delta[i,j] for i in B_j[p]])
            D_jp = (W_j[p+1] - W_j[p])/(2*gamma[j,0]*Delta_sum)
        return D_jp

    def get_D_j(self,j,W_j,B_j):
        """
        Input:
            j - (int) index of a node
            W_j - (list) output of get_W_j(j)
            B_j - (list of lists) output of get_B_j(j,W_j)
        """
        D_j = []
        for p in range(0,len(B_j)-1):
            D_j.append(self.get_D_jp(j,W_j,B_j,p))
        if B_j[-1] == []:
            D_j.append(np.inf)
        else: 
            D_j.append(min(D_j))
        return D_j

    def get_D(self,W,B):
        """
        Input:
            W - (list) output of get_W
            B - (list of lists) output of get_B
        Output: 
            D - list of length N. The jth entry is a list of length B[j]. D[j][p] 
                is a scalar which is used by get_theta_jp to get a choice of theta
                which maximally separates the interval (W[j][p],W[j][p+1]). 
        """
        try:
            D = self._D
        except AttributeError:
            D = []
            for j in range(self.Network.size()):
                D.append(self.get_D_j(j,W[j],B[j]))
            self._D = D
        return D

    def get_theta_jp(self,j,W_j,B_j,D_j,p):
        """
        Get a choice of theta which maximall separates teh interval (W_j[p],W_j[p+1]). 
        Input:
            j - (int) index of a node
            W_j - (list) output of get_W_j(j)
            B_j - (list of lists) output of get_B_j(j,W_j)
            p - (int) requires 0<p<=len(W_j)
        Output:
            theta - numpy array wich shape [N,N]. If k != j or i is not in B_j[p]
                    then theta[i,k] = 0. 
        """
        N = self.Network.size()
        theta_jp = np.zeros([N,N])
        B_jp = B_j[p]
        if B_jp == []:
            return theta_jp
        gamma = self.gamma
        Delta = self.Delta
        D_jp = D_j[p]
        if p == 0:
            theta_jp[B_jp[0],j] = 0
        else:
            theta_jp[B_jp[0],j] = (W_j[p] + gamma[j,0]*Delta[B_jp[0],j]*D_jp)/gamma[j,0]
        for q in range(1,len(B_jp)):
            i_q = B_jp[q]
            i_qm1 = B_jp[q-1]
            theta_jp[i_q,j] = theta_jp[i_qm1,j] + (Delta[i_qm1,j] + Delta[i_q,j])*D_jp
        return theta_jp

    def get_theta_j(self,j,W_j,B_j,D_j):
        """
        Get a choice of theta which maximally separates each interval (W_j[p],W_j[p+1]) for 
        each p in range(len(B_j)) = (0,...,len(W_j) - 2). 
        Input:
            j - network node index
            W_j - W[j] where W = get_W()
            B_j - B[j] where B = _get_B(W)
            D_j - D[j] where D = get_D()
        Output:
            theta - numpy array with shape (N,N). If k != j or j->i is not an edge
                    then theta[i,k] = 0. 
        """
        return sum([self.get_theta_jp(j,W_j,B_j,D_j,p) for p in range(len(B_j))])

    def get_redundant_theta_j(self,j,W_j,B_j,max_D):
        """
        Get an optimal choice of theta[:,j] when theta[i,j] is greater than the largest
        value of Lambda_j for each i. 
        Input: 
            j - network node index
            W_j - W[j] where W = get_W()
            B_j - B[j] where B = _get_B(W)
            max_D - Largest value in D := get_D()
        Output:
            theta - numpy array with shape (N,N). If k != j or j->i is not an edge
                    then theta[i,k] = 0. 
        """
        Delta = self.Delta
        N = self.Network.size()
        theta_j = np.zeros([N,N])
        B_jn = B_j[-1]
        theta_j[B_jn[0],j] = W_j[-2] + Delta[B_jn[0],j]*max_D
        for q in range(1,len(B_jn)):
            i_q = B_jn[q]
            i_qm1 = B_jn[q-1]
            theta_j[i_q,j] = theta_j[i_qm1,j] + (Delta[i_qm1,j] + Delta[i_q,j])*max_D
        return theta_j

    def optimal_theta(self):
        """
        Get a choice of theta which produces the best optimal_eps when L, Delta, 
        and the DSGRN parameter node are fixed.  
        Output:
            theta - numpy array with shape (N,N). theta[i,j] = 0 if j->i is not an edge. 
                    theta[i,j] = 0 when j->i is an edge is possible when there is a 
                    threshold theta[i,j] which is smaller than all values attained by 
                    Lambda_j. 
        """
        W = self.get_W()
        B = self._get_B(W)
        D = self.get_D(W,B)
        redundant = []
        N = self.Network.size()
        theta = np.zeros([N,N])
        for j in range(N):
            if len(B[j][-1]) == len(self.Network.outputs(j)): #all thresholds for node j above all words
                redundant.append(j)
            else:
                theta += self.get_theta_j(j,W[j],B[j],D[j])
        if len(redundant) == N: #all thresholds for every node above all words
            theta = self.theta
            theta[theta != 0] = np.inf
            return theta
        max_D = 0
        for j in range(N):
            for p in range(len(D[j])):
                if D[j][p] != np.inf:
                    max_D = max(max_D,D[j][p])
        for j in redundant:
            theta += self.get_redundant_theta_j(j,W[j],B[j],max_D)
        return theta

    def extreme_slopes_with_optimal_theta(self):
        """
        Compute the minimum and maximum slopes attained when theta and eps are 
        chosen optimally. 
        Output:
            (max_slope,min_slope)
        """
        W = self.get_W()
        B = self._get_B(W)
        D = self.get_D(W,B)
        min_D = np.inf
        max_D = 0
        for j in range(self.Network.size()):
            for p in range(len(D[j])):
                if D[j][p] != np.inf:
                    min_D = min(min_D,D[j][p])
                    max_D = max(max_D,D[j][p])
        return 1/(2*min_D), 1/(2*max_D)

    def extreme_hill_coefficients_with_optimal_theta(self):
        """
        Compute the minumum and maximum hill coefficients corresponding to the 
        ramp functions obtained by choosing theta and eps optimally. 
        Output:
            (max_n,min_n)
        """
        W = self.get_W()
        B = self._get_B(W)
        D = self.get_D(W,B)
        theta_opt = self.optimal_theta()
        RFs = self.ramp_object_array
        RF2HF_map = RampToHillFunctionMap()
        largest_n = 1
        smallest_n = np.inf
        for j in range(self.Network.size()):
            for p in range(len(B[j])):
                for i in B[j][p]:
                    RF = RFs[i,j]
                    eps = self.Delta[i,j]*D[j][p]
                    HF = RF2HF_map(RF,eps)
                    n = HF.n
                    if n > largest_n:
                        largest_n = n
                    if n < smallest_n:
                        smallest_n = n
        return largest_n, smallest_n

    
    ####################################

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
                print(gamma)
                if gamma[j,0]*(theta[i,j] + eps[i,j]) in W_j or \
                    gamma[j,0]*(theta[i,j] - eps[i,j]) in W_j:
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
        # theta_j[theta_j == 0] = np.inf
        for i in range(self.Network.size()):
            if i not in self.Network.outputs(j):
                theta_j[i] = np.inf
        index_list = []
        while(sum(theta_j == np.inf) < self.Network.size()):
            i = np.argmin(theta_j)
            index_list.append(i)
            theta_j[i] = np.inf
        return index_list

    def all_theta_orders(self):
        orders = []
        for j in range(self.Network.size()):
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
        Entries corresponding to singular directions are 0
        """
        Lambda = self.Lambda(kappa)
        eq = np.zeros([self.Network.size(),1])
        for r in kappa.regular_directions():
            gamma_r = self.gamma[r,0]
            eq[r,0] = Lambda[r]/gamma_r
        return eq

    def reg_equilibrium_cells_from_FPs(self,FPs):
        theta_orders = self.all_theta_orders()
        eq_cells = []
        for FP in FPs:
            cur_projection = []
            for j, coord in enumerate(FP):
                left_index = coord - 1
                right_index = coord
                if left_index == -1:
                    left_target = -np.inf
                else:
                    left_target = theta_orders[j][left_index]
                if right_index == len(theta_orders[j]):
                    right_target = np.inf
                else:
                    right_target = theta_orders[j][right_index]
                cur_projection.append((left_target,right_target))
            eq_cells.append(Cell(self.theta,*cur_projection))
        return eq_cells

    def reg_equilibria_from_FPs(self,FPs):
        eq_cells = self.reg_equilibrium_cells_from_FPs(FPs)
        eq = []
        Lambda = self.Lambda
        gamma = self.gamma
        for kappa in eq_cells:
            eq.append(Lambda(kappa)/gamma)
        return eq


    def __repr__(self):
        return 'RampSystem(Network = {},L = {},Delta = {},theta = {},gamma = {})'.format(self.Network.specification(),self.L,self.Delta,self.theta,self.gamma)
