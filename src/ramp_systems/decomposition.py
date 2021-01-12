"""
Functions for decomposing a RampSystem at a loop characteristic cell. 
"""
import DSGRN
import numpy as np

def decompose(RS,LCC):
    """
    Decompose a RampSystem into CyclicFeedbackSystems at a loop characteristic
    cell. 

    :param RS: RampSystem object
    :param LCC: loop characteristic cell object 
    :return: list of tuples of the form (CFS,cycle) where CFS is a CyclicFeedbackSystem
    object and cycle is the cycle in RS associated to CFS. 
    """
    if not RS.is_regular():
        raise ValueError('decompose requires RampSystem parameters are regular.')
    if not np.allequal(RS.theta,LCC.theta):
        raise ValueError('theta for RS and LCC must agree.')
    nodes_seen = set()
    CFS_list = []
    rho = LCC.rho
    for j in LCC.singular_directions():
        if j in nodes_seen:
            continue
        cur_cycle = []
        cur_node = j
        next_node = -1
        while next_node != j:
            nodes_seen.add(cur_node)
            cur_cycle.append(cur_node)
            cur_node = rho[cur_node]
            next_node = cur_node
        CFS_list.append((get_CFS_from_cycle(RS,cur_cycle,LCC),cur_cycle))
    return CFS_list

def get_CFS_from_cycle(RS,cycle,LCC):
    """
    Creates a CyclicFeedbackSystem object which corresponds to cycle at the loop
    characteristic cell defined by rho

    :param RS: RampSystem object
    :param cycle: list defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]
    :param LCC: LoopCharacteristicCell object. 
    :return: CyclicFeedbackSystem object, CFS. CFS.Network node j corresponds to 
    RS.Network node cycle[j]. 
    """
    CFS_network = make_cycle_subnetwork(RS.Network,cycle)
    CFS_theta = get_cycle_thresholds(RS,cycle)
    CFS_L,CFS_Delta = get_cycle_L_and_Delta(RS,cycle,LCC)
    CFS_gamma = get_cycle_gamma(RS,cycle)
    return CyclicFeedbackSystem(CFS_network,CFS_L,CFS_Delta,CFS_theta,CFS_gamma)

def get_cycle_gamma(RS,cycle):
    gamma = np.array([RS.gamma[cycle[j]] for j in range(len(cycle))])
    return gamma

def get_cycle_L_and_Delta(RS,cycle,LCC):
    """
    Get the L and Delta values of a cycle associated with a ramp system near the 
    loop characteristic cell defined by rho. 
    
    :param RS: RampSystem object
    :param cycle: list defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]
    Each entry should be a singular direction of LCC.
    :param LCC: LoopCharacteristicCell object.
    :return: (L,Delta), two len(cycle) x len(cycle) numpy arrays giving the L and Delta values
    for the cycle decomposition.
    """
    cycle_N = len(cycle)
    cycle_L = np.zeros([cycle_N,cycle_N])
    cycle_Delta = np.zeros([cycle_N,cycle_N])
    test_point = np.zeros([RS.Network.size(),1])
    theta = RS.theta
    for r in LCC.regular_directions():
        theta_low = 0 if LCC(r)[0] == -np.inf else theta[LCC(r)[0],r]
        theta_high = 2*theta_low if LCC(r)[1] == np.inf else theta[LCC(r)[1],r]
        test_point[r,0] = (theta_low + theta_high)/2
    for s in LCC.singular_directions():
        test_point[s,0] = theta[LCC(s)[0],s]
    for j in range(cycle_N):
        if j < cycle_N - 1:
            jplus1 = j+1
        else: 
            jplus1 = 0
        low_test_point = test_point.copy()
        low_test_point[cycle[j]] = (theta[cycle[jplus1],cycle[j]] + LCC.theta_rho_minus(cycle[j]))/2
        Lambda_low = RS.R(low_test_point)[cycle[jplus1]]
        high_test_point = test_point.copy()
        theta_rho_plus = LCC.theta_rho_plus(cycle[j])
        if theta_rho_plus == np.inf:
            high_test_point[cycle[j]] = 2*theta[cycle[jplus1],cycle[j]]
        else: 
            high_test_point[cycle[j]] = (theta[cycle[jplus1],cycle[j]] + theta_rho_plus)/2
        Lambda_high = RS.R(high_test_point)[cycle[jplus1]]
        print(j,low_test_point,high_test_point)
        if Lambda_low < Lambda_high:
            cycle_L[jplus1,j] = Lambda_low
            cycle_Delta[jplus1,j] = Lambda_high - Lambda_low
        else:
            cycle_L[jplus1,j] = Lambda_high
            cycle_Delta[jplus1,j] = Lambda_low - Lambda_high
    return cycle_L, cycle_Delta
    
        


def get_cycle_thresholds(RS,cycle):
    """
    Get the threshold values associated with a cycle of a ramp system. 
    
    :param RS: RampSystem object
    :param cycle: list defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]   
    :return: numpy array of size len(cycle) x len(cycle) 
    """
    cycle_N = len(cycle)
    cycle_theta = np.zeros([cycle_N,cycle_N])
    cycle_theta[0,-1] = RS.theta[cycle[0],cycle[-1]]
    for j in range(len(cycle)-1):
        cycle_theta[j+1,j] = RS.theta[cycle[j+1],cycle[j]]
    return cycle_theta

def make_cycle_subnetwork(network,cycle):
    """
    Create the DSGRN network described by cycle which is a subnetwork of network.

    :param network: DSGRN network object
    :param cycle: list defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]   
    :return: DSGRN network object, cycle_net. Node j of cycle_net corresponds to node
    cycle[j] of network. All nodes are specified as inessential.
    """
    cycle_spec = network.name(cycle[0]) + ' : ' + network.name(cycle[-1]) + '\n'
    for j in range(len(cycle)-1):
        cycle_spec += network.name(cycle[j+1]) + ' : ' + network.name(cycle[j]) + '\n' 
    return DSGRN.Network(cycle_spec)
    

#########################################################
## Loop Characteristic Cell object
#########################################################
class LoopCharacteristicCell:
    
    def __init__(self,theta,*projections):
        """
        Create a LoopCharacteristicCell object.

        :param *projections: jth argument is either a pair of indices i_1, i_2 or single
        index i which correspond to target node(s) of node j. The jth projection of the
        cell is given by (theta[i_1,j],theta[i_2,j]) or {theta[i,j]}, respectively. 
        """
        self.theta = np.array(theta,dtype = 'float')
        self.pi = [(projections[j],) if type(projections[j]) == int else tuple(projections[j]) for j in range(len(projections))]
        self._set_rho()

    def regular_directions(self):
        #only compute this once
        try:
            reg_dir = self._reg_dir
        except AttributeError:
            reg_dir = set(j for j in range(len(self.pi)) if len(self.pi[j]) == 2)
            self._reg_dir = reg_dir
        return reg_dir

    def singular_directions(self):
        #only compute this once
        try: 
            sin_dir = self._sin_dir
        except AttributeError:
            sin_dir = set(j for j in range(len(self.pi)) if len(self.pi[j]) == 1)
            self._sin_dir = sin_dir
        return sin_dir

    def __call__(self,j):
        return self.pi[j]

    def _set_rho(self):
        rho = [[] for i in range(len(self.pi))]
        for j in self.regular_directions():
            rho[j] = j
        for j in self.singular_directions():
            rho[j] = self.pi[j][0]
        self.rho = rho

    def rho_plus(self,j):
        """
        Compute the index i where theta[rho[j],j]<theta[i,j] are consecutive
        thresholds.

        :param j: singular direction of the cell
        """
        rho = self.rho
        theta = self.theta
        if theta[rho[j],j] == theta[:,j].max():
            return np.inf
        else:
            difference_array = theta[:,j] - theta[rho[j],j]
            difference_array[difference_array <= 0] = np.inf
            return np.argmin(difference_array)
    
    def rho_minus(self,j):
        """
        Compute the index i where theta[i,j]<theta[rho[j],j] are consecutive thresholds.

        :param j: singular direction of the cell
        """
        rho = self.rho
        theta = self.theta
        if theta[rho[j],j] == theta[:,j].min():
            return -np.inf
        else:
            difference_array = theta[:,j] - theta[rho[j],j]
            difference_array[difference_array >= 0] = -np.inf
            return np.argmax(difference_array)
    
    def theta_rho_minus(self,j):
        rho_minus = self.rho_minus(j)
        if rho_minus == -np.inf:
            return 0
        else: 
            return self.theta[rho_minus,j]

    def theta_rho_plus(self,j):
        rho_plus = self.rho_plus(j)
        if rho_plus == np.inf:
            return np.inf
        else:
            return self.theta[rho_plus,j]