"""
Functions for decomposing a RampSystem at a loop characteristic cell. 
"""
import DSGRN
import numpy as np
from ramp_systems.cyclic_feedback_system import CyclicFeedbackSystem
import dsgrn_utilities.graphtranslation as Gtrans
import itertools
import sympy
from ramp_systems.cell import Cell



def get_saddles(RS,LCC):
    """
    Get all the saddle nodes that occur at a loop characteristic cell while (Z,eps)
    is weakly equivalent to (Z,0). Assumes a parameterization of eps that keeps all
    slopes constant within the cyclic feedback system. 

    :param RS: RampSystem object
    :param LCC: Cell object corresponding to a loop characteristic cell of RS.
    :return: list of tuples of the form 
    TO DO: need equilibria values for off cycle directions
    """
    if not RS.is_opaque(LCC):
        return []
    CFS_list = decompose(RS,LCC)
    saddles = []
    s = sympy.symbols('s')
    for CFS, cycle in CFS_list:
        if CFS.cfs_sign == -1:
            continue
        saddle_points, eps_func = CFS.pos_loop_bifurcations()
        CFS_eps_matrix = np.array(eps_func.subs(s,1))
        RS_eps_matrix = CFS_matrix_to_RS_matrix(RS,cycle,CFS_eps_matrix)
        for s_val, x_val in saddle_points:
            eps = RS_eps_matrix*s_val
            if RS.is_weakly_equivalent(eps):
                RS_eps_func = sympy.Matrix(RS_eps_matrix)*s
                saddles.append((s_val,CFS_vector_to_RS_vector(RS,cycle,x_val),RS_eps_func))
    return saddles



def CFS_vector_to_RS_vector(RS,cycle,CFS_vector):
    N = RS.Network.size()
    RS_vector = np.zeros([N,1])
    cycle_N = len(cycle)
    for j in range(cycle_N):
        RS_vector[cycle[j],0] = CFS_vector[j,0]
    return RS_vector

def CFS_matrix_to_RS_matrix(RS,cycle,CFS_matrix):
    """
    Create a N x N numpy array which has the entries of CFS_matrix in the appropriate
    positions where N = RS.Network.size()

    :param RS: RampSystem object
    :param cycle: list defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]
    :param CFS_matrix: N x N numpy array
    """
    N = RS.Network.size()
    RS_matrix = np.zeros([N,N])
    cycle_N = len(cycle)
    for j in range(cycle_N):
        if j < cycle_N-1:
            jplus1 = j+1
        else: 
            jplus1 = 0
        RS_matrix[cycle[jplus1],cycle[j]] = CFS_matrix[jplus1,j]
    return RS_matrix



def decompose(RS,LCC):
    """
    Decompose a RampSystem into CyclicFeedbackSystems at a loop characteristic
    cell. 

    :param RS: RampSystem object
    :param LCC: Loop characteristic cell represeneted as a Cell object.
    :return: list of tuples of the form (CFS,cycle) where CFS is a CyclicFeedbackSystem
    object and cycle is the cycle in RS associated to CFS. cycle[j] is the node of RS
    associated to node j of CFS.
    """
    if not RS.is_regular():
        raise ValueError('decompose requires RampSystem parameters are regular.')
    if not np.array_equal(RS.theta,LCC.theta):
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
    :param LCC: Loop characteristic cell represeneted as a Cell object. 
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
    :param LCC: Loop characteristic cell represeneted as a Cell object. 
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
    



