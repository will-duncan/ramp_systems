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
    Get all the saddle nodes that occur at a loop characteristic cell 
    while (Z,eps) is weakly equivalent to (Z,0). Uses a parameterization 
    of eps that keeps all slopes constant within the cyclic feedback system. 

    :param RS: RampSystem object
    :param LCC: Cell object corresponding to a loop characteristic cell of RS.
    :return: list of tuples of the form (s_val,(x_val,stable),eps_func,border_crossing_index)
    """
    CFS_list = decompose(RS,LCC)
    saddles = {CFS[1]:[] for CFS in CFS_list}
    if not RS.is_opaque(LCC):
        return saddles
    equilibria_dict = cycle_equilibria(RS,LCC,CFS_list)
    s = sympy.symbols('s')
    for CFS,cycle in CFS_list:
        if CFS.cfs_sign == -1:
            #negative CFSs don't have saddle nodes
            continue
        saddle_points, eps_func = CFS.get_bifurcations()
        CFS_eps_matrix = np.array(eps_func.subs(s,1))
        RS_eps_matrix = CFS_matrix_to_RS_matrix(RS,cycle,CFS_eps_matrix)
        for j in range(len(saddle_points)):
            for s_val, x_val in saddle_points[j]:
                eps = RS_eps_matrix*s_val
                if RS.is_weakly_equivalent(eps):
                    saddle_dict = equilibria_dict.copy()
                    # get off cycle equilibria. stable set to True for this x_val
                    # is a hack so that CFS_equilibria_RS_equilibria gives stable == True
                    # when the off cycle equilibrium is stable. 
                    saddle_dict[cycle] = [(x_val,True)] 
                    RS_saddles = CFS_equilibria_to_RS_equilibria(RS,LCC,saddle_dict,CFS_list)
                    RS_eps_func = sympy.Matrix(RS_eps_matrix)*s
                    saddles[cycle].extend([(s_val,RS_saddles[i],RS_eps_func,j) for i in range(len(RS_saddles))])
    return saddles

def cycle_equilibria(RS,LCC,CFS_list):
    """
    Return a dictionary of lists of equilibria of the cyclic feedback systems in 
    the cyclic feedback system decomposition of RS at LCC with eps = 0. 

    :param RS: RampSystem object
    :param LCC: Cell object, required to be a loop characteristic cell.
    :param CFS_list: output of decompose(RS,LCC)
    :return: dictionary with entries cycle:equilibria where equilibria is a list of 
    tuples of the form (eq_val, stable)
    """
    equilibria_dict = {CFS[1]:[] for CFS in CFS_list}
    for CFS,cycle in CFS_list:
        eq_list = CFS.equilibria()
        for cycle_eq in eq_list:
            if in_cell_neighborhood(LCC,cycle,cycle_eq):
                equilibria_dict[cycle].append(cycle_eq)
    return equilibria_dict

def CFS_equilibria_to_RS_equilibria(RS,LCC,CFS_eq_dict,CFS_list):
    """
    Compute all the RS_equilibria defined by the equilibria in CFS_eq_dict.

    :param RS: RampSystem object
    :param LCC: Cell object which is assumed to be a loop characteristic cell
    :param CFS_eq_dict: dictionary of cyclic feedback system equilibria. keys must be 
    the cycles in CFS_list and values are lists of equilibria corresponding to the cycle.
    :param CFS_list: output of decompose(RS,LCC)
    :return: list of tuples of the form (eq_val,stable). stable is a boolean which is 
    True if the equilibrium at eq_val is stable. 
    """
    reg_eq = RS.equilibria_regular_directions(LCC)
    RS_eq_list = []
    cycle_list = [cycle for _,cycle in CFS_list]
    for eq_parts in itertools.product(*[CFS_eq_dict[cycle] for cycle in cycle_list]):
        cur_eq_val = reg_eq.copy()
        cur_eq_stable = True
        for i in range(len(eq_parts)):
            cycle_contribution = CFS_vector_to_RS_vector(RS,cycle_list[i],eq_parts[i][0])
            cur_eq_val += cycle_contribution
            if cur_eq_stable and not eq_parts[i][1]:
                cur_eq_stable = False
        RS_eq_list.append((cur_eq_val,cur_eq_stable))
    return RS_eq_list


def CFS_vector_to_RS_vector(RS,cycle,CFS_vector,off_cycle_vec = None):
    N = RS.Network.size()
    if off_cycle_vec is None:
        RS_vector = np.zeros([N,1])
    else: 
        RS_vector = off_cycle_vec
    cycle_N = len(cycle)
    for j in range(cycle_N):
        RS_vector[cycle[j],0] = CFS_vector[j,0]
    return RS_vector

def RS_vector_to_CFS_vector(cycle,RS_vector):
    CFS_vector = np.zeros([len(cycle),1])
    for j in range(len(cycle)):
        CFS_vector[j,0] = RS_vector[cycle[j],0]
    return CFS_vector

def CFS_matrix_to_RS_matrix(RS,cycle,CFS_matrix):
    """
    Create a N x N numpy array which has the entries of CFS_matrix in the appropriate
    positions where N = RS.Network.size()

    :param RS: RampSystem object
    :param cycle: tuple defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]
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

def RS_matrix_to_CFS_matrix(cycle,RS_matrix):
    """
    Create a len(cycle) x len(cycle) numpy array CFS_matrix satisfying CFS_matrix[j+1,j] = RS_matrix[cycle[j+1],cycle[j]]

    :param cycle: tuple defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]
    :param RS_matrix: square numpy array
    :return: square numpy array of size len(cycle) x len(cycle)
    """
    cycle_N = len(cycle)
    CFS_matrix = np.zeros([cycle_N,cycle_N])
    for j in range(cycle_N-1):
        CFS_matrix[j+1,j] = RS_matrix[cycle[j+1],cycle[j]]
    CFS_matrix[0,-1] = RS_matrix[cycle[0],cycle[-1]]
    return CFS_matrix



def in_cell_neighborhood(tau,cycle,cycle_eq):
    """
    Returns True if cycle_eq_val lies in the neighborhood of loop characteristic 
    cell tau projected onto the cycle directions and False otherwise.

    :param tau: Cell object
    :param cycle: tuple defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]
    :param cycle_eq: tuple of the form (eq_val,stable)
    :return: bool
    """
    eq_val = cycle_eq[0]
    for j in range(len(cycle)):
        if j < len(cycle) - 1:
            jplus1 = j + 1
        else: 
            jplus1 = 0
        left = tau.theta_rho_minus(cycle[j])
        right = tau.theta_rho_plus(cycle[j])
        if eq_val[j] < left or eq_val[j] > right:
            return False
    return True 
    
    
def decompose(RS,LCC):
    """
    Decompose a RampSystem into CyclicFeedbackSystems at a loop characteristic
    cell. 

    :param RS: RampSystem object
    :param LCC: Loop characteristic cell represeneted as a Cell object.
    :return: list of tuples of the form (CFS,cycle) where CFS is a CyclicFeedbackSystem
    object and cycle is a tuple representing the cycle in RS associated to CFS. 
    cycle[j] is the node of RS associated to node j of CFS.
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
        CFS_list.append((get_CFS_from_cycle(RS,cur_cycle,LCC),tuple(cur_cycle)))
    return CFS_list

def get_CFS_from_cycle(RS,cycle,LCC):
    """
    Creates a CyclicFeedbackSystem object which corresponds to cycle at the loop
    characteristic cell defined by rho

    :param RS: RampSystem object
    :param cycle: tuple defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]
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
    :param cycle: tuple defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]
    Each entry should be a singular direction of LCC.
    :param LCC: Loop characteristic cell represeneted as a Cell object. 
    :return: (L,Delta), two len(cycle) x len(cycle) numpy arrays giving the L and Delta values
    for the cycle decomposition.
    """
    cycle_N = len(cycle)
    cycle_L = np.zeros([cycle_N,cycle_N])
    cycle_Delta = np.zeros([cycle_N,cycle_N])
    for j in range(cycle_N):
        if j < cycle_N - 1:
            jplus1 = j+1
        else: 
            jplus1 = 0
        Lambda_left = RS.Lambda(LCC,cycle[j],-1)[cycle[jplus1]]
        Lambda_right = RS.Lambda(LCC,cycle[j],1)[cycle[jplus1]]
        if Lambda_left < Lambda_right:
            cycle_L[jplus1,j] = Lambda_left
            cycle_Delta[jplus1,j] = Lambda_right - Lambda_left
        else:
            cycle_L[jplus1,j] = Lambda_right
            cycle_Delta[jplus1,j] = Lambda_left - Lambda_right
    return cycle_L,cycle_Delta

def get_cycle_thresholds(RS,cycle):
    """
    Get the threshold values associated with a cycle of a ramp system. 
    
    :param RS: RampSystem object
    :param cycle: tuple defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]   
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
    :param cycle: tuple defining a cycle. The cycle is defined by cycle[0]->cycle[1] ->...->cycle[n]->cycle[0]   
    :return: DSGRN network object, cycle_net. Node j of cycle_net corresponds to node
    cycle[j] of network. All nodes are specified as inessential.
    """
    cycle_spec = ''
    for j in range(len(cycle)):
        if j > 0:
            jminus1 = j - 1
        else:
            jminus1 = len(cycle) - 1
        if not network.interaction(cycle[jminus1],cycle[j]):
            interaction_str = '~'
        else: 
            interaction_str = ''
        cycle_spec += network.name(cycle[j]) + ' : ' + interaction_str + network.name(cycle[jminus1]) + '\n' 
    return DSGRN.Network(cycle_spec)
    



