from ramp_systems.decomposition import *
import DSGRN
from ramp_systems.ramp_system import RampSystem
from ramp_systems.cyclic_feedback_system import CyclicFeedbackSystem
import sympy

def test_get_saddles():
    ## test on two independent toggle switches
    N,L,Delta,theta,gamma = two_independent_toggles()
    RS = RampSystem(N,L,Delta,theta,gamma)
    # loop characteristic cell with only first loop
    LCC = Cell(RS.theta,1,0,(3,np.inf),(-np.inf,2))
    saddles = get_saddles(RS,LCC)
    CFS = CyclicFeedbackSystem(*neg_edge_toggle())
    CFS_saddles,eps_func = CFS.get_bifurcations()
    s = sympy.symbols('s')
    expected_saddle_val = np.zeros([4,1])
    CFS_saddle_val = CFS_saddles[0][0][1]
    expected_saddle_val = np.array([[CFS_saddle_val[0,0]],[CFS_saddle_val[1,0]],[L[3,2]+ Delta[3,2]],[L[2,3]]])
    for saddle in saddles[(0,1)]:
        eps = saddle[2].subs(s,saddle[0])
        assert(np.allclose(RS(saddle[1][0],eps),np.zeros([4,1])))
        assert(np.array_equal(saddle[1][0],expected_saddle_val))
    # LCC with both loops
    LCC = Cell(RS.theta,1,0,3,2)
    saddles = get_saddles(RS,LCC)
    assert(len(saddles[(0,1)]) == 3)
    assert(len(saddles[(2,3)]) == 3)

    ## test that throwing out bifurcations that occur past weak equivalence are thrown out
    N,L,Delta,theta,gamma = almost_two_independent_toggles()
    RS = RampSystem(N,L,Delta,theta,gamma)
    LCC = Cell(RS.theta,1,0,(3,np.inf),(-np.inf,2))
    saddles = get_saddles(RS,LCC)
    assert(len(saddles[(0,1)]) == 0)
    theta[2,0] = .3
    Delta[2,0] = .5
    theta[3,2] = 1.1
    RS = RampSystem(N,L,Delta,theta,gamma)
    print(RS.Delta,Delta)
    LCC.theta = RS.theta
    saddles = get_saddles(RS,LCC)
    assert(len(saddles[(0,1)]) == 1)
    

    


def test_decompose():
    RS = RampSystem(*toggle_plus_parameters())
    ## test getting parameters
    cycle = (0,1)
    cycle_theta = get_cycle_thresholds(RS,cycle)
    assert(np.array_equal(cycle_theta,np.array([[0,RS.theta[0,1]],[RS.theta[1,0],0]])))
    LCC = Cell(RS.theta,1,0)
    cycle_L,cycle_Delta = get_cycle_L_and_Delta(RS,cycle,LCC)
    L01 = (RS.L[0,0] + RS.Delta[0,0])*RS.L[0,1]
    Delta01 = (RS.L[0,0] + RS.Delta[0,0])*(RS.L[0,1] + RS.Delta[0,1]) - L01
    L10 = RS.L[1,0]*(RS.L[1,1] + RS.Delta[1,1])
    Delta10 = (RS.L[1,0] + RS.Delta[1,0])*(RS.L[1,1] + RS.Delta[1,1]) - L10
    assert(np.array_equal(cycle_L,np.array([[0,L01],[L10,0]])))
    assert(np.array_equal(cycle_Delta,np.array([[0,Delta01],[Delta10,0]])))
    ## test decompose
    CFS_list = decompose(RS,LCC)
    assert(len(CFS_list) == 1)
    CFS = CFS_list[0][0]
    cycle_out = CFS_list[0][1]
    cycle_net = DSGRN.Network('X0 : X1 \n X1 : X0')
    assert(cycle_out == cycle)
    assert(CFS == CyclicFeedbackSystem(cycle_net,cycle_L,cycle_Delta,cycle_theta,RS.gamma))

    ## test getting parameters
    cycle = (0,)
    cycle_theta = get_cycle_thresholds(RS,cycle)
    assert(np.array_equal(cycle_theta,np.array([[RS.theta[0,0]]])))
    LCC = Cell(RS.theta,0,(0,np.inf))
    cycle_L, cycle_Delta = get_cycle_L_and_Delta(RS,cycle,LCC)
    L00 = RS.L[0,0]*RS.L[0,1]
    Delta00 = (RS.L[0,0] + RS.Delta[0,0])*RS.L[0,1] - L00
    assert(cycle_L == L00)
    assert(np.array_equal(cycle_Delta,np.array([[Delta00]])))
    ## test decompose
    CFS_list = decompose(RS,LCC)
    assert(len(CFS_list) == 1)
    CFS = CFS_list[0][0]
    cycle_out = CFS_list[0][1]
    cycle_net = DSGRN.Network('X0 : X0')
    assert(cycle_out == cycle)
    assert(CFS == CyclicFeedbackSystem(cycle_net,cycle_L,cycle_Delta,cycle_theta,RS.gamma[0,0]))

    ## test decompose
    LCC = Cell(RS.theta,0,1)
    CFS_list = decompose(RS,LCC)
    assert(len(CFS_list) == 2)
    cycle_list = [CFS_list[0][1],CFS_list[1][1]]
    assert((0,) in cycle_list and (1,) in cycle_list)

    ## decompose two independent toggles
    RS = RampSystem(*two_independent_toggles())
    LCC = Cell(RS.theta,1,0,(3,np.inf),(-np.inf,2))
    CFS_list = decompose(RS,LCC)
    assert(len(CFS_list) == 1)
    assert(CFS_list[0][1] == (0,1))
    CFS = CyclicFeedbackSystem(*neg_edge_toggle())
    assert(CFS_list[0][0] == CFS)
    
  

def test_make_cycle_subnetwork():
    network = four_node_network()
    cycle = (0,1)
    cycle_net = make_cycle_subnetwork(network,cycle)
    assert(cycle_net.name(0) == network.name(0))
    assert(cycle_net.name(1) == network.name(1))
    assert(cycle_net.outputs(0) == [1])
    assert(cycle_net.outputs(1) == [0])
    assert(cycle_net.interaction(0,1) == cycle_net.interaction(0,1))
    assert(cycle_net.interaction(1,0) == cycle_net.interaction(1,0))

    cycle = (0,3,1)
    cycle_net = make_cycle_subnetwork(network,cycle)
    for j in range(len(cycle)):
        assert(cycle_net.name(j) == network.name(cycle[j]))
    
    network,L,Delta,theta,gamma = two_independent_toggles()
    cycle = (0,1)
    cycle_net = make_cycle_subnetwork(network,cycle)
    assert(cycle_net.outputs(0) == [1])
    assert(cycle_net.outputs(1) == [0])
    assert(cycle_net.inputs(0) == [1])
    assert(cycle_net.outputs(1) == [0])
    
def test_Cell():
    theta = np.array([[1,2,3],[2,1,2],[3,3,1]],dtype = 'float')
    LCC = Cell(theta,(1,2),2,1)
    assert(LCC.regular_directions() == {0})
    assert(LCC.singular_directions() == {1,2})
    assert(LCC.pi == [(1,2),(2,),(1,)])
    assert(LCC(0) == (1,2))
    assert(LCC(2) == (1,))
    assert(LCC.rho == [0,2,1])
    
    assert(LCC.rho_plus(1) == np.inf)
    assert(LCC.rho_minus(1) == 0)
    assert(LCC.rho_plus(2) == 0)
    assert(LCC.rho_minus(2) == 2)
    assert(LCC.theta_rho_minus(2) == 1)
    assert(LCC.theta_rho_plus(2) == 3)

    top_cell_list = [Cell(theta,(1,2),(0,2),(2,1)),Cell(theta,(1,2),(0,2),(1,0)),\
        Cell(theta,(1,2),(2,np.inf),(2,1)), Cell(theta,(1,2),(2,np.inf),(1,0)) ]
    top_cells_out = LCC.top_cells()
    assert(len(top_cells_out) == len(top_cell_list))
    for cell in top_cells_out:
        assert(cell in top_cell_list)


##############
## Networks ##
#############

def four_node_network():
    spec = 'X0 : (X2)(~X1)\n X1: (X0)(X3)\n X2 : X1\n X3 : (X2)(X0)'
    return DSGRN.Network(spec)

def toggle_plus_parameters():
    """Theta is chosen optimally"""
    N = DSGRN.Network("X0 : (X0)(~X1) \n X1 : (X0)(X1)")
    #W[0] = [0,4.5,5.5,13.5,16.5,inf]
    #W[1] = [0,3.5,10.5,12.5,37.5,inf]
    L = np.array([[2,2.25],[.7,5]])
    Delta = np.array([[4,.5],[1.8,10]])
    theta = np.array([[11,24],[15,7]])
    gamma = np.array([1,1])
    return N,L,Delta,theta,gamma

def two_independent_toggles():
    #tests assume these parameter values
    N = DSGRN.Network("X0 : ~X1 \n X1 : ~X0 \n X2 : ~X3 \n X3 : ~X2")
    L = np.array([[0,.5,0,0],[.5,0,0,0],[0,0,0,.5],[0,0,.5,0]])
    Delta = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]],dtype='float')
    theta = np.array([[0,1.3,0,0],[1,0,0,0],[0,0,0,1.3],[0,0,1,0]])
    gamma = np.array([1,1,1,1])
    return N,L,Delta,theta,gamma

def neg_edge_toggle():
    #tests assume these parameter values
    N = DSGRN.Network("X0 : ~X1 \n X1 : ~X0")
    L = np.array([[0,.5],[.5,0]])
    Delta = np.array([[0,1],[1,0]])
    theta = np.array([[0,1.3],[1,0]])
    gamma = np.array([1,1])
    return N,L,Delta,theta,gamma

def almost_two_independent_toggles():
    N = DSGRN.Network("X0 : ~X1 \n X1 : ~X0 \n X2 : (X0)(~X3) \n X3 : ~X2")
    L = np.array([[0,.5,0,.0],[.5,0,0,0],[.5,0,0,.5],[0,0,.5,0]])
    Delta = np.array([[0,1,0,0],[1,0,0,0],[1,0,0,1],[0,0,1,0]],dtype='float')
    theta = np.array([[0,1.3,0,0],[1,0,0,0],[1,0,0,1.3],[0,0,1,0]])
    gamma = np.array([1,1,1,1])
    return N,L,Delta,theta,gamma