
from ramp_systems.ramp_function import RampFunction
from ramp_systems.ramp_system import RampSystem
from ramp_systems.cell import Cell
import numpy as np
from ramp_to_hill.saddle_node_maps import *
from ramp_systems.cyclic_feedback_system import CyclicFeedbackSystem
import sympy
import DSGRN
import ramp_systems.decomposition as decomposition

def test_ramp_to_hill_function_map():
    the_map = RampToHillFunctionMap()
    RF = RampFunction(sign = 1,L=1,Delta=1,theta=1)
    eps = .5
    out = the_map(RF,eps)
    assert(out.L == RF.L)
    assert(out.Delta == RF.Delta)
    assert(out.sign == RF.sign)
    assert(out.theta == RF.theta)
    assert(np.allclose(out.n,7.8715805,rtol = 1e-5))
    
    RF = RampFunction(sign = 1,L=1,Delta=1,theta=1)
    eps = .1
    out = the_map(RF,eps)
    assert(out.sign == RF.sign)
    assert(np.allclose(out.n,23.9582121,rtol = 1e-5))

class TestRampToHillSaddleMap:
    
    def test_ramp_system_map(self):
        #test on two independent toggle switches
        N,L,Delta,theta,gamma = self.two_independent_toggles()
        RS = RampSystem(N,L,Delta,theta,gamma)
        the_map = RampToHillSaddleMap(N)
        #loop characteristic cell with only first loop
        LCC = Cell(RS.theta,1,0,(3,np.inf),(-np.inf,2))
        saddles = decomposition.get_saddles(RS,LCC)
        saddle = saddles[(0,1)][0]
        hill_system, x_hill = the_map.ramp_system_map(RS,saddle,(0,1),LCC)
        assert(np.array_equal(hill_system.n[2:4,2:4], np.array([[0,np.inf],[np.inf,0]])))
        assert(np.array_equal(hill_system.Delta,Delta))
        assert(np.array_equal(hill_system.gamma,gamma))
        assert(np.array_equal(hill_system.theta,theta))
        assert(np.allclose(hill_system.n[0,1],12.1399,rtol = 1e-2))
        assert(np.allclose(hill_system.n[1,0],10.2267,rtol = 1e-2))
        assert(np.allclose(hill_system.L[0,1],.6560,rtol = 1e-2))
        assert(np.allclose(hill_system.L[1,0],.5981,rtol = 1e-2))
        assert(np.allclose(x_hill,np.array([[.7958],[1.5098],[L[3,2]+Delta[3,2]],[L[2,3]]]),rtol=1e-2))

        #toggle plus multiplicative interactions
        N,L,Delta,theta,gamma = self.toggle_plus_multiplicative()
        RS = RampSystem(N,L,Delta,theta,gamma)
        the_map = RampToHillSaddleMap(N)
        LCC = Cell(RS.theta,1,0)
        saddles = decomposition.get_saddles(RS,LCC)
        saddle = saddles[(0,1)][0]
        hill_saddles = the_map.map_all_saddles(RS,LCC)
        assert(len(hill_saddles[(0,1)]) == 1)
        hill_system,x_hill,stable = hill_saddles[(0,1)][0]
        assert(stable == True)
        assert(np.array_equal(hill_system.Delta,Delta))
        assert(np.array_equal(hill_system.gamma,gamma))
        assert(np.array_equal(hill_system.theta,theta))
        assert(hill_system.n[0,0] == np.inf)
        assert(hill_system.n[1,1] == np.inf)
        assert(np.allclose(hill_system.n[0,1],12.1399,rtol = 1e-2))
        assert(np.allclose(hill_system.n[1,0],10.2267,rtol = 1e-2))
        assert(np.allclose(hill_system.L[0,1],.6560,rtol = 1e-2))
        assert(np.allclose(hill_system.L[1,0],.5981,rtol = 1e-2))
        assert(hill_system.L[1,1] == L[1,1])
        assert(hill_system.L[0,0] == L[0,0])
        assert(np.allclose(x_hill,np.array([[.7958],[1.5098]]),rtol=1e-2))

        #toggle plus additive interactions
        N,L,Delta,theta,gamma = self.toggle_plus_additive()
        RS = RampSystem(N,L,Delta,theta,gamma)
        the_map = RampToHillSaddleMap(N)
        LCC = Cell(RS.theta,1,0)
        saddles = decomposition.get_saddles(RS,LCC)
        saddle = saddles[(0,1)][0]
        hill_system,x_hill = the_map.ramp_system_map(RS,saddle,(0,1),LCC)
        assert(np.array_equal(hill_system.Delta,Delta))
        assert(np.array_equal(hill_system.gamma,gamma))
        assert(np.array_equal(hill_system.theta,theta))
        assert(hill_system.n[0,0] == np.inf)
        assert(hill_system.n[1,1] == np.inf)
        assert(np.allclose(hill_system.n[0,1],12.1399,rtol = 1e-2))
        assert(np.allclose(hill_system.n[1,0],10.2267,rtol = 1e-2))
        assert(hill_system.L[1,1] == L[1,1])
        assert(hill_system.L[0,0] == L[0,0])
        assert(np.allclose(hill_system.L[0,1],.4060,rtol=1e-2))
        assert(np.allclose(hill_system.L[1,0],.3481,rtol=1e-2))
        assert(np.allclose(x_hill,np.array([[.7958],[1.5098]]),rtol=1e-2))

        #test adjustment of off cycle equilibria
        N,L,Delta,theta,gamma = self.almost_two_independent_toggles()
        RS = RampSystem(N,L,Delta,theta,gamma)
        the_map = RampToHillSaddleMap(N)
        LCC = Cell(RS.theta,1,0,(3,np.inf),(-np.inf,2))
        saddles = decomposition.get_saddles(RS,LCC)
        saddle = saddles[(0,1)][0]
        hill_system,x_hill = the_map.ramp_system_map(RS,saddle,(0,1),LCC)
        assert(np.allclose(x_hill,np.array([[.7958],[1.5098],[L[3,2]+Delta[3,2]],[L[2,3]]]),rtol=1e-2))
        assert(hill_system.is_equilibrium(x_hill))
        assert(hill_system.is_saddle(x_hill))


    def test_cyclic_feedback_system_map(self):
        N,L,Delta,theta,gamma = self.neg_edge_toggle()
        L[0,1] = .5
        Delta[0,1] = 1
        theta[0,1] = 1.3
        L[1,0] = .5
        Delta[1,0] = 1
        theta[1,0] = 1
        gamma = [1,1]
        the_map = RampToHillSaddleMap(N)
        CFS = CyclicFeedbackSystem(N,L,Delta,theta,gamma)

        #bifurcations = CFS.get_bifurcations(eps_func)[0]
        bifurcations,eps_func = CFS.get_bifurcations()
        assert(len(bifurcations[0]) == 1)
        for bifurcation in bifurcations[0]:
            hill_sys_parameter, x_hill = the_map.cyclic_feedback_system_map(CFS,bifurcation,eps_func,0)
        assert(np.array_equal(CFS.Delta, hill_sys_parameter.Delta))
        assert(np.array_equal(CFS.theta, hill_sys_parameter.theta))
        assert(np.array_equal(hill_sys_parameter.sign,np.array([[0,-1],[-1,0]])))
        assert(np.allclose(hill_sys_parameter.L[0,1],.6560,rtol = 1e-2))
        assert(np.allclose(hill_sys_parameter.L[1,0],.5981,rtol = 1e-2))
        assert(np.allclose(hill_sys_parameter.n[0,1],12.1399,rtol = 1e-2))
        assert(np.allclose(hill_sys_parameter.n[1,0],10.2267,rtol = 1e-2))
        assert(np.allclose(x_hill,np.array([[.7958],[1.5098]]),rtol=1e-2))

        hill_saddles = the_map.map_all_saddles(CFS,eps_func)
        assert(len(hill_saddles) == 1)
        assert(hill_saddles[0][0] == hill_sys_parameter)
        assert(hill_sys_parameter != HillSystemParameter(N,[[0,1],[1,0]],L,Delta,theta,[[0,1],[1,0]],gamma))
        assert(np.array_equal(hill_saddles[0][1], x_hill))

        gamma = [1.1,0.9]
        CFS = CyclicFeedbackSystem(N,L,Delta,theta,gamma)
        crossings = CFS.border_crossings(eps_func)[0]
        bifurcations = CFS.get_bifurcations(eps_func)[0]
        assert(len(bifurcations[0]) == 1)
        assert(np.allclose(bifurcations[0][0][0], .45622,rtol=1e-4))
        assert(np.allclose(bifurcations[0][0][1],np.array([[.54377],[1.66667]]),rtol = 1e-4))
        hill_sys_parameter, x_hill = the_map.cyclic_feedback_system_map(CFS,bifurcations[0][0],eps_func,0)
        assert(np.allclose(hill_sys_parameter.L[0,1], .77468,rtol = 1e-4))
        assert(np.allclose(hill_sys_parameter.L[1,0],.5922,rtol = 1e-4))
        assert(np.allclose(x_hill,np.array([[.82535],[1.58024]])))



    

    def positive_toggle(self):
        #tests assume these parameter values
        N = DSGRN.Network("X0 : X1 \n X1 : X0")
        L = np.array([[0,1],[1,0]])
        Delta = np.array([[0,1],[1,0]])
        theta = np.array([[0,1.5],[1.5,0]])
        gamma = np.array([1,1])
        return N,L,Delta,theta,gamma

    def neg_edge_toggle(self):
        #tests don't assume these parameter values
        N = DSGRN.Network("X0 : ~X1 \n X1 : ~X0")
        L = np.array([[0,.5],[.5,0]])
        Delta = np.array([[0,1],[1,0]])
        theta = np.array([[0,1.3],[1,0]])
        gamma = np.array([1,1])
        return N,L,Delta,theta,gamma

    def two_independent_toggles(self):
        #tests assume these parameter values
        N = DSGRN.Network("X0 : ~X1 \n X1 : ~X0 \n X2 : ~X3 \n X3 : ~X2")
        L = np.array([[0,.5,0,0],[.5,0,0,0],[0,0,0,.5],[0,0,.5,0]])
        Delta = np.array([[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]],dtype='float')
        theta = np.array([[0,1.3,0,0],[1,0,0,0],[0,0,0,1.3],[0,0,1,0]])
        gamma = np.array([[1],[1],[1],[1]])
        return N,L,Delta,theta,gamma

    def toggle_plus_multiplicative(self):
        N = DSGRN.Network("X0 : (~X1)(X0) \n X1:(~X0)(X1)")
        L = np.array([[1,.5],[.5,1]])
        Delta = np.array([[2.1,1],[1,2.1]],dtype='float')
        theta = np.array([[2.1,1.3],[1,2.1]])
        gamma = np.array([[1],[1]])
        return N,L,Delta,theta,gamma
        return N,L,Delta,theta,gamma

    def toggle_plus_additive(self):
        N = DSGRN.Network("X0 : (~X1)+(X0) \n X1:(~X0)+(X1)")
        L = np.array([[.25,.25],[.25,.25]])
        Delta = np.array([[2.1,1],[1,2.1]],dtype='float')
        theta = np.array([[2.1,1.3],[1,2.1]])
        gamma = np.array([[1],[1]])
        return N,L,Delta,theta,gamma

    def almost_two_independent_toggles(self):
        N = DSGRN.Network("X0 : ~X1 \n X1 : ~X0 \n X2 : (X0)(~X3) \n X3 : ~X2")
        L = np.array([[0,.5,0,.0],[.5,0,0,0],[1,0,0,.5],[0,0,.5,0]])
        Delta = np.array([[0,1,0,0],[1,0,0,0],[1,0,0,1],[0,0,1,0]],dtype='float')
        theta = np.array([[0,1.3,0,0],[1,0,0,0],[2,0,0,1.3],[0,0,1.1,0]])
        gamma = np.array([[1],[1],[1],[1]])
        return N,L,Delta,theta,gamma