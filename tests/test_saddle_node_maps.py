
from ramp_systems.ramp_function import RampFunction
import numpy as np
from ramp_to_hill.saddle_node_maps import *
from ramp_systems.cyclic_feedback_system import CyclicFeedbackSystem
import sympy
import DSGRN

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
        s = sympy.symbols('s')
        eps_func = sympy.Matrix([[0,1],[1,0]])*s

        bifurcations = CFS.get_bifurcations(eps_func)[0]
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

        hill_systems,x_hill_pts = the_map(CFS,eps_func)
        assert(len(hill_systems) == 1)
        assert(len(x_hill_pts) == 1)
        assert(hill_systems[0] == hill_sys_parameter)
        assert(hill_sys_parameter != HillSystemParameter(N,[[0,1],[1,0]],L,Delta,theta,[[0,1],[1,0]],gamma))
        assert(np.array_equal(x_hill_pts[0], x_hill))

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