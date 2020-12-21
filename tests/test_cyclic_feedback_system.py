
import DSGRN
from ramp_systems.cyclic_feedback_system import CyclicFeedbackSystem
import sympy
import numpy as np


class TestCyclicFeedbackSystem:

    def test_set_attributes(self):
        pos_cfs = CyclicFeedbackSystem(*self.positive_toggle())
        assert(pos_cfs.cfs_sign == 1)
        assert(pos_cfs.rho==[1,0])
        assert(pos_cfs.rho_inv == [1,0])
        assert(pos_cfs.edge_sign == [1,1])

        neg_cfs = CyclicFeedbackSystem(*self.negative_toggle())
        assert(neg_cfs.cfs_sign == -1)
        assert(neg_cfs.rho==[1,0])
        assert(neg_cfs.rho_inv == [1,0])
        assert(neg_cfs.edge_sign == [1,-1])

        three_node_cfs = CyclicFeedbackSystem(*self.three_node_network())
        assert(three_node_cfs.cfs_sign == -1)
        assert(three_node_cfs.rho == [1,2,0])
        assert(three_node_cfs.rho_inv == [2,0,1])
        assert(three_node_cfs.edge_sign == [1,1,-1])

        try:
            not_cfs = CyclicFeedbackSystem(*self.toggle_plus())
        except ValueError:
            assert(True)
        else:
            assert(False)

    def test_singular_equilibrium(self):
        N,L,Delta,theta,gamma = self.positive_toggle()
        pos_cfs = CyclicFeedbackSystem(N,L,Delta,theta,gamma)
        x = pos_cfs.singular_equilibrium()
        assert(np.allclose(x(.1),np.array([[1.5],[1.5]])))
        
        N,L,Delta,theta,gamma = self.neg_edge_toggle()
        pos_cfs = CyclicFeedbackSystem(N,L,Delta,theta,gamma)
        U1 = Delta[0,1] + L[0,1]
        U2 = Delta[1,0] + L[1,0]
        theta1 = theta[0,1]
        theta2 = theta[1,0]
        Delta1 = Delta[0,1]
        Delta2 = Delta[1,0]
        
        s = sympy.symbols('r')
        eps_func = sympy.Matrix([[0,1],[1,0]])*s
        s_val = .1
        eps1 = s_val
        eps2 = s_val
        m1 = Delta1/(2*eps1)
        m2 = Delta2/(2*eps2)
        #solution for gamma1 = gamma2 = 1
        x1 = (U1 + (theta1 - eps1 - U2)*m1 - (theta2-eps2)*m1*m2)/(1-m1*m2)
        x2 = (U2 + (theta2 - eps2 - U1)*m2 - (theta1-eps1)*m1*m2)/(1-m1*m2)
        expected = np.array([[x1],[x2]])
        x = pos_cfs.singular_equilibrium(eps_func)
        assert(np.allclose(x(s_val),expected))
    
    def test_in_singular_domain(self):
        pos_cfs = CyclicFeedbackSystem(*self.positive_toggle())
        eps = np.array([[0,.5],[.5,0]])
        
        x = np.array([.5,.5])
        assert(not pos_cfs.in_singular_domain(x,eps))
        assert(not pos_cfs.in_singular_domain(x,eps,0))
        
        x = np.array([1.5,1.5])
        assert(pos_cfs.in_singular_domain(x,eps))
        assert(pos_cfs.in_singular_domain(x,eps,1))

        x = np.array([.5,1.5])
        assert(not pos_cfs.in_singular_domain(x,eps))
        assert(pos_cfs.in_singular_domain(x,eps,0))
        assert(not pos_cfs.in_singular_domain(x,eps,1))


        
    def test_pos_bifurcations(self):
        N,L,Delta,theta,gamma = self.neg_edge_toggle()
        L[0,1] = .5
        Delta[0,1] = 1
        theta[0,1] = 1.3
        L[1,0] = .5
        Delta[1,0] = 1
        theta[1,0] = 1
        gamma = [1,1]
        cfs = CyclicFeedbackSystem(N,L,Delta,theta,gamma)
        s = sympy.symbols('s')
        eps_func = sympy.Matrix([[0,1],[1,0]])*s
        x_eq = cfs.singular_equilibrium(eps_func,lambdify=False)
        zero_crossings= cfs.j_border_crossings(0,x_eq,eps_func)
        assert(len(zero_crossings) == 1)
        for crossing in zero_crossings:
            assert(np.allclose(crossing[0],.3162,rtol = 1e-4))
        crossings,eps_func_out = cfs.border_crossings(eps_func)
        assert(eps_func_out == eps_func)
        assert(crossings[0][0][0] == zero_crossings[0][0])
        assert(len(crossings[1]) == 1)
        for crossing in crossings[1]:
            assert(np.allclose(crossing[0],.67202))
        #get_bifurcations
        bifurcations = cfs.get_bifurcations(eps_func)[0]
        assert(not cfs.in_singular_domain(x_eq.subs(s,.37202),np.array([[0,.37202],[.37202,0]]),1))
        assert(len(bifurcations[0]) == 1)
        for s_val in bifurcations[0]:
            assert(np.allclose(s_val[0],.3162,rtol=1e-4))
        #make sure border_crossings runs on three nodes
        cfs = CyclicFeedbackSystem(*self.three_node_network())
        crossings, eps_func = cfs.border_crossings()
        assert(True)

    def test_neg_bifurcations(self):
        N,L,Delta,theta,gamma = self.negative_toggle()
        L[0,1] = .95
        Delta[0,1] = 1
        theta[0,1] = 1.3
        L[1,0] = .5
        Delta[1,0] = 1
        theta[1,0] = 1
        gamma = [1,1]
        cfs = CyclicFeedbackSystem(N,L,Delta,theta,gamma)
        bifurcations = cfs.neg_loop_bifurcations()
        for j in range(3):
            assert(len(bifurcations[j]) == 0)

        three_node = CyclicFeedbackSystem(*self.three_node_network())
        bifurcations,eps_func_out = three_node.neg_loop_bifurcations()
        for j in range(3):
            assert(len(bifurcations[j]) == 0)
        assert(len(bifurcations[3]) == 1)
        for bif in bifurcations[3]:
            assert(np.allclose(bif[0],.25,rtol=1e-4))
            assert(np.allclose(np.array([[1],[1],[1]]),bif[1],rtol = 1e-4))

        #TO DO: find example of a stability changing
        #border crossing bifurcation in negative cycle


  

    ## CyclicFeedbackSystem parameters ##
    def neg_edge_toggle(self):
        #tests don't assume these parameter values
        N = DSGRN.Network("X0 : ~X1 \n X1 : ~X0")
        L = np.array([[0,.5],[.5,0]])
        Delta = np.array([[0,1],[1,0]])
        theta = np.array([[0,1.3],[1,0]])
        gamma = np.array([1,1])
        return N,L,Delta,theta,gamma
    
    def positive_toggle(self):
        #tests assume these parameter values
        N = DSGRN.Network("X0 : X1 \n X1 : X0")
        L = np.array([[0,1],[1,0]])
        Delta = np.array([[0,1],[1,0]])
        theta = np.array([[0,1.5],[1.5,0]])
        gamma = np.array([1,1])
        return N,L,Delta,theta,gamma

    def three_node_network(self):
        #tests assume these parameter values
        N = DSGRN.Network("X0 : X2 \n X1: X0 \n X2: ~X1")
        L = np.array([[0,0,.5],[0.5,0,0],[0,0.5,0]])
        Delta = np.array([[0,0,1],[1,0,0],[0,1,0]])
        theta = np.array([[0,0,1],[1,0,0],[0,1,0]])
        gamma = np.array([1,1,1])
        return N,L,Delta,theta,gamma  

    def negative_toggle(self):
        #tests don't assume these parameter values
        N = DSGRN.Network("X0 : X1 \n X1 : (~X0)")
        L = np.array([[0,1],[1,0]])
        Delta = np.array([[0,1],[1,0]])
        theta = np.array([[0,1],[1,0]])
        gamma = np.array([1,1])
        return N,L,Delta,theta,gamma
    
    def toggle_plus(self):
        #tests don't assume these parameter values
        N = DSGRN.Network("X0 : (X1)(X0) \n X1 : (~X0)")
        L = np.array([[0,1],[1,0]])
        Delta = np.array([[0,1],[1,0]])
        theta = np.array([[0,1],[1,0]])
        gamma = np.array([1,1])
        return N,L,Delta,theta,gamma