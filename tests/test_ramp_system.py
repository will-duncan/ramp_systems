import DSGRN
import numpy as np
from ramp_systems.ramp_function import *
from ramp_systems.ramp_system import *

def test_power_set():
    my_list = [0,1,2]
    my_list_power_set = {(),(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)}
    power_set_out = {s for s in power_set(my_list)}
    assert(my_list_power_set.issubset(power_set_out))
    assert(my_list_power_set.issuperset(power_set_out))




class TestRampModel:

    def test_func_array(self):
        N, L, Delta, theta, gamma = self.toggle_switch_parameters()
        RS = RampSystem(N,L,Delta,theta,gamma)
        func_array = RS.func_array
        
        sign_list = [-1,-1]
        index_list = [[0,1],[1,0]]
        for k in range(len(sign_list)):
            index = index_list[k]
            i = index[0]
            j = index[1]
            R = RampFunction(sign_list[k],L[i,j],Delta[i,j],theta[i,j])
            assert(R(theta[i,j]-1) == func_array(theta[i,:]-1)[i,j])
            assert(np.isnan(func_array(theta[i,:])[i,j]))
            assert(R(theta[i,j]+1) == func_array(theta[i,:]+1)[i,j])

    def test_call(self):
        """__call__ depends on 'R' attribute so this implicitly tests _set_R"""

        #simplest test: toggle switch
        N, L, Delta, theta, gamma = self.toggle_switch_parameters()
        RS = RampSystem(N,L,Delta,theta,gamma)
        eps = np.array([[0,.5],[.5,0]])
        R01 = RampFunction(sign=-1,L=L[0,1],Delta=Delta[0,1],theta=theta[0,1])
        R10 = RampFunction(sign=-1,L=L[1,0],Delta=Delta[1,0],theta=theta[1,0])
        x_list = [[.1,1],[.5,1],[1,1],[1.5,1],[2,1],[1,2.5]]
        for x in x_list:
            x = np.array(x)
            R_vec = np.array([R01(x[1],eps[0,1]),R10(x[0],eps[1,0])])
            expected = -gamma*x + R_vec
            assert(np.array_equal(expected, RS(x,eps)))
        
        L = np.array([[0,1],[4,0]])
        Delta = np.array([[0,1],[1,0]])
        gamma = np.array([1,1])
        theta = np.array([[0,4],[.5,0]])
        RS = RampSystem(N,L,Delta,theta,gamma)
        assert(np.array_equal(RS.R([5,5]),np.array([1,4])))
        assert(np.array_equal(RS.R([0,0]),np.array([2,5])))

        #TO DO: test with interaction between edges

    def test_get_W(self):
        N, L, Delta, theta, gamma = self.toggle_switch_parameters()
        RS = RampSystem(N,L,Delta,theta,gamma)
        W = RS.get_W()
        assert([0,L[0,1],L[0,1]+Delta[0,1],np.inf] == W[0])
        assert([0,L[1,0],L[1,0]+Delta[1,0],np.inf] == W[1])

        L = np.array([[0,1],[2,0]])
        Delta = np.array([[0,1],[1,0]])
        gamma = np.array([1,1])
        theta = np.array([[0,4],[.5,0]])
        RS = RampSystem(N,L,Delta,theta,gamma)
        W = RS.get_W()
        assert([0,L[0,1],L[0,1]+Delta[0,1],np.inf] == W[0])
        assert([0,L[1,0],L[1,0]+Delta[1,0],np.inf] == W[1])

    def test_get_eps_jp(self):
        #toggle switch test
        N, L, Delta, theta, gamma = self.toggle_switch_parameters()
        L = np.array([[0,1],[2,0]])
        Delta = np.array([[0,1],[1,0]])
        gamma = np.array([1,1])
        theta = np.array([[0,.5],[.4,0]])
        RS = RampSystem(N,L,Delta,theta,gamma)
        assert(RS.is_regular() )
        W = RS.get_W()
        j = 1
        B = RS.get_B(W)
        eps_j1_out = RS.get_eps_jp(j,W[j],B[j],1)
        eps_j1_expected = np.array([[0,1.5],[0,0]])
        assert(np.array_equal(eps_j1_out,eps_j1_expected))
        zero = np.zeros([2,2])
        assert(np.array_equal(RS.get_eps_jp(j,W[j],B[j],2),zero))
        assert(np.array_equal(RS.get_eps_jp(j,W[j],B[j],3),zero))

        #TO DO: test with interaction between edges. 

    def test_optimal_eps(self):
        N, L, Delta, theta, gamma = self.toggle_switch_parameters()
        L = np.array([[0,1],[2,0]])
        Delta = np.array([[0,1],[1,0]])
        gamma = np.array([1,1])
        theta = np.array([[0,1],[.5,0]])
        
        RS = RampSystem(N,L,Delta,theta,gamma)
        assert(RS.is_regular() )
        expected = np.array([[0,1],[.5,0]])
        assert(np.array_equal(RS.optimal_eps(),expected ))
        assert(~RS.is_regular(expected))


    def test_is_regular(self):
        """Inital tests of is_regular. Additional testing in test_optimal_eps as a sanity check."""
        N, L, Delta, theta, gamma = self.toggle_switch_parameters()
        RS = RampSystem(N,L,Delta,theta,gamma)
        RS.L = np.array([[0,1],[1,0]])
        RS.Delta = np.array([[0,1],[1,0]])
        RS.theta = np.array([[0,1],[1,0]])
        RS.gamma = np.array([1,1])
        assert(~RS.is_regular() )
        RS.theta = np.array([[0,.8],[1.2,0]])
        assert(RS.is_regular() ) 
        assert(RS.is_regular)
        eps = np.array([[0,.2],[0,0]])
        assert(~RS.is_regular(eps) )
        eps = np.array([[0,0],[.2,0]])
        assert(~RS.is_regular(eps) )
        eps = np.array([[0,.1],[.1,0]])
        assert(RS.is_regular(eps))

        

    def toggle_switch_parameters(self):
        N = DSGRN.Network("X0 : (~X1)\n X1 : (~X0)")    
        L = np.array([[0,1],[2,0]])
        Delta = np.array([[0,1],[2,0]])
        theta = np.array([[0,1],[2,0]])
        gamma = np.array([1,2])
        return N, L, Delta, theta, gamma