import DSGRN
from ramp_systems.ramp_function import *
from ramp_systems.ramp_model import *

def test_power_set():
    my_list = [0,1,2]
    my_list_power_set = {(),(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)}
    power_set_out = {s for s in power_set(my_list)}
    # print(power_set_out)
    # print(my_list_power)
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

        

    def toggle_switch_parameters(self):

        N = DSGRN.Network("X0 : (~X1)\n X1 : (~X0)")    
        L = np.array([[0,1],[2,0]])
        Delta = np.array([[0,1],[2,0]])
        theta = np.array([[0,1],[2,0]])
        gamma = np.array([1,2])
        return N, L, Delta, theta, gamma