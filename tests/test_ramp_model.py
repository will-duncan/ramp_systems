import DSGRN
from ramp_systems.ramp_model import *

def test_power_set():
    my_list = [0,1,2]
    my_list_power_set = {(),(0,),(1,),(2,),(0,1),(0,2),(1,2),(0,1,2)}
    power_set_out = {s for s in power_set(my_list)}
    # print(power_set_out)
    # print(my_list_power)
    assert(my_list_power_set.issubset(power_set_out))
    assert(my_list_power_set.issuperset(power_set_out))

class TestRampFunction:
    
    def test_call(self):
        
        L = 1
        Delta = 1
        theta = 1
        sign = 1
        R = RampFunction(sign,L,Delta,theta)
        #eps == 0
        assert(R(.5,0) == L)
        assert(np.isnan(R(1,0)))
        assert(R(1.5,0) == L + Delta)
        #handles no eps argument
        assert(R(.5) == R(.5,0))
        assert(np.isnan(R(1)))
        assert(R(1.5) == R(1.5,0))
        #eps != 0
        eps = .5
        m = Delta/(2*eps)
        R_at_theta = L + Delta/2
        assert(R(theta-2*eps,eps) == L)
        assert(R(theta-eps,eps) == L)
        assert(R(theta-eps/2,eps) ==  R_at_theta + sign*m*(-eps/2))
        assert(R(theta,eps) == R_at_theta)
        assert(R(theta+eps/2,eps) == R_at_theta + sign*m*(eps/2))
        assert(R(theta+eps,eps) == L+Delta)
        assert(R(theta+2*eps,eps) == L+Delta)
        #handles vectors
        x = np.array([.5,1,1.5])
        R_at_x = np.array([L,np.nan,L+Delta])
        R_out = R(x)
        for i in range(len(x)):
            assert( (R_at_x[i] == R_out[i]) \
                or (np.isnan(R_at_x[i]) and np.isnan(R_out[i])) )
        #changing parameters after initialization
        R.sign = -1
        assert(R(.5) == L+Delta)
        assert(np.isnan(R(1)))
        assert(R(1.5) == L)

    def test_dx(self):
        L = 1
        Delta = 1
        theta = 1
        sign = 1
        R = RampFunction(sign,L,Delta,theta)
        #handles eps = 0
        assert(R.dx(theta-.5,0) == 0)
        assert(np.isnan(R.dx(theta,0)))
        assert(R.dx(theta+.5,0) == 0)
        #handles no eps argument
        assert(R.dx(theta-.5,0) == R.dx(theta-.5))
        assert(np.isnan(R.dx(theta)))
        assert(R.dx(theta+.5,0) == R.dx(theta+.5))
        #eps != 0
        eps = .5
        m = Delta/(2*eps)
        assert(R.dx(theta-2*eps,eps) == 0)
        assert(R.dx(theta,eps) == sign*m)
        assert(R.dx(theta+2*eps,eps) == 0)
        assert(np.isnan(R.dx(theta+eps,eps)))
        #handles vectors
        x = [theta-eps,theta]
        dx_at_x = np.array([0,np.nan])
        dx_out = R.dx(x)
        for i in range(len(x)):
            assert( (dx_at_x[i] == dx_out[i]) \
                or (np.isnan(dx_at_x[i]) and np.isnan(dx_out[i])) )
        #changing parameters after initialization
        R.sign = -sign
        assert(R.dx(theta,eps) == np.array(-sign*m))

    def test_eq(self):
        L = 1
        Delta = 1
        theta = 1
        sign = 1
        R1 = RampFunction(sign,L,Delta,theta)
        R2 = RampFunction(sign,L,Delta,theta)
        assert(R1 == R2)
        R2.sign = -R1.sign
        assert(R1 != R2)
        assert(R1 != 0)


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