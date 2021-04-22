import DSGRN
import numpy as np
from ramp_systems.ramp_function import *

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
        R_out = R.evaluate(x)
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