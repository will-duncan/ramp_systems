
from ramp_systems.ramp_function import RampFunction
import numpy as np
from ramp_to_hill.saddle_node_maps import RampToHillFunctionMap

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
    

