import numpy as np
from ramp_to_hill.hill_system import *
from ramp_systems.ramp_system import RampSystem
import DSGRN

def test_num_unique_vectors():
    vectors = [np.array([1,2,3]),np.array([1,1,1]),np.array([1,2,3])]
    assert(num_unique_vectors(vectors) == 2)
    vectors = [np.array([1,1,1]),np.array([1,2,3]),np.array([1,2,3])]
    assert(num_unique_vectors(vectors) == 2)

def test_find_equilibria():
    net = DSGRN.Network("X0:~X1 \n X1:~X0")
    L = [[0,1],[1,0]]
    Delta = [[0,1],[1,0]]
    theta = [[0,1.5],[1.5,0]]
    gamma = [1,1]
    RS = RampSystem(net,L,Delta,theta,gamma)
    n = make_hill_coefficient_array(net,50)
    sign = make_sign_from_network(net)
    HS = HillSystemParameter(net,sign,L,Delta,theta,n,gamma)
    ## test finding a single equilibrium
    eq = find_equilibrium([1,2],HS,100,tol = 1e-3)
    assert(eq.shape == (2,))
    #large hill coefficient so hill equilibrium should be close to DSGRN equilibrium
    assert(np.allclose(eq,np.array([1,2]),atol = 1e-1))
    ## test finding equilibria from DSGRN fixed points. 
    FPs = [(0,1),(1,0)]
    eq = find_hill_equilibria_from_FPs(FPs,HS,RS,100,tol = 1e-3)
    assert(len(eq) == 2)
    assert(np.allclose(eq[0],np.array([1,2]),atol = 1e-1))
    assert(np.allclose(eq[1],np.array([2,1]),atol = 1e-1))

