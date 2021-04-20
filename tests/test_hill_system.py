import numpy as np
from ramp_to_hill.hill_system import *

def test_num_unique_vectors():
    vectors = [np.array([1,2,3]),np.array([1,1,1,]),np.array([1,2,3])]
    assert(num_unique_vectors(vectors) == 2)