"""
CyclicFeedbackSystem class and methods. This is a child class of the RampSystem
class (see ramp_system.py).

    Author: William Duncan
"""
from ramp_systems.ramp_system import *

class CyclicFeedbackSystem(RampSystem):

    def __init__(self,Network,L,Delta,theta,gamma):
        """Requires that the network is a cyclic feedback network."""
        RampSystem.__init__(self,Network,L,Delta,theta,gamma)
        self._set_cfs_sign()

    def _set_cfs_sign(self):
        """
        Computes the sign of the cycle and sets it as the attribute 'cfs_sign'. 
        Raises an exception if the network is not a cyclic feedback network. 
        """
        Network = self.Network
        node = 0
        next_node = -1
        cfs_sign = 1
        while(next_node != 0):
            output = Network.outputs(node)
            if len(output) != 1:
                raise ValueError("CyclicFeedbackSystem requires Network is a cyclic\
                     feedback network but at least one node had number of outputs\
                         different from 1.")
            next_node = output[0]
            if not Network.interaction(node,next_node): #node represses next_node
                cfs_sign *= -1
            node = next_node
        self.cfs_sign = cfs_sign

    