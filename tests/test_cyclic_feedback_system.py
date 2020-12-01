
import DSGRN
from ramp_systems.cyclic_feedback_system import *



class TestCyclicFeedbackSystem:

    def test_cfs_sign(self):
        pos_cfs = CyclicFeedbackSystem(*self.positive_toggle())
        assert(pos_cfs.cfs_sign == 1)
        neg_cfs = CyclicFeedbackSystem(*self.negative_toggle())
        assert(neg_cfs.cfs_sign == -1)
        try:
            not_cfs = CyclicFeedbackSystem(*self.toggle_plus())
        except ValueError:
            assert(True)
        else:
            assert(False)

    def positive_toggle(self):
        N = DSGRN.Network("X0 : X1 \n X1 : X0")
        L = np.array([[0,1],[1,0]])
        Delta = np.array([[0,1],[1,0]])
        theta = np.array([[0,1],[1,0]])
        gamma = np.array([1,1])
        return N,L,Delta,theta,gamma

    def negative_toggle(self):
        N = DSGRN.Network("X0 : X1 \n X1 : (~X0)")
        L = np.array([[0,1],[1,0]])
        Delta = np.array([[0,1],[1,0]])
        theta = np.array([[0,1],[1,0]])
        gamma = np.array([1,1])
        return N,L,Delta,theta,gamma
    
    def toggle_plus(self):
        N = DSGRN.Network("X0 : (X1)(X0) \n X1 : (~X0)")
        L = np.array([[0,1],[1,0]])
        Delta = np.array([[0,1],[1,0]])
        theta = np.array([[0,1],[1,0]])
        gamma = np.array([1,1])
        return N,L,Delta,theta,gamma