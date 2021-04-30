
import numpy as np
import itertools
import random

def cell_from_coordinates(RS,*coords):
    """
    Create a Cell object from state transition graph coordinates.
    :param RS: RampSystem object
    :param *coords: jth argument is an integer corresponding to the jth coordinate
    """
    theta = RS.theta
    theta_orders = RS.all_theta_orders()
    projections = []
    for j,coord in enumerate(coords):
        theta_j = theta_orders[j]
        if coord == 0:
            projections.append((-np.inf,theta_j[0]))
        elif coord < len(theta_j):
            projections.append((theta_j[coord-1],theta_j[coord]))
        else: #coord == len(theta_j)
            projections.append((theta_j[-1],np.inf))
    return Cell(theta,*projections)


class Cell:
    
    def __init__(self,theta,*projections):
        """
        Create a Cell object.

        :param theta: matrix of threshold values
        :param *projections: jth argument is either a pair of indices i_1, i_2 or single
        index i which correspond to target node(s) of node j. The jth projection of the
        cell is given by (theta[i_1,j],theta[i_2,j]) or {theta[i,j]}, respectively. 
        """
        self.theta = np.array(theta,dtype = 'float')
        self.pi = [(projections[j],) if not hasattr(projections[j],'__iter__') else tuple(projections[j]) for j in range(len(projections))]
        self._set_rho()

    def regular_directions(self):
        #only compute this once
        try:
            reg_dir = self._reg_dir
        except AttributeError:
            reg_dir = set(j for j in range(len(self.pi)) if len(self.pi[j]) == 2)
            self._reg_dir = reg_dir
        return reg_dir

    def singular_directions(self):
        #only compute this once
        try: 
            sin_dir = self._sin_dir
        except AttributeError:
            sin_dir = set(j for j in range(len(self.pi)) if len(self.pi[j]) == 1)
            self._sin_dir = sin_dir
        return sin_dir

    def sample(self):
        """
        Get a point contained in the cell where for each regular direction j,
        x[j] is chosen uniformly from the interval (theta[a_j,j],theta[b_j,j]) where
        theta[b_j,j] = 2*theta[a_j,j] when b_j = inf 
        """
        x = np.zeros(len(self.pi))
        rho = self.rho
        theta = self.theta
        pi = self.pi
        for j in self.singular_directions():
            x[j] = theta[rho[j],j]
        for j in self.regular_directions():
            if pi[j][0] == -np.inf:
                left = 0
            else:
                left = theta[pi[j][0],j]
            if pi[j][1] == np.inf:
                right = 2*left
            else:
                right = theta[pi[j][1],j]
            x[j] = random.uniform(left,right)
        return x

    def __call__(self,j):
        return self.pi[j]

    def _set_rho(self):
        rho = [[] for i in range(len(self.pi))]
        for j in self.regular_directions():
            rho[j] = j
        for j in self.singular_directions():
            rho[j] = self.pi[j][0]
        self.rho = rho

    def rho_plus(self,j):
        """
        Compute the index i where theta[rho[j],j]<theta[i,j] are consecutive
        thresholds.

        :param j: singular direction of the cell
        """
        rho = self.rho
        theta = self.theta
        if theta[rho[j],j] == theta[:,j].max():
            return np.inf
        else:
            difference_array = theta[:,j] - theta[rho[j],j]
            difference_array[difference_array <= 0] = np.inf
            return np.argmin(difference_array)
    
    def rho_minus(self,j):
        """
        Compute the index i where theta[i,j]<theta[rho[j],j] are consecutive thresholds.

        :param j: singular direction of the cell
        """
        rho = self.rho
        theta = self.theta
        if theta[rho[j],j] == theta[:,j].min():
            return -np.inf
        else:
            difference_array = theta[:,j] - theta[rho[j],j]
            difference_array[difference_array >= 0] = -np.inf
            return np.argmax(difference_array)
    
    def theta_rho_minus(self,j):
        rho_minus = self.rho_minus(j)
        if rho_minus == -np.inf:
            return 0
        else: 
            return self.theta[rho_minus,j]

    def theta_rho_plus(self,j):
        rho_plus = self.rho_plus(j)
        if rho_plus == np.inf:
            return np.inf
        else:
            return self.theta[rho_plus,j]

    def top_cells(self):
        top_cell_list = []
        sd = self.singular_directions()
        for plus_or_minus in itertools.product([-1,1],repeat = len(sd)):
            pi = self.pi.copy()
            for j in sd:
                if plus_or_minus == -1:
                    pi[j] = (self.rho_minus(j),self.pi[j][0])
                else: 
                    pi[j] = (self.pi[j][0],self.rho_plus(j))
            top_cell_list.append(Cell(self.theta,*pi))
        return top_cell_list

    def __eq__(self,other):
        if not np.array_equal(self.theta,other.theta):
            return False
        if not self.pi == other.pi:
            return False
        return True

    def __repr__(self):
        string = 'Cell(theta = {}'.format(self.theta)
        for pi_j in self.pi:
            string += ',' + str(pi_j)
        string += ')'
        return string

    def is_regular(self):
        if len(self.singular_directions()) == 0:
            return True
        else:
            return False
