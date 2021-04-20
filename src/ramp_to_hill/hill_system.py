import numpy as np
from scipy.integrate import solve_ivp


def find_equilibrium(x0,HS,max_time,tol = 1e-3):
    """
    Simulate the ODE to equilibrium starting from x0
    Input: 
        x0 - initial condition
        HS - HillSystemParameter object
        max_time - run the ode from time points [0,max_time]. If the solver reaches
                   max_time before finding an equilibrium, then report that an equilibrium 
                   was not found
    Output:
        x - value of the equilibrium, if found within max_time. If not found, returns -1
    """
    def ode(t,y,HS = HS):
        rhs = -HS.gamma*y + HS.lambda_value(y)
        return rhs
    def at_equilibrium(t,y,HS = HS,tol = tol):
        val = np.linalg.norm(ode(t,y)) - tol
        if val < 0:
            return 0
        else:
            return val
    at_equilibrium.terminal = True
    integration_interval = (0,max_time)
    sol = solve_ivp(ode,integration_interval,x0,method = 'BDF',events = at_equilibrium)
    if sol.status == 1: #at_equilibrium triggered stopping integration
        return sol.y[:,-1]
    else: 
        return -1

def find_hill_equilibria_from_FPs(FPs,HS,RS,max_time,tol = 1e-3):
    """
    Use DSGRN equilibria as initial conditions for finding Hill equilibria. 
    Input:  
        FPs - list of fixed point coordinates computed by DSGRN
        HS - HillSystemParameter object
        RS - RampSystem object
        max_time - maximum time to run the ODE for each equilibrium search attempt. 
    Output:
        eq - list of Nx1 numpy arrays. An entry is -1 if find_equilibrium didn't find an
             equilibrium within max_time.  len(eq) == len(FPs)
    """
    reg_DSGRN_equilibria = RS.reg_equilibria_from_FPs(FPs)
    hill_eq = [find_equilibrium(x0.reshape([x0.shape[0]]),HS,max_time,tol = tol) for x0 in reg_DSGRN_equilibria]
    return hill_eq

def num_unique_vectors(vectors,tol = 1e-3):
    """
    Given a list of vectors, count the number which are unique up to some tolerance
    """
    repeat_indices = []
    num_unique = 0
    for j, vec0 in enumerate(vectors):
        if j in repeat_indices:
            continue
        num_unique += 1
        for i, vec1 in enumerate(vectors[j+1:]):
            i = i+j+1
            if i in repeat_indices:
                continue
            if np.allclose(vec0,vec1,rtol = tol):
                repeat_indices.append(i)
    return num_unique

def hill_value(x,hill_parameter):
    sign = hill_parameter.sign
    theta = hill_parameter.theta
    Delta = hill_parameter.Delta
    L = hill_parameter.L
    n = hill_parameter.n
    if sign == 1:
        return L + Delta/((theta/x)**n + 1)
    if sign == -1:
        return L + Delta/((x/theta)**n + 1)

def hill_second_derivative_root(*args):
    if len(args) == 1:
        hill_parameter = args[0]
        theta = hill_parameter.theta
        n = hill_parameter.n
    elif len(args) == 2:
        theta = args[0]
        n = args[1]
    else:
        raise TypeError('hill_second_derivative_root() takes 1 or 2 position arguments\
            but {} were given.'.format(len(args)))
    return theta*((n-1)/(n+1))**(1/n)

def hill_derivative_magnitude(x,*args):
    if len(args) == 1:
        hill_parameter = args[0]
        sign = hill_parameter.sign
        Delta = hill_parameter.Delta
        theta = hill_parameter.theta
        n = hill_parameter.n
    elif len(args) == 4:
        sign = args[0]
        Delta = args[1]
        theta = args[2]
        n = args[3]
    else: 
        raise TypeError('hill_derivative() takes 1 or 4 positiional arguments\
            but {} were given.'.format(len(args)))
    if n == np.inf:
        if theta == x:
            return np.inf
        else: 
            return 0
    return Delta*n/(theta*(theta/x)**(n-1) + 2*x + x*(x/theta)**n)

def make_hill_coefficient_array(Network,n):
    """
    Make a hill coefficient array consistent with the network topology with each
    hill coefficient equal to n
    Input:
        Network - DSGRN network object
        n - float or integer greater than 1
    Output:
        numpy array with entry [i,j] equal to n if j->i is an edge and 0 otherwise
    """
    N = Network.size()
    hill_coefficients = np.zeros([N,N])
    for j in range(N):
        for i in Network.outputs(j):
            hill_coefficients[i,j] = n
    return hill_coefficients

def make_sign_from_network(Network):
    """
    Make an NxN numpy array describing the interaction sign between edges
    Input:
        Network - DSGRN network object
    Output:
        numpy array with 1 if j->i, -1 if j-|i, and 0 otherwise. 
    """
    N = Network.size()
    sign = np.zeros([N,N])
    for j in range(N):
        for i in Network.outputs(j):
            sign[i,j] = 1 if Network.interaction(j,i) else -1
    return sign

class HillParameter:

    def __init__(self,sign,L,Delta,theta,n):
        """
        Input:
            sign - either 1 or -1
            L,Delta,theta,n - parameters for a hill function
        """
        self.sign = sign
        self.L = L
        self.Delta = Delta
        self.theta = theta
        self.n = n
    
    def __repr__(self):
        sign = self.sign
        L = self.L
        Delta = self.Delta
        theta = self.theta
        n = self.n
        return 'HillParameter({},{},{},{},{})'.format(sign,L,Delta,theta,n)

    def func_value(self,x):
        return hill_value(x,self)

    def dx_value(self,x):
        return self.sign*hill_derivative_magnitude(x,self)


class HillSystemParameter:

    def __init__(self,Network,sign,L,Delta,theta,n,gamma):
        """
        Input:
            gamma - length N lists
            sign,L,Delta,theta,n - NxN arrays
        """
        self.Network = Network
        N = Network.size()
        self.sign = np.array(sign)
        self.L = np.array(L)
        self.Delta = np.array(Delta)
        self.theta = np.array(theta)
        self.n = np.array(n)
        self.gamma = np.array(gamma).reshape([N])
    
    def __eq__(self,other):
        if isinstance(other,HillSystemParameter):
            return np.array_equal(self.sign,other.sign) and np.array_equal(self.L, other.L) \
                and np.array_equal(self.Delta, other.Delta) and np.array_equal(self.theta,other.theta) \
                and np.array_equal(self.n, other.n) and np.array_equal(self.gamma,other.gamma)
        else: 
            return False
    

    def hill_parameter(self,i,j):
        return HillParameter(self.sign[i,j],self.L[i,j],self.Delta[i,j],self.theta[i,j],self.n[i,j])


    def lambda_value(self,x):
        Network = self.Network
        N = Network.size()
        val = np.zeros([N])
        for i in range(N):
            cur_prod = 1
            for source_set in Network.logic(i):
                cur_sum = 0
                for j in source_set:
                    cur_param = self.hill_parameter(i,j)
                    cur_sum += cur_param.func_value(x[j])
                cur_prod *= cur_sum
            val[i] = cur_prod
        return val

    def is_equilibrium(self,x,tol = 1e-4):
        N = self.Network.size()
        x = np.array(x).reshape([N])
        return np.allclose(self.lambda_value(x)-self.gamma*x,np.zeros([N]),atol=tol)

    def Jacobian(self,x):
        N = self.Network.size()
        J = np.diag(-self.gamma)
        for i in range(N):
            for j in self.Network.inputs(i):
                cur_prod = 1
                for source_set in self.Network.logic(i):
                    cur_sum = 0
                    if j in source_set:
                        cur_sum = self.hill_parameter(i,j).dx_value(x[j])
                    else:
                        for k in source_set:
                            cur_sum += self.hill_parameter(i,k).func_value(x[k])
                cur_prod *= cur_sum
            J[i,j] += cur_prod
        return j


    def is_saddle(self,x,tol = 1e-4):
        N = self.Network.size()
        x = np.array(x).reshape([N,1])
        if not self.is_equilibrium(x,tol=tol):
            return False
        J = self.Jacobian(x)
        if np.linalg.matrix_rank(J) == N:
            return False
        return True


