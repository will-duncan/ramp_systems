# Dependencies
DSGRN and dependencies, sympy. 

# Installation
From the command line, do 
```bash
. install.sh
```

To make sure the package is running properly, run
```bash
pytest tests
```
from the command line. 

# Documentation

Most functions have a docstring giving details about their inputs and outputs.
Notation is written to be consistent with *Equilibria and their Stability in Networks
with Steep Sigmoidal Functions* (W. Duncan and T. Gedeon, in preparation.  

# Modules and Functionality

## ramp_systems

### ramp_systems.py

Implements the RampSystem class. Given $L, U, \Delta$, and $\theta$, the optimal choice
of $\varepsilon$ to minimize ramp function slopes while maintianing the equilibria 
predicted by DSGRN can be found with
```python
RampSystem.optimal_eps()
``` 
The right hand side of the ODE is computed by calling an instance of the class:
```python
RampSystem(x[,eps = None])
```

#### To be implemented:

- Finding all saddle node bifurcations that occur as a function of $\varepsilon$
assuming singular domains don't overlap. 
- Optimizing $\theta$ and $\varepsilon$ to minimize ramp function slopes while 
maintaining DSGRN equilibria. 

### cyclic_feedback_system.py

Implements the CyclicFeedbackSystem class, a sub-class of the RampSystem class designed
for cyclic feedback systems. Main functionality is through 
```python
CyclicFeedbackSystem.get_bifurcations([eps_func = None])
```
which computes all bifurcations as $\varepsilon$ is varied according to the parameterization
provided by ```eps_func```, which defaults to a choice that keeps all slopes the same.
Currently bifurcations in negative cyclic feedback systems is only implemented for 
systems with identical degradation rates for each variable $\gamma_i = \gamma$. 

## ramp_to_hill

### saddle_node_maps.py

Implements the RampToHillSaddleMap class which maps saddle nodes in ramp systems
to saddle nodes in hill systems. Currently only implemented for cyclic feedback
systems, for which the map is a bijection. 

#### To be implemented:

- Ramp system to Hill system map for a general network. This will not be a bijective
map. 