#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________

from pyomo.environ import *
from pyomo.dae import *
from Optimal_Control2 import m

# Discretize model using Backward Finite Difference method
# discretizer = TransformationFactory('dae.finite_difference')
# discretizer.apply_to(m,nfe=20,scheme='BACKWARD')

# Discretize model using Orthogonal Collocation

m.pprint()
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m, nfe=1, ncp=3, scheme='LAGRANGE-RADAU')
discretizer.reduce_collocation_points(m, var=m.u, ncp=1, contset=m.t)

solver = SolverFactory('ipopt', tee=False)

results = solver.solve(m, tee=False)

states = ['x1', 'x2']

for x in states:
    xv = getattr(m, x)
    idx = xv.keys()
    for k in idx:
        kr = k[1:]
        xv[0, kr].set_value(2069)

m.x1.pprint()
m.x2.pprint()