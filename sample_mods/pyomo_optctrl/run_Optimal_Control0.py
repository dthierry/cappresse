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
from Optimal_Control import m

# Discretize model using Backward Finite Difference method
# discretizer = TransformationFactory('dae.finite_difference')
# discretizer.apply_to(m,nfe=20,scheme='BACKWARD')

# Discretize model using Orthogonal Collocation
discretizer = TransformationFactory('dae.collocation')
discretizer.apply_to(m,nfe=20,ncp=3,scheme='LAGRANGE-RADAU')
discretizer.reduce_collocation_points(m,var=m.u,ncp=1,contset=m.t)

solver=SolverFactory('ipopt')

results = solver.solve(m,tee=True)

x1 = []
x2 = []
u = []
t=[]
