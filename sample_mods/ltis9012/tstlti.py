from __future__ import division, print_function
from pyomo.environ import *
from ltis9012 import ltiis9016
from pyomo.opt import SolverFactory

af = ltiis9016(1, 3)

af.pprint()
af.u1[0] = 1.22
ip = SolverFactory('ipopt')
ip.solve(af, tee=True)

af_steady = ltiis9016(1, 3, steady=True)
af_steady.pprint()
af_steady.u1.pprint()