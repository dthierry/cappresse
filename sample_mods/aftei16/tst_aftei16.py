from __future__ import division, print_function
from pyomo.environ import *
from aftei16 import aftei16
from pyomo.opt import SolverFactory

af = aftei16(1, 3)

af.pprint()
af.u1[0] = 1.22
af.u2[0]= 3.4
ip = SolverFactory('ipopt')
ip.solve(af, tee=True)

af_steady = aftei16(1, 3, steady=True)
af_steady.pprint()
af_steady.u1.pprint()