from pyomo.environ import *
from nmpc_mhe.dync.MHEGen import MheGen
from nmpc_mhe.mods.distl.dist_col import DistDiehlNegrete

y = ["T", "Mv", "Mv1", "Mvn"]
states = ["x", "M"]
x_noisy = ["x", "M"]

ntrays = 42
y_vars = {"T": [(i,) for i in range(1, ntrays + 1)],
          "Mv": [(i,) for i in range(2, ntrays)],
          "Mv1":[((),)],
          "Mvn":[((),)]}

x_vars = {"x": [(i,) for i in range(1, ntrays + 1)],
          "M": [(i,) for i in range(1, ntrays + 1)]}

e = MheGen(d_mod=DistDiehlNegrete, y=y, x_noisy=x_noisy, y_vars=y_vars, x_vars=x_vars, states=states)
