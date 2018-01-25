#!/usr/bin/env python

from __future__ import print_function
from pyomo.environ import *
from pyomo.core.base import Constraint, Objective, Suffix, minimize
from pyomo.opt import ProblemFormat, SolverFactory
from nmpc_mhe.dync.DynGenv2 import DynGen
from sample_mods.bfb.nob5_hi_t import bfb_dae
from snapshots.snap_shot import snap
import sys, os
import itertools, sys
from pyomo.opt import ReaderFactory, ResultsFormat, ProblemFormat
from numpy.random import normal as npm
import random
from shutil import copyfile

states = ["Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
# x_noisy = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "mom"]
# x_noisy = ["Hse"]
x_noisy = ["Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
u = ["u1"]
u_bounds = {"u1":(162.183495794 * 0.0005, 162.183495794 * 10000)}
ref_state = {("c_capture", ((),)): 0.50}
# ref_state = {("c_capture", ((),)): 0.66}  nominal
# Known targets 0.38, 0.4, 0.5

nfe_mhe = 10
y = ["Tgb", "vg"]
nfet = 10
ncpx = 3
nfex = 5
tfe = [i for i in range(1, nfe_mhe + 1)]
lfe = [i for i in range(1, nfex + 1)]
lcp = [i for i in range(1, ncpx + 1)]
lc = ['c', 'h', 'n']

y_vars = {
    "Tgb": [i for i in itertools.product(lfe, lcp)],
    "vg": [i for i in itertools.product(lfe, lcp)]
    }
# x_vars = dict()
x_vars = {
          # "Nge": [i for i in itertools.product(lfe, lcp, lc)],
          # "Hge": [i for i in itertools.product(lfe, lcp)],
          "Nsc": [i for i in itertools.product(lfe, lcp, lc)],
          "Hsc": [i for i in itertools.product(lfe, lcp)],
          "Nse": [i for i in itertools.product(lfe, lcp, lc)],
          "Hse": [i for i in itertools.product(lfe, lcp)],
          "Hgc": [i for i in itertools.product(lfe, lcp)],
          "Hge": [i for i in itertools.product(lfe, lcp)],
          # "mom": [i for i in itertools.product(lfe, lcp)]
          }

# States -- (5 * 3 + 6) * fe_x * cp_x.
# For fe_x = 5 and cp_x = 3 we will have 315 differential-states.
s = DynGen(bfb_dae, 800/nfe_mhe, states, u, k_aug_executable="/home/dav0/k2/KKT_matrix/src/k_aug/k_aug")
# 10 fe & _t=1000 definitely degenerate
# 10 fe & _t=900 definitely degenerate
# 10 fe & _t=120 sort-of degenerate
# 10 fe & _t=50 sort-of degenerate
# 10 fe & _t=50 eventually sort-of degenerate
# 10 fe & _t=1 eventually sort-of degenerate
s.SteadyRef.dref = snap
s.load_iguess_steady()
s.SteadyRef.create_bounds()
s.solve_steady_ref()
s.SteadyRef.report_zL(filename="mult_ss")

s.load_d_s(s.PlantSample)

s.ipopt.solve(s.SteadyRef, keepfiles=True)
finame = s.ipopt._soln_file
cwd = os.getcwd()
filename = "ref_ss.sol"
# copyfile(finame, cwd + "/ref_ss.sol")
with open("file_a", "w") as file:
    for var in s.SteadyRef.component_data_objects(Var):
        var.set_value(0)
        val = var.value
        file.write(str(val))
        file.write('\n')
    file.close()
reader = ReaderFactory(ResultsFormat.sol)
results = reader(filename)
_, smapid = s.SteadyRef.write("whathevs.nl", format=ProblemFormat.nl)
smap = s.SteadyRef.solutions.symbol_map[smapid]
results._smap = smap
s.SteadyRef.solutions.load_from(results)
with open("file_b", "w") as file:
    for var in s.SteadyRef.component_data_objects(Var):
        val = var.value
        file.write(str(val))
        file.write('\n')
    file.close()

s.ipopt.solve(s.SteadyRef, tee=True, load_solutions=False, report_timing=True)



# example
