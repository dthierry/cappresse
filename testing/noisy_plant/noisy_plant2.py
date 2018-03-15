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
from numpy.random import normal as npm

"""This time we actually use a normal distribution
    Now, how do I get this in a dict?"""
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
# sys.exit()
s.SteadyRef.create_bounds()
s.get_state_vars()
s.SteadyRef.report_zL(filename="mult_ss")

s.load_d_s(s.PlantSample)
s.PlantSample.create_bounds()
s.solve_dyn(s.PlantSample)

s.solve_dyn(s.PlantSample, stop_if_nopt=True)
for i in range(1, 100):
    s.solve_dyn(s.PlantSample, stop_if_nopt=True)
    s.update_state_real()
    s.print_r_dyn()
    s.cycleSamPlant(plant_step=True)
    whatHappensNext = npm(0, 0.01)
    for state in s.states:
        x_ic = getattr(s.PlantSample, state + "_ic")
        for key in x_ic.keys():
            x_ic[key].value += x_ic[key].value * whatHappensNext  # one percent perturbation

for i in range(1, 100):
    s.solve_dyn(s.PlantSample, stop_if_nopt=True)
    s.update_state_real()
    s.print_r_dyn()
    s.cycleSamPlant(plant_step=True)