from __future__ import print_function
from pyomo.environ import *
from pyomo.core.base import Constraint, Objective, Suffix, minimize
from pyomo.opt import ProblemFormat, SolverFactory
from nmpc_mhe.dync.MHEGen import MheGen
from nmpc_mhe.mods.bfb.nob3 import bfb_dae
from snap_shot import snap
import sys, os
import itertools, sys
from numpy.random import normal as npm

states = ["Hgb", "Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
# x_noisy = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "mom"]
x_noisy = ["Hse"]
u = ["u1"]
u_bounds = {"u1":(162.183495794 * 0.0005, 162.183495794 * 10000)}

ref_state = {("c_capture", ((),)): 0.55}
# Known targets 0.38, 0.4, 0.5


y = ["Tgb", "vg"]
nfet = 2
ncpx = 3
nfex = 5
tfe = [i for i in range(1, nfet + 1)]
lfe = [i for i in range(1, nfex + 1)]
lcp = [i for i in range(1, ncpx + 1)]
lc = ['c', 'h', 'n']

y_vars = {
    "Tgb": [(1,1), (5,3)],
    "vg": [(1,1), (5,3)],
    # "yb": [(1,1, 'c'), (5, 3, 'c')],
    }
x_vars = dict()
x_vars = {
          "Hse": [(1, 1)],
          }

# States -- (5 * 3 + 6) * fe_x * cp_x.
# For fe_x = 5 and cp_x = 3 we will have 315 differential-states.

e = MheGen(d_mod=bfb_dae,
           y=y,
           x_noisy=x_noisy,
           y_vars=y_vars,
           x_vars=x_vars,
           states=states,
           u=u,
           ref_state=ref_state,
           u_bounds=u_bounds,
           diag_QR=True,
           IgnoreProcessNoise=True,
           nfe_t=10, _t=500)
e.ss.dref = snap

e.load_iguess_ss()
# sys.exit()
e.ss.create_bounds()
e.solve_ss()
e.ss.report_zL(filename="mult_ss")
e.load_d_s(e.d1)
e.d1.create_bounds()
e.solve_d(e.d1)

q_cov = {}
# for i in tfe:
#     if i < nfet:
#         for j in itertools.product(lfe, lcp, lc):
#             q_cov[("Ngb", j), ("Ngb", j), i] = 0.01*5.562535786e-05
#             q_cov[("Ngc", j), ("Ngc", j), i] = 0.01*0.000335771530697
#             q_cov[("Nsc", j), ("Nsc", j), i] = 0.01*739.786503718
#             q_cov[("Nge", j), ("Nge", j), i] = 0.01*0.0100570141164
#             q_cov[("Nse", j), ("Nse", j), i] = 0.01*641.425020561
for i in tfe:
    if i < nfet:
        for j in [(1,1)]:
            q_cov[("Hse", j), ("Hse", j), i] = 561353.476801 * 0.01


m_cov = {}
for i in tfe:
    for j in [(1,1), (5,3)]:
        m_cov[("Tgb", j), ("Tgb", j), i] = 2
        m_cov[("vg", j), ("vg", j), i] = 0.1

# for i in lfe:
#     for j in [(1,1, 'c'), (5,3, 'c')]:
#         m_cov[("yb", j), ("yb", j), i] = 1e-04

u_cov = {}
for i in tfe:
    u_cov["u1", i] = 5



e.set_covariance_meas(m_cov)
e.set_covariance_disturb(q_cov)
e.set_covariance_u(u_cov)
e.create_rh_sfx()  #: Reduced hessian computation
e.find_target_ss()  #: Compute target-steady state (beforehand)
e.ss2.report_zL(filename="mult_ss0.txt")

# Preparation phase
e.init_lsmhe_prep(e.d1, update=False)
e.lsmhe.display(filename="mhe0")
with open("bounds_sim.txt", "w") as f:
    e.lsmhe.ipopt_zL_out.display(ostream=f)
    e.lsmhe.ipopt_zL_in.display(ostream=f)
    f.close()

dum = e.d_mod(1, e.ncp_t, _t=e.hi_t)

dum.create_bounds()
#e.init_step_mhe(dum, e.nfe_t)
tst = e.solve_d(e.d1, skip_update=False)  #: Pre-loaded mhe solve
e.find_target_ss()  #: Compute target-steady state (beforehand)
e.ss2.report_zL(filename="mult_ss2.txt")

# For ideal nmpc
for i in range(0, 10):
    print(str(i) + "--"*20, file=sys.stderr)
    print("*"*100)

    e.lsmhe.display(filename="mhe" + str(i))

    for ii in range(1, e.nfe_t + 1):
        e.lsmhe.hi_t[ii].value = value(e.lsmhe.hi_t[ii]) * (1 + i*0.2)
    e.lsmhe.hi_t.display()
    e.solve_d(e.lsmhe)
