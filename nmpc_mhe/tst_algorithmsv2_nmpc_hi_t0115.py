from __future__ import print_function
from pyomo.environ import *
from pyomo.core.base import Constraint, Objective, Suffix, minimize
from pyomo.opt import ProblemFormat, SolverFactory
from nmpc_mhe.dync.NMPCGenv2 import NmpcGen
from nmpc_mhe.mods.bfb.nob5_hi_t import bfb_dae
from snap_shot import snap
import sys, os
import itertools, sys
from numpy.random import normal as npm
#   SWITCH TO JUST ONE COLLOCATION POINT AND FINITE ELEMENT
states = ["Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
# x_noisy = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "mom"]
# x_noisy = ["Hse"]
x_noisy = ["Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
u = ["u1"]
u_bounds = {"u1":(162.183495794 * 0.0005, 162.183495794 * 10000)}
ref_state = {("c_capture", ((),)): 0.63}
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

e = NmpcGen(bfb_dae, 400/nfe_mhe, states, u,
            ref_state=ref_state, u_bounds=u_bounds,
            nfe_tnmpc=nfe_mhe, ncp_tnmpc=1,
            nfe_t=5, ncp_t=1)

# 10 fe & _t=1000 definitely degenerate
# 10 fe & _t=900 definitely degenerate
# 10 fe & _t=120 sort-of degenerate
# 10 fe & _t=50 sort-of degenerate
# 10 fe & _t=50 eventually sort-of degenerate
# 10 fe & _t=1 eventually sort-of degenerate
e.SteadyRef.dref = snap
e.load_iguess_steady()


e.SteadyRef.create_bounds()
e.solve_steady_ref()
e.SteadyRef.report_zL(filename="mult_ss")
e.load_d_s(e.PlantSample)
e.PlantSample.create_bounds()
e.solve_dyn(e.PlantSample)

q_cov = {}
for i in tfe:
    for j in itertools.product(lfe, lcp, lc):
        q_cov[("Nse", j), ("Nse", j), i] = 7525.81478168 * 0.005
        q_cov[("Nsc", j), ("Nsc", j), i] = 117.650089456 * 0.005
#             q_cov[("Nse", j), ("Nse", j), i] = 735.706082714 * 0.005
for i in tfe:
    for j in itertools.product(lfe, lcp):
        # q_cov[("Hge", j), ("Hge", j), i] = 2194.25390583 * 0.005
        q_cov[("Hse", j), ("Hse", j), i] = 731143.716603 * 0.005
        q_cov[("Hsc", j), ("Hsc", j), i] = 16668.3312216 * 0.005
        q_cov[("Hge", j), ("Hge", j), i] = 2166.86838591 * 0.005
        q_cov[("Hgc", j), ("Hgc", j), i] = 47.7911012193 * 0.005
        # q_cov[("mom", j), ("mom", j), i] = 1.14042251669 * 0.005

# for i in lfe:
#     for j in [(1,1, 'c'), (5,3, 'c')]:
#         m_cov[("yb", j), ("yb", j), i] = 1e-04

u_cov = {}
for i in [i for i in range(1, nfe_mhe+1)]:
    u_cov["u1", i] = 162.183495794 * 0.005

m_cov = {}
for i in tfe:
    for j in itertools.product(lfe, lcp):
        m_cov[("Tgb", j), ("Tgb", j), i] = 40 * 0.005
        m_cov[("vg", j), ("vg", j), i] = 0.902649386907 * 0.005


e.find_target_ss()  #: Compute target-steady state (beforehand)

#: Create NMPC
e.create_nmpc()
e.update_targets_nmpc()
e.compute_QR_nmpc(n=-1)
e.new_weights_olnmpc(10000, 1e+08)
e.solve_dyn(e.PlantSample, stop_if_nopt=True)
ipsr = SolverFactory('ipopt', executable="/home/dav0/Apps/IpoptSR/Ipopt/build/bin/ipoptSR")
for i in range(1, 1000):
    ps = e.solve_dyn(e.PlantSample, stop_if_nopt=False)
    e.PlantSample.write_nl(name="baddie.nl")
    e.PlantSample.pprint(filename="baddie.txt")
    e.PlantSample.snap_shot(filename="baddie.py")
    e.PlantSample.report_zL(filename="bad_bounds")
    if ps != 0:
        e.PlantSample.write_nl(name="baddie.nl")
        e.PlantSample.pprint(filename="baddie.txt")
        e.PlantSample.snap_shot(filename="baddie.py")
        e.PlantSample.report_zL(filename="bad_bounds")
        e.solve_dyn(e.PlantSample, stop_if_nopt=True)
    e.update_state_real()  # update the current state
    e.update_soi_sp_nmpc()
    #
    e.initialize_olnmpc(e.PlantSample, "real")
    e.load_init_state_nmpc(src_kind="state_dict", state_dict="real")
    stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300)
    # if stat_nmpc != 0:
    #     stat_nmpc = e.solve_dyn(e.olnmpc,
    #                             stop_if_nopt=True,
    #                             skip_update=False,
    #                             iter_max=300, ma57_pivtol=1e-12)
    if stat_nmpc != 0:
        strategy = 1
        if strategy == 1:
            if e.nfe_tnmpc == 1:
                pass
            else:
                e.create_nmpc(newnfe=e.ncp_tnmpc-1, newncp=1)
            e.update_targets_nmpc()
            e.compute_QR_nmpc(n=-1)
            e.new_weights_olnmpc(10000, 1e+02)
            e.initialize_olnmpc(e.PlantSample, "real")
            e.load_init_state_nmpc(src_kind="state_dict", state_dict="real")
            stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300)
        else:
            e.olnmpc.write_nl(name="bad.nl")
            # e.olnmpc.pprint(filename="bad_" + str(i))
            with open("ipopt.opt", "w") as f:
                f.write("linear_solver ma57\n"
                        "ma57_dep_tol 1e-8\nbig_M 1e30\n")
                f.close()
            ipsr.solve(e.olnmpc, tee=True)
    e.update_u(e.olnmpc)
    e.print_r_nmpc()
    e.cycleSamPlant(plant_step=True)
    e.plant_uinject(e.PlantSample, src_kind="dict", nsteps=10, skip_homotopy=True)
    # e.plant_input_gen(e.PlantSample,  "mod", src=e.ss2)

