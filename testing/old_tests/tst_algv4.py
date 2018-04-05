#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from pyomo.environ import *
from pyomo.core.base import Constraint, Objective, Suffix, minimize
from pyomo.opt import ProblemFormat, SolverFactory
from nmpc_mhe.dync.NMPCGenv2 import NmpcGen
from sample_mods.bfb.nob5_hi_t import bfb_dae
from pyomo.opt import TerminationCondition
from snapshots.snap_shot import snap
import sys, os
import itertools, sys
from numpy.random import normal as npm

""" """

def main():

    states = ["Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
    x_noisy = ["Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
    u = ["u1"]
    u_bounds = {"u1":(162.183495794 * 0.0005, 162.183495794 * 10000)}
    ref_state = {("c_capture", ((),)): 0.50}

    nfe_mhe = 10
    y = ["Tgb", "vg"]
    nfet = 10
    ncpx = 3
    nfex = 5
    tfe = [i for i in range(1, nfe_mhe)]
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
    #: 1600 was proven to be solveable
    e = NmpcGen(bfb_dae, 1600/nfe_mhe, states, u,
               ref_state=ref_state, u_bounds=u_bounds,
               nfe_t=5, ncp_t=1,
               k_aug_executable="/home/dav0/k_aug/src/k_aug/k_aug"
               )

    # e.SteadyRef.dref = snap
    # e.load_iguess_steady()
    # e.SteadyRef.pprint(filename="whatevs")
    e.SteadyRef.clear_bounds()
    e.load_solfile(e.SteadyRef, "ref_ss.sol")  #: Loads solfile snap

    results = e.ipopt.solve(e.SteadyRef, tee=True, load_solutions=True, report_timing=True)
    if results.solver.termination_condition != TerminationCondition.optimal:
        print("Oh no!")
        sys.exit()
    e.get_state_vars(skip_solve=True)
    e.load_d_s(e.PlantSample)
    e.solve_dyn(e.PlantSample)
    e.solve_dyn(e.PlantSample, stop_if_nopt=True)

    e.find_target_ss()  #: Compute target-steady state (beforehand)

    #: Create NMPC
    e.create_nmpc()
    e.update_targets_nmpc()
    e.compute_QR_nmpc(n=-1)
    e.new_weights_olnmpc(1e-04, 1e+06)
    # results = e.ipopt.solve(e.SteadyRef, tee=True, load_solutions=True, report_timing=True, keepfiles=True)
    # e.write_solfile(e.SteadyRef, "ss", solve=False)
    # e.SteadyRef.display(filename="whatevs")
    # with open("ipopt.opt", "w") as f:
    #     f.write("linear_solver ma57\n")
    #     f.write("mu_init 1e-08\n")
    #     f.write("dual_inf_tol 1e-05\n")
    #     f.write("constr_viol_tol 1e-07\n")
    #     f.write("print_info_string yes\n")
    #     f.close()
    # results = e.ipopt.solve(e.SteadyRef, tee=True, load_solutions=True, report_timing=True, keepfiles=True)
    for i in range(1,10000):
        stat = e.solve_dyn(e.PlantSample, stop_if_nopt=False, tag="plant")
        e.update_state_real()  # update the current state
        e.update_soi_sp_nmpc()


        e.initialize_olnmpc(e.PlantSample, "real")
        e.load_init_state_nmpc(src_kind="state_dict", state_dict="real")
        stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300,
                                jacobian_regularization_value=1e-04, tag="olnmpc")

        if stat_nmpc != 0:
            e.olnmpc.write_nl(name="bad.nl")
            e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300, jacobian_regularization_value=1e-04,
                        tag="olnmpc")
        e.update_u(e.olnmpc)
        e.print_r_nmpc()

        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        e.noisy_plant_manager(sigma=0.001, action="apply", update_level=True)


if __name__ == "__main__":
    main()
