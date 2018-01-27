#!/usr/bin/env python

from __future__ import print_function
from pyomo.environ import *
from pyomo.core.base import Constraint, Objective, Suffix, minimize
from pyomo.opt import ProblemFormat, SolverFactory
from nmpc_mhe.dync.MHEGenv2 import MheGen
from sample_mods.bfb.nob5_hi_t import bfb_dae
from snapshots.snap_shot import snap
import sys, os
import itertools, sys
from numpy.random import normal as npm
from pyomo.opt import ReaderFactory, ResultsFormat, TerminationCondition

"""We would like to get the sol file from the mhe and the corresponding pyomo params"""


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
    tfe = [i for i in range(1, nfe_mhe + 1)]
    lfe = [i for i in range(1, nfex + 1)]
    lcp = [i for i in range(1, ncpx + 1)]
    lc = ['c', 'h', 'n']

    y_vars = {
        "Tgb": [i for i in itertools.product(lfe, lcp)],
        "vg": [i for i in itertools.product(lfe, lcp)]
        }

    x_vars = {
              "Nsc": [i for i in itertools.product(lfe, lcp, lc)],
              "Hsc": [i for i in itertools.product(lfe, lcp)],
              "Nse": [i for i in itertools.product(lfe, lcp, lc)],
              "Hse": [i for i in itertools.product(lfe, lcp)],
              "Hgc": [i for i in itertools.product(lfe, lcp)],
              "Hge": [i for i in itertools.product(lfe, lcp)]}

    # States -- (5 * 3 + 6) * fe_x * cp_x.
    # For fe_x = 5 and cp_x = 3 we will have 315 differential-states.

    e = MheGen(bfb_dae, 800/nfe_mhe, states, u, x_noisy, x_vars, y, y_vars,
               nfe_tmhe=nfe_mhe, ncp_tmhe=1,
               nfe_tnmpc=nfe_mhe, ncp_tnmpc=1,
               ref_state=ref_state, u_bounds=u_bounds,
               nfe_t=5, ncp_t=1,
               k_aug_executable="/home/dav0/k2/KKT_matrix/src/k_aug/k_aug"
               )


    # filename = "/home/dav0/nmpc_mhe_q/testing/ref_ss.sol"
    # # copyfile(finame, cwd + "/ref_ss.sol")
    # reader = ReaderFactory(ResultsFormat.sol)
    # results = reader(filename)
    # _, smapid = e.SteadyRef.write("whathevs.nl", format=ProblemFormat.nl)
    # smap = e.SteadyRef.solutions.symbol_map[smapid]
    # results._smap = smap
    # e.SteadyRef.solutions.load_from(results)

    e.load_solfile(e.SteadyRef, "ref_ss.sol")  #: Loads solfile snap
    #
    results = e.ipopt.solve(e.SteadyRef, tee=True, load_solutions=True, report_timing=True)
    if results.solver.termination_condition != TerminationCondition.optimal:
        print("Oh no!")
        sys.exit()

    e.get_state_vars(skip_solve=True)
    e.SteadyRef.report_zL(filename="mult_ss")
    e.load_d_s(e.PlantSample)
    e.PlantSample.create_bounds()
    e.solve_dyn(e.PlantSample)

    q_cov = {}
    for i in tfe:
        for j in itertools.product(lfe, lcp, lc):
            q_cov[("Nse", j), ("Nse", j), i] = 7525.81478168 * 0.005
            q_cov[("Nsc", j), ("Nsc", j), i] = 117.650089456 * 0.005
    for i in tfe:
        for j in itertools.product(lfe, lcp):
            q_cov[("Hse", j), ("Hse", j), i] = 731143.716603 * 0.005
            q_cov[("Hsc", j), ("Hsc", j), i] = 16668.3312216 * 0.005
            q_cov[("Hge", j), ("Hge", j), i] = 2166.86838591 * 0.005
            q_cov[("Hgc", j), ("Hgc", j), i] = 47.7911012193 * 0.005

    u_cov = {}
    for i in [i for i in range(1, nfe_mhe+1)]:
        u_cov["u1", i] = 162.183495794 * 0.005

    m_cov = {}
    for i in tfe:
        for j in itertools.product(lfe, lcp):
            m_cov[("Tgb", j), ("Tgb", j), i] = 40 * 0.005
            m_cov[("vg", j), ("vg", j), i] = 0.902649386907 * 0.005


    # e.set_covariance_meas(m_cov)
    #
    # e.set_covariance_disturb(q_cov)
    #
    # e.set_covariance_u(u_cov)



    e.create_rh_sfx()  #: Reduced hessian computation
    # e.init_lsmhe_prep(e.PlantSample)

    # e.shift_mhe()
    # e.init_step_mhe()
    #
    e.param_reader(e.lsmhe, "gimmemyparams.json")

    e.solve_dyn(e.lsmhe,
                skip_update=False,
                max_cpu_time=600,
                ma57_pre_alloc=5, tag="lsmhe", keepfiles=True, keepsolve=True,
                loadsolve=True, solfile="LSMHE(Least-SquaresMHE)_1516998809_26038.sol")  #: Pre-loaded mhe solve
    e.param_writer(e.lsmhe, "gimmemyparams.json")
    # e.write_solfile(e.lsmhe, ma57_pre_alloc=5)

    sys.exit()
    # print(e.ipopt._soln_file, type(e.ipopt._soln_file))
    e.check_active_bound_noisy()
    e.load_covariance_prior()
    e.set_state_covariance()

    e.regen_objective_fun()  #: Regen erate the obj fun
    e.deact_icc_mhe()  #: Remove the initial conditions

    e.set_prior_state_from_prior_mhe()  #: Update prior-state
    # we have the mhe that we want to solve here
    e.find_target_ss(loadsolve=True,
                     solfile="SteadyRef2(reference)_1516999382_61461.sol")  #: Compute target-steady state (beforehand)
    # e.write_solfile(e.SteadyRef2)




if __name__ == "__main__":
    main()
