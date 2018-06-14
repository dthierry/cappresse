#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from nmpc_mhe.dync.MHEGenv2 import MheGen
from sample_mods.bfb.nob5_hi_t import bfb_dae
import itertools, sys
from pyomo.environ import *
from pyomo.opt import *
"""Testing the new preparation phases with ideal strategies"""

def main():
    u_weight = 1E-04
    states = ["Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
    x_noisy = ["Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
    u = ["u1"]
    u_bounds = {"u1":(162.183495794 * 0.0005, 162.183495794 * 10000)}
    ref_state = {("c_capture", ((),)): 0.50}

    nfe_mhe = 20
    y = ["Tgb", "vg"]
    nfet = 10
    ncpx = 3
    nfex = 5
    tfe = [i for i in range(0, nfe_mhe)]
    lfe = [i for i in range(0, nfex)]
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
    e = MheGen(bfb_dae, 1600/nfe_mhe, states, u, x_noisy, x_vars, y, y_vars,
               nfe_tmhe=nfe_mhe, ncp_tmhe=1,
               nfe_tnmpc=nfe_mhe, ncp_tnmpc=1,
               ref_state=ref_state, u_bounds=u_bounds,
               nfe_t=5, ncp_t=1,
               k_aug_executable="/home/dav0/k_aug/src/k_aug/k_aug",
               dot_driver_executable="/home/dav0/k_aug/src/k_aug/dot_driver/dot_driver"
               )

    e.load_solfile(e.SteadyRef, "ref_ss.sol")  #: Loads solfile snap
    results = e.ipopt.solve(e.SteadyRef, tee=True, load_solutions=True, report_timing=True)

    e.get_state_vars(skip_solve=True)
    e.SteadyRef.report_zL(filename="mult_ss")
    e.load_d_s(e.PlantSample)
    e.PlantSample.create_bounds()
    e.solve_dyn(e.PlantSample)

    e.find_target_ss()  #: Compute target-steady state (beforehand)

    K = SolverFactory('k_aug', executable='/home/dav0/devzone/k_aug/bin/k_aug')
    K.options["deb_kkt"] = ""
    #: Create NMPC
    e.solve_dyn(e.PlantSample, stop_if_nopt=True)
    # isnap = [i*50 for i in range(1, 25)]
    isnap = [i*25 for i in range(2, 30)]
    j = 1
    cw_u = 1e+06
    for i in range(1, 600):
        if i in isnap:
            keepsolve=False
            wantparams=False
        else:
            keepsolve=False
            wantparams=False
        if i == 200:
            j = 1
            ref_state = {("c_capture", ((),)): 0.63}
            e.change_setpoint(ref_state=ref_state, keepsolve=True, wantparams=True, tag="sp")
        elif i == 400:
            j = 1
            ref_state = {("c_capture", ((),)): 0.5}
            e.change_setpoint(ref_state=ref_state, keepsolve=True, wantparams=True, tag="sp")

        stat = e.solve_dyn(e.PlantSample, stop_if_nopt=False, tag="plant", keepsolve=keepsolve, wantparams=wantparams)
        if stat == 1:
            e.noisy_plant_manager(action="remove")
            e.solve_dyn(e.PlantSample, stop_if_nopt=True, tag="plant", keepsolve=keepsolve,
                        wantparams=wantparams)  #: Try again (without noise)

        e.update_state_real()  # update the current state
        e.update_soi_sp_nmpc()

        e.print_r_mhe()
        e.print_r_dyn()
        #

        e.update_u(e.SteadyRef2)  #: Get the resulting input for k+1

        e.print_r_nmpc()
        #
        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        # e.noisy_plant_manager(sigma=0.001, action="apply", update_level=True)
        j += 1

        if i in 1:
            e.PlantSample.ofun = Objective(expr=1)
            e.PlantSample.write("my_file.nl")
            K.solve(e.PlantSample)
            sys.exit()



if __name__ == "__main__":
    main()
