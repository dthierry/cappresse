#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
# from nmpc_mhe.dync.MHEGenv2 import MheGen
from nmpc_mhe.dync.DynGenv2 import DynGen
from sample_mods.ltis9012.ltis9012 import ltiis9016
import itertools, sys
import sys
from shutil import copyfile
from pyomo.environ import *
from pyomo.opt import *
"""Testing the new preparation phases with ideal strategies"""

def main():
    lkcond = [i * 20 for i in range(0, 31)]

    u_weight = 1E-04
    states = ["x"]
    x_noisy = ["x"]
    u = ["u1"]
    #u_bounds = {"u1":(0, 100), "u2":(0, 100)}
    ref_state = {("x", (1,)): 5}

    nfe_mhe = 20
    y = ["Tgb", "vg"]
    nfet = 10
    ncpx = 3
    nfex = 5

    # States -- (5 * 3 + 6) * fe_x * cp_x.
    # For fe_x = 5 and cp_x = 3 we will have 315 differential-states.
    #: 1600 was proven to be solveable
    e = DynGen(ltiis9016, 1600/nfe_mhe, states, u,
               nfe_tmhe=nfe_mhe, ncp_tmhe=1,
               nfe_tnmpc=nfe_mhe, ncp_tnmpc=1,
               ref_state=ref_state,
               nfe_t=5, ncp_t=1,
               k_aug_executable="/home/dav0/k_aug/bin/k_aug",
               dot_driver_executable="/home/dav0/k_aug/src/k_aug/dot_driver/dot_driver"
               )

    e.load_solfile(e.SteadyRef, "ref_ss.sol")  #: Loads solfile snap
    results = e.ipopt.solve(e.SteadyRef, tee=True, load_solutions=True, report_timing=True)

    e.get_state_vars(skip_solve=True)
    e.load_d_s(e.PlantSample)
    e.PlantSample.create_bounds()
    e.solve_dyn(e.PlantSample)

    # e.find_target_ss()  #: Compute target-steady state (beforehand)

    K = SolverFactory('k_aug', executable='/home/dav0/k_aug/bin/k_aug')
    K.options["deb_kkt"] = ""

    #: Create NMPC
    e.solve_dyn(e.PlantSample, stop_if_nopt=True)
    # isnap = [i*50 for i in range(1, 25)]
    isnap = [i*25 for i in range(2, 30)]
    j = 1
    cw_u = 1e+06
    e.PlantSample.dcdp = Suffix(direction=Suffix.EXPORT)
    e.PlantSample.var_order = Suffix(direction=Suffix.EXPORT)

    ##
    with open("states_sens.txt", "w") as f:
        f.close()
    #:whatnot
    ii = 1
    for x in e.states:
        con = getattr(e.PlantSample, x + '_icc')
        with open("states_sens.txt", "a") as f:
            for key in con.keys():
                if not con[key].active:
                    continue
                f.write(x + "\t" + str(key) + "\n")
            f.close()
        # con.set_suffix_value(e.PlantSample.dcdp, 1)
        var = getattr(e.PlantSample, x)
        for key in var.keys():
            if key[1] == e.ncp_t:
                if var[key].stale:
                    continue
                con[key[2:]].set_suffix_value(e.PlantSample.dcdp, ii)
                var[key].set_suffix_value(e.PlantSample.var_order, ii)
                ii += 1
    f = open("suf0.txt", "w")
    e.PlantSample.var_order.pprint(ostream=f)
    f.close()
    e.PlantSample.dum_of = Objective(expr=1,sense=minimize)
    e.PlantSample.write_nl(name="whatevs0.nl")
    kaug = SolverFactory("k_aug",
                         executable="/home/dav0/k_aug/bin/k_aug")
    kaug.options["compute_dsdp"] = ""
    f = open("suf1.txt", "w")
    e.PlantSample.var_order.pprint(ostream=f)
    f.close()
    e.PlantSample.write_nl(name="whatevs.nl")
    kaug.solve(e.PlantSample, tee=True)
    #sys.exit()
   

    ##


    for i in range(1, 600):
        if i in isnap:
            keepsolve=False
            wantparams=False
        else:
            keepsolve=False
            wantparams=False
        if i == 200:
            j = 1
        elif i == 400:
            j = 1

        stat = e.solve_dyn(e.PlantSample, stop_if_nopt=False, tag="plant", keepsolve=keepsolve, wantparams=wantparams)
        if stat == 1:
            e.noisy_plant_manager(action="remove")
            e.solve_dyn(e.PlantSample, stop_if_nopt=True, tag="plant", keepsolve=keepsolve,
                        wantparams=wantparams)  #: Try again (without noise)

        e.update_state_real()  # update the current state

        e.print_r_dyn()
        #
        # e.SteadyRef2.pprint()
        # e.update_u(e.SteadyRef2)  #: Get the resulting input for k+1

        #
        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        # e.noisy_plant_manager(sigma=0.001, action="apply", update_level=True)
        j += 1

        if i in lkcond:
            kaug.solve(e.PlantSample, tee=True)
            copyfile('dxdp_.dat', './dxdptrunk/dxdp_' + str(i) + '_.txt')
            print('sensitivity matrix file copied!!!!')



if __name__ == "__main__":
    main()
