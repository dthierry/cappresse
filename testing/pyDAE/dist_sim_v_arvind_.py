#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from nmpc_mhe.pyomo_dae.MHEGen_pyDAE import MheGen_DAE
from sample_mods.distc_pyDAE.dist_cc_v_arvind import mod
from nmpc_mhe.aux.utils import load_iguess, create_bounds
from pyomo.environ import *
from pyomo.core.base import Var, Constraint, Param
from pyomo.opt import SolverFactory, ProblemFormat
from numpy import random
import sys

__author__ = "David Thierry"

def disp_vars(mod, file):
    if not file is None:
        with open(file, "w") as f:
            for i in mod.component_objects(Var):
                i.display(ostream=f)
    else:
        for i in mod.component_objects(Var):
            i.display()

def disp_cons(mod, file):
    if not file is None:
        with open(file, "w") as f:
            for i in mod.component_objects(Constraint):
                i.pprint(ostream=f)
    else:
        for i in mod.component_objects(Constraint):
            i.pprint()

def disp_params(mod, file):
    if not file is None:
        with open(file, "w") as f:
            for i in mod.component_objects(Param):
                i.pprint(ostream=f)
    else:
        for i in mod.component_objects(Param):
            i.pprint()


def main():
    states = ["M"]
    state_bounds = {"M": (1.0, 1e+07),
                    "T": (200, 500),
                    "pm": (1.0, 5e+07),
                    "pn": (1.0, 5e+07),
                    "L": (0.0, 1e+03),
                    "V": (0.0, 1e+03),
                    "x": (0.0, 1.0),
                    "y": (0.0, 1.0),
                    "hl": (1.0, 1e+07),
                    "hv": (1.0, 1e+07),
                    "Qc": (0.0, 1e+08),
                    "D": (0.0, 1e+04),
                    "Vm": (0.0, 1e+04),
                    #"Mv": (0.155 + 1e-06, 1e+04),
                    #"Mv1": (8.5 + 1e-06, 1e+04),
                    #"Mvn": (0.17 + 1e-06, 1e+04)
                    }

    measurements = ["T", "Mv", "Qc"]
    controls = ["u1", "u2"]
    u_bounds = {"u1": (0000.1, 99.999), "u2": (0, None)}
    ref_state1 = {("T", (29,)): 343.15, ("T", (14,)): 361.15}
    ref_state2 = {("T", (29,)): 345.22, ("T", (14,)): 356.23}
    ref_state = ref_state1
    ds = {}
    for k in ref_state2.keys():
        ds[k] = ref_state1[k] - ref_state2[k]



    e = MheGen_DAE(mod, 60, states, controls, states, measurements,
                   u_bounds=u_bounds,
                   ref_state=ref_state,
                   override_solver_check=True,
                   var_bounds=state_bounds,
                   nfe_t=10,
                   k_aug_executable='/home/dav0/in_dev_/kaugma57/bin/k_aug',
                   #ipopt_executable='/home/dav0/PycharmProjects/ipopt_van/build/bin/ipopt')
                   ipopt_executable='/home/dav0/in_dev_/ipopt_vanilla_l1/builds/ipopt_l1/bin/ipopt')
    Q = {}
    U = {}
    R = {}
    Q["M"] = 1
    R["T"] = 6.25e-02
    R["Qc"] = 1.00E+06
    R["Mv"] = 10e-08
    R["Mv1"] = 10e-08
    R["Mvn"] = 10e-08
    U["u1"] = 7.72700925775773761472464684629813E-01 * 0.01
    U["u2"] = 1.78604740940007800236344337463379E+06 * 0.001

    e.set_covariance_disturb(Q)
    e.set_covariance_u(U)
    e.set_covariance_meas(R)
    e.create_rh_sfx()

    e.get_state_vars()

    create_bounds(e.SteadyRef, bounds=state_bounds)
    ipopt = SolverFactory('/home/dav0/in_dev_/ipopt_vanilla_l1/builds/ipopt_l1/bin/ipopt')
    ipopt.options["bound_push"] = 1e-07
    ipopt.options["linear_solver"] = "ma57"
    f = open("ipopt.opt", "w")
    f.close()
    ipopt.solve(e.SteadyRef, tee=True)
    e.SteadyRef.pprint(filename="ccs.txt")
    e.SteadyRef.write(filename="whatevs.nl", io_options={"symbolic_solver_labels": True})
    sys.exit()
    #ipopt.solve(e.SteadyRef, tee=True)

    e.load_iguess_steady()
    ipopt.solve(e.PlantSample,
                tee=True)

    # disp_vars(e.lsmhe, "vars_mhe")
    # disp_params(e.lsmhe, "parm_mhe")

    #: Prepare NMPC
    e.find_target_ss()
    ii = 0
    f = open("errors_mhe", "w")
    f.close()
    f = open("errors_nmpc", "w")
    f.close()
    stat_mhe = 0
    stat_nmpc = 0
    #: Problem loop
    for i in range(0, 1200):
        if i == 350:
            ref_state = ref_state2
            e.change_setpoint(ref_state=ref_state, keepsolve=False, wantparams=False, tag="sp")
        elif 700 <= i < 1050:
            ii += 1
            ref_state = {}
            for k in ref_state2.keys():
                ref_state[k] = ref_state2[k] + ds[k] * ii/350
            e.change_setpoint(ref_state=ref_state, keepsolve=False, wantparams=False, tag="sp")

        #: Plant

        e.solve_dyn(e.PlantSample, stop_if_nopt=True, halt_on_ampl_error=True)

        switch = False
        for var in e.PlantSample.nu_l.itervalues():
            if value(var) > 0.5:
                switch = True
        for var in e.PlantSample.Mv_n_.itervalues():
            if value(var) > 0.5:
                switch = True

        if switch:
            print("Switch is on \n\n\n")
            with open("complementarities", "w") as f:
                e.PlantSample.Mv_p_.display(ostream=f)
                e.PlantSample.Mv_n_.display(ostream=f)
                e.PlantSample.nu_l.display(ostream=f)

        e.update_state_real()  # Update the current state
        e.update_soi_sp_nmpc()  #: To keep track of the state of interest.

        e.update_measurement()  # Update the current measurement
        e.compute_y_offset()  #: Get the offset for y
        #: State-estimation MHE

        e.print_r_mhe()
        e.print_r_dyn()
        e.print_r_nmpc()


        e.update_u(e.SteadyRef2)

        #: Plant cycle
        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        #if i > 10:
            #e.PlantSample.display(filename="plant.txt")
            #e.PlantSample.u2[:].set_value(1E+09)
        e.noisy_plant_manager(sigma=0.0001, action="apply", update_level=True)
        #: 0.001 is a good level


if __name__ == '__main__':
    e = main()
