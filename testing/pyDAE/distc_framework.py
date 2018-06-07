#!/usr/bin/env python
# -*- coding: utf-8
from __future__ import division
from __future__ import print_function

from nmpc_mhe.pyomo_dae.MHEGen_pyDAE import MheGen_DAE
from sample_mods.distc_pyDAE.distcpydaemod import mod
from nmpc_mhe.aux.utils import load_iguess, create_bounds
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
    states = ["x", "M"]
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
                    "Mv": (0.155 + 1e-06, 1e+04),
                    "Mv1": (8.5 + 1e-06, 1e+04),
                    "Mvn": (0.17 + 1e-06, 1e+04)
                    }

    measurements = ["T", "Mv", "Mv1", "Mvn"]
    controls = ["u1", "u2"]
    u_bounds = {"u1": (0000.1, 99.999), "u2": (0, None)}
    ref_state = {("T", (29,)): 343.15, ("T", (14,)): 361.15}



    e = MheGen_DAE(mod, 6, states, controls, states, measurements,
                   u_bounds=u_bounds,
                   ref_state=ref_state,
                   override_solver_check=True,
                   var_bounds=state_bounds,
                   nfe_t=10)
    Q = {}
    U = {}
    R = {}
    Q["x"] = 1e-05
    Q["M"] = 1
    R["T"] = 6.25e-02
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
    ipopt = SolverFactory('ipopt')
    ipopt.options["bound_push"] = 1e-07
    ipopt.solve(e.SteadyRef, tee=True)

    e.load_iguess_steady()
    ipopt.solve(e.PlantSample,
                tee=True,
                symbolic_solver_labels=True)

    e.init_lsmhe_prep(e.PlantSample)
    e.shift_mhe()
    e.init_step_mhe()
    e.solve_dyn(e.lsmhe,
                skip_update=False,
                max_cpu_time=600,
                ma57_pre_alloc=5, tag="lsmhe")
    # disp_vars(e.lsmhe, "vars_mhe")
    # disp_params(e.lsmhe, "parm_mhe")

    e.prior_phase()
    e.deact_icc_mhe()  #: Remove the initial conditions
    #: Prepare NMPC
    e.find_target_ss()
    e.create_nmpc()
    e.create_suffixes_nmpc()
    e.update_targets_nmpc()
    e.compute_QR_nmpc(n=-1)
    e.new_weights_olnmpc(1e-04, 1e+00)

    #: Problem loop
    for i in range(0, 300):

        #: Plant
        e.solve_dyn(e.PlantSample, stop_if_nopt=True)

        e.update_state_real()  # Update the current state
        e.update_soi_sp_nmpc()  #: To keep track of the state of interest.

        e.update_measurement()  # Update the current measurement
        e.compute_y_offset()  #: Get the offset for y
        #: State-estimation MHE
        e.preparation_phase_mhe(as_strategy=False)
        stat = e.solve_dyn(e.lsmhe,
                           skip_update=False, iter_max=500,
                           jacobian_regularization_value=1e-04,
                           max_cpu_time=600,
                           tag="lsmhe",
                           keepsolve=False,
                           wantparams=False)
        if stat != 0:
            sys.exit()
        #: Prior-phase and arrival cost
        e.update_state_mhe()  #: get the state from mhe
        e.prior_phase()

        e.print_r_mhe()
        e.print_r_dyn()
        #: Control NMPC
        e.preparation_phase_nmpc(as_strategy=False, make_prediction=False)
        try:
            stat_nmpc = e.solve_dyn(e.olnmpc,
                                    skip_update=False,
                                    max_cpu_time=300,
                                    tag="olnmpc")
        except ValueError:
            stat_nmpc = e.solve_dyn(e.olnmpc,
                                    skip_update=False,
                                    max_cpu_time=300,
                                    tag="olnmpc")

        if stat_nmpc != 0:
            stat_nmpc = e.solve_dyn(e.olnmpc,
                                    skip_update=False,
                                    max_cpu_time=300,
                                    tag="olnmpc")
            if stat_nmpc != 0:
                sys.exit()
        e.print_r_nmpc()
        e.update_u(e.olnmpc)
        #: Plant cycle
        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        e.noisy_plant_manager(sigma=0.0001, action="apply", update_level=True)


if __name__ == '__main__':
    e = main()
