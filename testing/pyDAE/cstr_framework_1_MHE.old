#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from nmpc_mhe.aux.utils import load_iguess
from nmpc_mhe.aux.utils import reconcile_nvars_mequations  #: counts n_var and m_equations from nl
from nmpc_mhe.pyomo_dae.MHEGen_pyDAE import MheGen_DAE
from sample_mods.cstr_rodrigo.cstr_c_nmpc import cstr_rodrigo_dae
import os, sys

__author__ = "David Thierry @dthierry" #: March 2018

def main():
    states = ["Ca", "T", "Tj"]

    measurements = ['T']
    controls = ["u1"]
    u_bounds = {"u1": (0, 1000)}
    ref_state = {("Ca", (0,)): 0.010}
    e = MheGen_DAE(cstr_rodrigo_dae, 2, states, controls, states, measurements,
                   u_bounds=u_bounds,
                   ref_state=ref_state,
                   override_solver_check=True,
                   k_aug_executable='/home/dav0/devzone/k_aug/cmake-build-k_aug/k_aug')

    #: We need k_aug to run this :(
    Q = {}
    U = {}
    R = {}

    Q['Ca'] = 1.11
    Q['T'] = 99.0
    Q['Tj'] = 1.1

    U['u1'] = 0.22

    R['T'] = 1.22
    e.set_covariance_disturb(Q)
    e.set_covariance_u(U)
    e.set_covariance_meas(R)
    e.lsmhe.Q_mhe.pprint()
    e.lsmhe.R_mhe.pprint()
    e.lsmhe.U_mhe.pprint()
    e.create_rh_sfx()

    e.get_state_vars()
    e.load_iguess_steady()
    load_iguess(e.SteadyRef, e.PlantSample, 0, 0)

    reconcile_nvars_mequations(e.lsmhe)
    e.solve_dyn(e.PlantSample)

    e.init_lsmhe_prep(e.PlantSample)
    # e.lsmhe.pprint(filename="f0")
    e.shift_mhe()
    # e.lsmhe.pprint(filename="f1")
    e.init_step_mhe()
    e.solve_dyn(e.lsmhe,
                skip_update=False,
                max_cpu_time=600,
                ma57_pre_alloc=5, tag="lsmhe")  #: Pre-loaded mhe solve
    e.check_active_bound_noisy()
    e.load_covariance_prior()
    e.set_state_covariance()

    e.regen_objective_fun()  #: Regen erate the obj fun
    e.deact_icc_mhe()  #: Remove the initial conditions

    for i in range(0, 20):  #: Five steps
        e.solve_dyn(e.PlantSample, stop_if_nopt=True)

        e.update_state_real()  # update the current state
        e.update_measurement()
        e.compute_y_offset()  #: Get the offset for y
        e.preparation_phase_mhe(as_strategy=False)

        stat = e.solve_dyn(e.lsmhe,
                           skip_update=False, iter_max=500,
                           jacobian_regularization_value=1e-04,
                           max_cpu_time=600, tag="lsmhe", keepsolve=False, wantparams=False)

        if stat == 1:  #: Try again
            e.lsmhe.write_nl(name="bad_mhe.nl")
            stat = e.solve_dyn(e.lsmhe,
                               skip_update=True,
                               max_cpu_time=600,
                               stop_if_nopt=True,
                               jacobian_regularization_value=1e-02,
                               linear_scaling_on_demand=True, tag="lsmhe")
            if stat != 0:
                sys.exit()
        e.update_state_mhe()  #: get the state from mhe
        #: At this point computing and loading the Covariance is not going to affect the sens update of MHE
        e.prior_phase()
        #
        e.print_r_mhe()
        e.print_r_dyn()

    return e


if __name__ == '__main__':
    e = main()
    file_resmhe = "res_mhe_label_" + e.res_file_suf + ".txt"
    file_resdyn = "res_dyn_label_" + e.res_file_suf + ".txt"
    os.remove(file_resdyn)
    os.remove(file_resmhe)
