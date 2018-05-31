#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from nmpc_mhe.aux.utils import load_iguess
from nmpc_mhe.pyomo_dae.MHEGen_pyDAE import MheGen_DAE
from sample_mods.cstr_rodrigo.cstr_c_nmpc import cstr_rodrigo_dae
import os, sys
import matplotlib.pyplot as plt

__author__ = "David Thierry @dthierry"  #: May 2018
#: FULL controller ideal-ideal nmpc-mhe


def main():

    states = ["Ca", "T", "Tj"]
    measurements = ['T']
    controls = ["u1"]
    u_bounds = {"u1": (200, 1000)}
    state_bounds = {"Ca": (0.0, None), "T": (2.0E+02, None), "Tj": (2.0E+02, None)}
    ref_state = {("Ca", (0,)): 0.010}
    mod = cstr_rodrigo_dae(2, 2)  #: Some model
    #: MHE-NMPC class
    e = MheGen_DAE(mod, 2, states, controls, states, measurements,
                   u_bounds=u_bounds,
                   ref_state=ref_state,
                   override_solver_check=True,
                   var_bounds=state_bounds, 
                   k_aug_executable='/home/dav0/devzone/k_aug/cmake-build-k_aug/bin/k_aug',
                   dot_driver_executable='/home/dav0/devzone/k_aug/src/k_aug/dot_driver')
    #: Covariance Matrices
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
    e.create_rh_sfx()
    e.get_state_vars()
    #: Initial guesses
    e.load_iguess_steady()
    load_iguess(e.SteadyRef, e.PlantSample, 0, 0)
    e.solve_dyn(e.PlantSample)
    #: Prepare MHE
    e.init_lsmhe_prep(e.PlantSample)
    e.shift_mhe()
    e.init_step_mhe()
    e.solve_dyn(e.lsmhe,
                skip_update=False,
                max_cpu_time=600,
                ma57_pre_alloc=5, tag="lsmhe")  #: Pre-loaded mhe solve

    e.prior_phase()
    e.deact_icc_mhe()  #: Remove the initial conditions
    #: Prepare NMPC
    e.find_target_ss()
    e.create_nmpc()
    e.create_suffixes_nmpc()
    e.update_targets_nmpc()
    e.compute_QR_nmpc(n=-1)
    e.new_weights_olnmpc(1E-04, 1e+06)
    #: Problem loop
    for i in range(0, 300):  #: Five steps
        #: set-point change
        if i in [30 * (j * 2) for j in range(0, 100)]:
            ref_state = {("Ca", (0,)): 0.018}
            e.change_setpoint(ref_state=ref_state, keepsolve=True, wantparams=True, tag="sp")
            e.compute_QR_nmpc(n=-1)
            e.new_weights_olnmpc(1e-04, 1e+06)
        #: set point change
        elif i in [30 * (j * 2 + 1) for j in range(0, 100)]:
            ref_state = {("Ca", (0,)): 0.021}
            e.change_setpoint(ref_state=ref_state, keepsolve=True, wantparams=True, tag="sp")
            e.compute_QR_nmpc(n=-1)
            e.new_weights_olnmpc(1e-04, 1e+06)

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
                           max_cpu_time=600, tag="lsmhe", keepsolve=False, wantparams=False)

        if stat != 0:
            sys.exit()
        #: Prior-phase and arrival cost
        e.update_state_mhe()  #: get the state from mhe
        e.prior_phase()

        e.print_r_mhe()
        e.print_r_dyn()
        #: Control NMPC
        e.preparation_phase_nmpc(as_strategy=False, make_prediction=False)
        stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300, tag="olnmpc")
        if stat_nmpc != 0:
            sys.exit()
        e.print_r_nmpc()
        e.update_u(e.olnmpc)
        #: Plant cycle
        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        e.noisy_plant_manager(sigma=0.0015, action="apply", update_level=True)

    #: print our state of interest
    plt.plot(e.soi_dict[("Ca", (0,))])
    plt.show()
    return e


if __name__ == '__main__':
    e = main()
    #: Cleanup
    # file_resmhe = "res_mhe_label_" + e.res_file_suf + ".txt"
    # file_resdyn = "res_dyn_label_" + e.res_file_suf + ".txt"
    # os.remove(file_resdyn)
    # os.remove(file_resmhe)
