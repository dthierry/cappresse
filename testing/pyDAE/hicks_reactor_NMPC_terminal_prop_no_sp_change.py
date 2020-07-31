#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from sample_mods.hicks_reactor.hicks_reactor_devin import hicks_reactor_devin_dae_w_AE
from nmpc_mhe.pyomo_dae.NMPCGen_pyDAE import NmpcGen_DAE
from nmpc_mhe.aux.utils import load_iguess
from nmpc_mhe.aux.utils import reconcile_nvars_mequations
import matplotlib.pyplot as plt
import sys, os

__author__ = "Kuan-Han Lin" #: Jul 2020

def main():
    states = ["zc", "zT"]
    controls = ["d_u1", "d_u2"]
    # u_bounds = {"d_u1": (-0.4167, 0.6), "d_u2": (-0.4750, 0.5)}
    # state_bounds = {"zc": (0.0, None), "zT":(0.0, None)}
    u_bounds = {"d_u1": (0.1667, 1.), "d_u2": (0.025, 1.)}
    state_bounds = {"zc": (0.0, 1.0), "zT":(0.0, None)}

    ref_state = {("zc", (0,)): 0.6416, ("zT", (0,)): 0.5387}
    mod = hicks_reactor_devin_dae_w_AE(1, 1)
    e = NmpcGen_DAE(mod, 1, states, controls,
                    nfe_tnmpc = 10,
                    var_bounds=state_bounds,
                    u_bounds=u_bounds,
                    ref_state=ref_state,
                    override_solver_check=True,
                    k_aug_executable='/home/khl/Apps/k_aug/bin/k_aug')
    return e


if __name__ == '__main__':
    e = main()
    e.get_state_vars()
    e.load_iguess_steady()
    load_iguess(e.SteadyRef, e.PlantSample, 0, 0)
    e.create_nmpc()
    reconcile_nvars_mequations(e.olnmpc)
    e.solve_dyn(e.PlantSample)
    e.find_target_ss()
    # e.PlantSample.display()
    e.create_suffixes_nmpc()
    e.update_targets_nmpc()
    # Q_nmpc = {("zc", (0,)): 10., ("zT", (0,)): 2.}
    # R_nmpc = {"d_u1": 1, "d_u2": 0.5}
    # e.compute_QR_nmpc(define_by_user = True, Q_nmpc = Q_nmpc, R_nmpc = R_nmpc)
    # e.add_terminal_property_nmpc(rhox = 50., rhou = 35.,
    #                                     simulate_points = 50, state_norm = 0.005)
    e.compute_QR_nmpc(n=-1)
    e.add_terminal_property_nmpc(rhox = 50., rhou = 35.,
                                        simulate_points = 50, state_norm = 0.005)
    e.new_weights_olnmpc(1., 1.)
    e.solve_dyn(e.PlantSample, stop_if_nopt=True) 

    # #here to set ic!!!!
    ps = e.PlantSample
    tend = ps.t[-1]
    now = 4
    pert_zc = [0.1, 0.1, 0.18, -0.2, -0.15]
    pert_zT = [0.052, -0.071, -0.002, 0.05, -0.03]
    ps.zc[(tend,(0,))].value = 0.6416 + pert_zc[now]
    ps.zT[(tend,(0,))].value = 0.5387 + pert_zT[now]
    
    e.update_state_real()  # update the current state
    e.update_soi_sp_nmpc()
    e.preparation_phase_nmpc(as_strategy=False, make_prediction=False, plant_state=True)
    # e.load_init_state_nmpc(src_kind="state_dict", state_dict="estimated")
    # e.olnmpc.pprint(filename="new_framework.txt")
    stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300,
                            jacobian_regularization_value=1e-04, tag="olnmpc",
                            keepsolve=False, wantparams=False)
    # e.olnmpc.objfun_nmpc.pprint()
    # e.olnmpc.xmpc_ref_nmpc.display()

    e.print_r_nmpc()
    e.update_u(e.olnmpc)  #: Get the resulting input for k+1
    e.cycleSamPlant(plant_step=True)
    e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
    # e.noisy_plant_manager(sigma=0.001, action="apply", update_level=True)
    for i in range(0, 40): #500
        # if i in [30 * (j * 2) for j in range(0, 100)]:
        # if i == 0:
        #     ref_state = {("zc", (0,)): 0.6416, ("zT", (0,)): 0.5387}
        #     e.change_setpoint(ref_state=ref_state, keepsolve=True, wantparams=True, tag="sp")
        #     e.compute_QR_nmpc(n=-1)
        #     e.add_terminal_property_nmpc(rhox = 50., rhou = 35.,
        #                                         simulate_points = 50, state_norm = 0.005)
        #     e.new_weights_olnmpc(1., 1.)
        # elif i in [30 * (j * 2 + 1) for j in range(0, 100)]:
        # if i == 40:
        #     ref_state = {("zc", (0,)): 0.76}#, ("zT", (0,)): 0.495638285400246}
        #     e.change_setpoint(ref_state=ref_state, keepsolve=True, wantparams=True, tag="sp")
        #     e.compute_QR_nmpc(n=-1)
        #     e.add_terminal_property_nmpc(rhox = 50., rhou = 35.,
        #                                         simulate_points = 50, state_norm = 0.005)
        #     e.new_weights_olnmpc(1., 1.)
        
        stat_plant = e.solve_dyn(e.PlantSample, stop_if_nopt=True)
        if stat_plant != 0:
            sys.exit()
        e.update_state_real()  # update the current state
        e.update_soi_sp_nmpc()
        e.preparation_phase_nmpc(as_strategy=False, make_prediction=False, plant_state=True)
        stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300,
                                jacobian_regularization_value=1e-04, tag="olnmpc",
                                keepsolve=False, wantparams=False)
        if stat_nmpc != 0:
            sys.exit()
        e.print_r_nmpc()
        e.update_u(e.olnmpc)  #: Get the resulting input for k+1
        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        # e.noisy_plant_manager(sigma=0.001, action="apply", update_level=True)

    plt.plot(e.soi_dict[("zc", (0,))])
    plt.show()
    plt.plot(e.soi_dict[("zT", (0,))])
    plt.show()


