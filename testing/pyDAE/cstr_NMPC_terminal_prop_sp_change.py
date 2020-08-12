#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from sample_mods.cstr_rodrigo.cstr_c_nmpc import cstr_rodrigo_dae
from nmpc_mhe.pyomo_dae.NMPCGen_pyDAE import NmpcGen_DAE
from nmpc_mhe.aux.utils import load_iguess
from nmpc_mhe.aux.utils import reconcile_nvars_mequations
import matplotlib.pyplot as plt
import sys, os

__author__ = "Kuan-Han Lin @kuanhanl" #: Jul 2020

def main():
    states = ["Ca", "T", "Tj"]
    controls = ["u1"]
    # u_bounds = {"u1": (200, 1000)}
    # state_bounds = {"Ca": (0.0, None), "T":(2.0E+02, None), "Tj":(2.0E+02, None)}
    u_bounds = {"u1": (200, 500)}
    state_bounds = {"Ca": (0.0, 1.), "T":(2.0E+02, 450.), "Tj":(2.0E+02, 400.)}
    
    ref_state = {("Ca", (0,)): 0.010, ('T', (0,)): 404.60344641487995}
    # ref_state = {("Ca", (0,)): 0.019, ('T', (0,)): 384.33582541449255}
    mod = cstr_rodrigo_dae(1, 1)
    e = NmpcGen_DAE(mod, 2, states, controls,
                    var_bounds=state_bounds,
                    u_bounds=u_bounds,
                    ref_state=ref_state,
                    override_solver_check=True,
                    k_aug_executable='/home/dav0/in_dev_/kslt/WorkshopFraunHofer/day3_caprese/k_aug/bin/k_aug')
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
    e.compute_QR_nmpc(n=-1)
    e.add_terminal_property_nmpc(rhox = 50., rhou = 35.,
                                        simulate_points = 50, state_norm = 0.005)
    e.new_weights_olnmpc(1E-04, 1e+06)
    e.solve_dyn(e.PlantSample, stop_if_nopt=True)
    
    #here to set ic!!!!
    ps = e.PlantSample
    ref = e.curr_state_target
    tend = ps.t[-1]
    now = 7
    pert_Ca = [-0.005, -0.005, 0.04, 0.03, 0.035, 0.032, 0.001, -0.008, -0.005]
    pert_T = [10., -18., 5., 10., -5, 8., -15, 12, 20]
    ps.Ca[(tend,(0,))].value = value(ps.Ca[(tend,(0,))]) + pert_Ca[now]
    ps.T[(tend,(0,))].value = value(ps.T[(tend,(0,))]) + pert_T[now]
    ## ps.Tj[(tend,(0,))].value = ref[('Tj', (0,))] + pert_Tj[now]
    
    e.update_state_real()  # update the current state
    e.update_soi_sp_nmpc()
    e.preparation_phase_nmpc(as_strategy=False, make_prediction=False, plant_state=True)
    # e.load_init_state_nmpc(src_kind="state_dict", state_dict="estimated")
    # e.olnmpc.pprint(filename="new_framework.txt")
    
    stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300,
                            jacobian_regularization_value=1e-04, tag="olnmpc",
                            keepsolve=False, wantparams=False)
    if stat_nmpc != 0:
        sys.exit()
    # e.olnmpc.objfun_nmpc.pprint()
    # e.olnmpc.xmpc_ref_nmpc.display()

    e.print_r_nmpc()
    e.update_u(e.olnmpc)  #: Get the resulting input for k+1
    e.cycleSamPlant(plant_step=True)
    e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
    e.noisy_plant_manager(sigma=0.001, action="apply", update_level=True)
    for i in range(0, 60):
        # if i in [30 * (j * 2) for j in range(0, 100)]:
        if i in [0, 30]:
            ref_state = {("Ca", (0,)): 0.019, ('T', (0,)): 384.33582541449255}
            e.change_setpoint(ref_state=ref_state, keepsolve=True, wantparams=True, tag="sp")
            e.compute_QR_nmpc(n=-1)
            e.add_terminal_property_nmpc(rhox = 100., rhou = 35.,
                                                simulate_points = 50, state_norm = 0.005)
            # e.delete_terminal_properties()
            e.new_weights_olnmpc(1e-04, 1e+06)
        # elif i in [30 * (j * 2 + 1) for j in range(0, 100)]:
        elif i in [15, 45]:
            ref_state = {("Ca", (0,)): 0.01, ('T', (0,)): 404.60344641487995}
            e.change_setpoint(ref_state=ref_state, keepsolve=True, wantparams=True, tag="sp")
            e.compute_QR_nmpc(n=-1)
            e.add_terminal_property_nmpc(rhox = 100., rhou = 35.,
                                                simulate_points = 50, state_norm = 0.005)
            e.new_weights_olnmpc(1e-04, 1e+06)
       
        stat_plant = e.solve_dyn(e.PlantSample, stop_if_nopt=True)
        if stat_plant != 0:
            sys.exit()
        e.update_state_real()  # update the current state
        e.update_soi_sp_nmpc()
        e.preparation_phase_nmpc(as_strategy=False, make_prediction=False, plant_state=True)
        # with open("modify1.txt", "w") as f:
        #     e.olnmpc.pprint(ostream = f)
        stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300,
                                jacobian_regularization_value=1e-04, tag="olnmpc",
                                halt_on_ampl_error = True,
                                keepsolve=False, wantparams=False)
        if stat_nmpc != 0:
            sys.exit()
        
        e.print_r_nmpc()
        e.update_u(e.olnmpc)  #: Get the resulting input for k+1
        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        e.noisy_plant_manager(sigma=0.01, action="apply", update_level=True)

    plt.plot(e.soi_dict[("Ca", (0,))])
    plt.show()
    plt.plot(e.soi_dict[("T", (0,))])
    plt.show()