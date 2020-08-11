#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

from pyomo.environ import *
from sample_mods.cstr_yeonsoo.cstr_yeonsoo import cstr_yeonsoo_dae
from nmpc_mhe.pyomo_dae.NMPCGen_pyDAE import NmpcGen_DAE
from nmpc_mhe.aux.utils import load_iguess
from nmpc_mhe.aux.utils import reconcile_nvars_mequations
import matplotlib.pyplot as plt
import sys, os

__author__ = "Kuan-Han Lin @kuanhanl" #: Jul 2020

def main():
    states = ["z1", "z2"]
    controls = ["d_u1", "d_u2"]   
    u_bounds = {"d_u1":(0,2500), "d_u2":(1,40)}
    state_bounds = {"z1":(0,1), "z2":(0,1)}
    
    ref_state = {("z1", (0,)):0.1768, ("z2",(0,)): 0.7083}
    mod = cstr_yeonsoo_dae(1, 1)
    Ns_nmpc = 3 #declare how many sampling times are needed for amsNMPC
    e = NmpcGen_DAE(mod, 1., states, controls,
                    var_bounds=state_bounds,
                    u_bounds=u_bounds,
                    ref_state=ref_state,
                    nfe_tnmpc = 50,
                    Ns_amsnmpc = Ns_nmpc,
                    override_solver_check=True,
                    k_aug_executable='/home/dav0/in_dev_/kslt/WorkshopFraunHofer/day3_caprese/k_aug/bin/k_aug',
                    dot_driver_executable='/home/dav0/in_dev_/kslt/WorkshopFraunHofer/day3_caprese/k_aug/dot_sens')
    
    e.get_state_vars()
    e.load_iguess_steady()
    e.create_nmpc()
    reconcile_nvars_mequations(e.olnmpc)
    e.solve_dyn(e.PlantSample)
    e.find_target_ss()

    e.create_suffixes_amsnmpc()
    e.update_targets_nmpc()
    e.compute_QR_nmpc(define_by_user = True, Q_nmpc = 10**6, R_nmpc = 2*10**(-3))
    e.new_weights_olnmpc(1., 1.)
    
    e.solve_dyn(e.PlantSample, stop_if_nopt=True)
    e.update_state_real()  # update the current state
    e.update_soi_sp_nmpc()
    e.print_r_nmpc()
    
    ref_info = {0: {("z1", (0,)):0.1768, ("z2",(0,)): 0.7083},
                     51: {("z1", (0,)):0.1768*1.1, ("z2",(0,)): 0.7083*0.9},
                     101: {("z1", (0,)):0.1768*0.9, ("z2",(0,)): 0.7083*1.1}}
    
    e.setup_sp_profile(ref_info)
    e.compute_QR_nmpc(define_by_user = True, Q_nmpc = 10**6, R_nmpc = 2*10**(-3), target_step = 0)
    
    #Prior preparation for olnmpc
    e.preparation_phase_nmpc(plant_state=True) #Don't call ams_strategy=True since real ic is used in prior preparation KH.L
    # e.load_init_state_nmpc(src_kind="state_dict", state_dict="real") #confirmed to be redundant. It was inclueded in the previous step.
    stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300,
                            jacobian_regularization_value=1e-04, tag="olnmpc",
                            keepsolve=False, wantparams=False)
    
    e.setup_info_for_extended_sensitivity()

    e.load_ds_int_and_dj()

    e.update_u(e.olnmpc)  #since ic in prior preparation is real, use this function not update_u_amsnmpc
    e.cycleSamPlant(plant_step=True)
    e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
    
    e.preparation_phase_nmpc(ams_strategy = True, plant_state = True)
    
    for i in range(0,150):
            
        #Plant
        e.solve_dyn(e.PlantSample, stop_if_nopt=True)
        e.update_state_real()  # update the current state
        e.update_soi_sp_nmpc() #update current soi from PlantSample and sp from SteadyRef2
        e.print_r_nmpc()
        
        #Online (su = sensitivity update)
        if i % Ns_nmpc == 0:
            #extended su NMPC,d1, this u is for i=1,4,7...
            e.sens_dot_amsnmpc(stage = 1, src = "real")
            e.update_u_amsnmpc()
        elif i % Ns_nmpc == 1:
            #extended su NMPC,d2, this u is for i=2,5,8...
            e.sens_dot_amsnmpc(stage = 2, src = "real")
            e.update_u_amsnmpc()
        elif i % Ns_nmpc == 2:
            #regular su NMPC,ds, this u is for i=3,6,9.... (not for i=0)
            e.sens_dot_amsnmpc(stage = 0, src = "real")
            e.update_u_amsnmpc()
            e.preparation_phase_nmpc(ams_strategy = True, plant_state = True) #predict the next Nsth step and initialize olnmpc
        
        #Offline
        if i % Ns_nmpc == 0:
            e.compute_QR_nmpc(define_by_user = True, Q_nmpc = 10**6, R_nmpc = 2*10**(-3), target_step = i+Ns_nmpc)    
            stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300, tag="olnmpc")
            if stat_nmpc != 0:
                sys.exit()
            
            e.setup_info_for_extended_sensitivity()

        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        e.noisy_plant_manager(sigma=0.0015, action="apply", update_level=True)
    
    plt.plot(e.soi_dict[("z1",(0,))])
    plt.show()
    plt.plot(e.soi_dict[("z2",(0,))])
    plt.show()
    return e

if __name__ == '__main__':
    e = main()