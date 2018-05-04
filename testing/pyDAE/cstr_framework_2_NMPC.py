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
import sys

__author__ = "David Thierry @dthierry" #: March 2018

def main():
    states = ["Ca", "T", "Tj"]
    controls = ["u1"]
    u_bounds = {"u1": (0, 1000)}
    ref_state = {("Ca", (0,)): 0.010}
    mod = cstr_rodrigo_dae(1, 1)
    e = NmpcGen_DAE(mod, 2, states, controls,
                    u_bounds=u_bounds,
                    ref_state=ref_state,
                    override_solver_check=True)
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
    e.PlantSample.display()
    e.create_suffixes_nmpc()
    e.update_targets_nmpc()
    e.compute_QR_nmpc(n=-1)
    e.new_weights_olnmpc(1E-04, 1e+06)
    e.solve_dyn(e.PlantSample, stop_if_nopt=True)
    e.update_state_real()  # update the current state
    e.update_soi_sp_nmpc()
    e.preparation_phase_nmpc(as_strategy=False, make_prediction=False, plant_state=True)
    # e.load_init_state_nmpc(src_kind="state_dict", state_dict="estimated")
    stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300,
                            jacobian_regularization_value=1e-04, tag="olnmpc",
                            keepsolve=False, wantparams=False)

    e.print_r_nmpc()
    e.update_u(e.olnmpc)  #: Get the resulting input for k+1
    e.cycleSamPlant(plant_step=True)
    e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
    # e.noisy_plant_manager(sigma=0.001, action="apply", update_level=True)
    for i in range(0, 100):
        if i in [30 * (j * 2) for j in range(0, 100)]:
            ref_state = {("Ca", (0,)): 0.019}
            e.change_setpoint(ref_state=ref_state, keepsolve=True, wantparams=True, tag="sp")
            e.compute_QR_nmpc(n=-1)
            e.new_weights_olnmpc(1e-04, 1e+06)
        elif i in [30 * (j * 2 + 1) for j in range(0, 100)]:
            ref_state = {("Ca", (0,)): 0.01}
            e.change_setpoint(ref_state=ref_state, keepsolve=True, wantparams=True, tag="sp")
            e.compute_QR_nmpc(n=-1)
            e.new_weights_olnmpc(1e-04, 1e+06)
        e.solve_dyn(e.PlantSample, stop_if_nopt=True)
        e.update_state_real()  # update the current state
        e.update_soi_sp_nmpc()
        e.preparation_phase_nmpc(as_strategy=False, make_prediction=False, plant_state=True)
        stat_nmpc = e.solve_dyn(e.olnmpc, skip_update=False, max_cpu_time=300,
                                jacobian_regularization_value=1e-04, tag="olnmpc",
                                keepsolve=False, wantparams=False)

        e.print_r_nmpc()
        e.update_u(e.olnmpc)  #: Get the resulting input for k+1
        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        # sys.exit()
        # e.noisy_plant_manager(sigma=0.01, action="apply", update_level=True)

    # print(e.soi_dict[key])
    plt.plot(e.soi_dict[("Ca", (0,))])
    plt.show()