#!/usr/bin/env python
# -*- coding: utf-8


from __future__ import division
from __future__ import print_function

from nmpc_mhe.pyomo_dae.MHEGen_pyDAE import MheGen_DAE
from sample_mods.distc_pyDAE.distcpydaemod import mod
from nmpc_mhe.aux.utils import load_iguess


__author__ = "David Thierry"


def main():
    states = ["x", "M"]
    state_bounds = {"x": (0, 1), "M":(0, None)}
    measurements = ["T", "Mv", "Mv1", "Mvn"]
    controls = ["u1", "u2"]
    u_bounds = {"u1": (0000.1, 99.999), "u2": (0, 1e+08)}
    ref_state = {("T", (29,)): 343.15, ("T", (14,)): 361.15}

    e = MheGen_DAE(mod, 100, states, controls, states, measurements,
                   u_bounds=u_bounds,
                   ref_state=ref_state,
                   override_solver_check=True,
                   var_bounds=state_bounds,
                   k_aug_executable="")
    Q = {}
    U = {}
    R = {}
    Q["x"] = 1e-05
    Q["M"] = 0.5
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
    e.load_iguess_steady()
    load_iguess(e.SteadyRef, e.PlantSample, 0, 0)

    # reconcile_nvars_mequations(e.lsmhe)
    e.solve_dyn(e.PlantSample)

    e.init_lsmhe_prep(e.PlantSample)
    e.shift_mhe()
    e.init_step_mhe()
    e.solve_dyn(e.lsmhe,
                skip_update=False,
                max_cpu_time=600,
                ma57_pre_alloc=5, tag="lsmhe")  #: Pre-loaded mhe solve

    e.prior_phase()
    e.deact_icc_mhe()  #: Remove the initial conditions

    e.find_target_ss()

    #: NMPC
    e.create_nmpc()
    e.create_suffixes_nmpc()
    e.update_targets_nmpc()
    e.compute_QR_nmpc(n=-1)
    e.new_weights_olnmpc(1E-04, 1e+06)


if __name__ == '__main__':
    main()