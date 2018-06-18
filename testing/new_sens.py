#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from nmpc_mhe.dync.MHEGenv2 import MheGen
from sample_mods.bfb.nob5_hi_t import bfb_dae
import itertools, sys
from subprocess import call


"""Testing the new preparation phases with ideal strategies"""

def main():
    call(["rm", "*.txt"])  #: Self cleanup (hopefuly)
    x_weight = 1E-04
    u_weight = 1E+06
    states = ["Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
    x_noisy = ["Hgc", "Nsc", "Hsc", "Hge", "Nse", "Hse"]
    u = ["u1"]
    u_bounds = {"u1":(162.183495794 * 0.0005, 162.183495794 * 10000)}

    # ref_state0 = {("c_capture", ((),)): 0.55739}
    ref_state0 = {("c_capture", ((),)): 0.657397366523 - 0.05}
    ref_state1 = {("c_capture", ((),)): 0.657397366523}
    ref_state2 = {("c_capture", ((),)): 0.657397366523 - 0.05 + 0.020}
    ref_state3 = {("c_capture", ((),)): 0.657397366523 - 0.020}
    ref_state = ref_state0
    dx = {}
    for k in ref_state1.keys():
        dx[k] = ref_state0[k] - ref_state1[k]


    nfe_mhe = 10
    y = ["Tgb", "vg"]
    nfet = 10
    ncpx = 3
    nfex = 5
    tfe = [i for i in range(0, nfe_mhe)]
    lfe = [i for i in range(0, nfex)]
    lcp = [i for i in range(1, ncpx + 1)]
    lc = ['c', 'h', 'n']

    y_vars = {
        "Tgb": [i for i in itertools.product(lfe, lcp)],
        "vg": [i for i in itertools.product(lfe, lcp)]
        }
    # x_vars = dict()
    x_vars = {
              # "Nge": [i for i in itertools.product(lfe, lcp, lc)],
              # "Hge": [i for i in itertools.product(lfe, lcp)],
              "Nsc": [i for i in itertools.product(lfe, lcp, lc)],
              "Hsc": [i for i in itertools.product(lfe, lcp)],
              "Nse": [i for i in itertools.product(lfe, lcp, lc)],
              "Hse": [i for i in itertools.product(lfe, lcp)],
              "Hgc": [i for i in itertools.product(lfe, lcp)],
              "Hge": [i for i in itertools.product(lfe, lcp)],
              # "mom": [i for i in itertools.product(lfe, lcp)]
              }

    # States -- (5 * 3 + 6) * fe_x * cp_x.
    # For fe_x = 5 and cp_x = 3 we will have 315 differential-states.
    #: 1600 was proven to be solveable
    e = MheGen(bfb_dae, 1200/nfe_mhe, states, u, x_noisy, x_vars, y, y_vars,
               nfe_tmhe=nfe_mhe, ncp_tmhe=1,
               nfe_tnmpc=nfe_mhe, ncp_tnmpc=1,
               ref_state=ref_state, u_bounds=u_bounds,
               nfe_t=nfe_mhe, ncp_t=1,
               k_aug_executable="/home/dav0/devzone/k_aug/bin/k_aug",
               dot_driver_executable="/home/dav0/devzone/k_aug/src/k_aug/dot_driver/dot_driver"
               # k_aug_executable="/home/dmolinat/k_aug/bin/k_aug",
               # dot_driver_executable="/home/dmolinat/k_aug/src/k_aug/dot_driver/dot_driver"
               )

    e.load_solfile(e.SteadyRef, "ref_ss.sol")  #: Loads solfile snap
    results = e.ipopt.solve(e.SteadyRef, tee=True, load_solutions=True, report_timing=True)

    e.get_state_vars(skip_solve=True)
    e.SteadyRef.report_zL(filename="mult_ss")
    e.load_d_s(e.PlantSample)
    e.PlantSample.create_bounds()
    e.solve_dyn(e.PlantSample)

    q_cov = {}
    for i in tfe:
        for j in itertools.product(lfe, lcp, lc):
            q_cov[("Nse", j), ("Nse", j), i] = 7525.81478168 * 0.005
            q_cov[("Nsc", j), ("Nsc", j), i] = 117.650089456 * 0.005
    #             q_cov[("Nse", j), ("Nse", j), i] = 735.706082714 * 0.005
    for i in tfe:
        for j in itertools.product(lfe, lcp):
            # q_cov[("Hge", j), ("Hge", j), i] = 2194.25390583 * 0.005
            q_cov[("Hse", j), ("Hse", j), i] = 731143.716603 * 0.005
            q_cov[("Hsc", j), ("Hsc", j), i] = 16668.3312216 * 0.005
            q_cov[("Hge", j), ("Hge", j), i] = 2166.86838591 * 0.005
            q_cov[("Hgc", j), ("Hgc", j), i] = 47.7911012193 * 0.005
            # q_cov[("mom", j), ("mom", j), i] = 1.14042251669 * 0.005

    # for i in lfe:
    #     for j in [(1,1, 'c'), (5,3, 'c')]:
    #         m_cov[("yb", j), ("yb", j), i] = 1e-04

    u_cov = {}
    for i in [i for i in range(0, nfe_mhe)]:
        u_cov["u1", i] = 162.183495794 * 0.005

    m_cov = {}
    for i in tfe:
        for j in itertools.product(lfe, lcp):
            m_cov[("Tgb", j), ("Tgb", j), i] = 40 * 0.005
            m_cov[("vg", j), ("vg", j), i] = 0.902649386907 * 0.005

    e.set_covariance_meas(m_cov)
    e.set_covariance_disturb(q_cov)
    e.set_covariance_u(u_cov)
    e.create_rh_sfx()  #: Reduced hessian computation


    e.init_lsmhe_prep(e.PlantSample)
    e.shift_mhe()
    e.init_step_mhe()
    e.lsmhe.pprint(filename="blyat.txt")
    e.solve_dyn(e.lsmhe,
                skip_update=False,
                max_cpu_time=1200,
                ma57_pre_alloc=5, tag="lsmhe")  #: Pre-loaded mhe solve

    e.check_active_bound_noisy()
    e.load_covariance_prior()
    e.set_state_covariance()

    e.regen_objective_fun()  #: Regen erate the obj fun
    e.deact_icc_mhe()  #: Remove the initial conditions

    e.set_prior_state_from_prior_mhe()  #: Update prior-state
    e.find_target_ss()  #: Compute target-steady state (beforehand)

    #: Create NMPC
    e.create_nmpc()
    e.create_suffixes_nmpc()
    e.update_targets_nmpc()
    e.compute_QR_nmpc(n=-1)
    e.new_weights_olnmpc(x_weight, u_weight)
    e.solve_dyn(e.PlantSample, stop_if_nopt=True)
    # isnap = [i*50 for i in range(1, 25)]
    isnap = [i*25 for i in range(2, 30)]
    j = 1
    ii = 0
    cw_u = 1e+06
    weights_ramp = {}
    f = open("errors_mhe", "w")
    f.close()
    f = open("errors_nmpc", "w")
    f.close()
    stat_mhe = 0
    stat_nmpc = 0
    for i in range(1, 1000):
        if i in isnap:
            keepsolve=False
            wantparams=False
        else:
            keepsolve=False
            wantparams=False
        if i == 250:
            j = 1
            ref_state = ref_state1
            e.change_setpoint(ref_state=ref_state, keepsolve=False, wantparams=False, tag="sp", iter_max=700)
            e.compute_QR_nmpc(n=-1)
        elif i == 500:
            j = 1
            ref_state = ref_state2
            e.change_setpoint(ref_state=ref_state, keepsolve=False, wantparams=False, tag="sp", iter_max=700)
            e.compute_QR_nmpc(n=-1)
        elif i == 750:
            j = 1
            ref_state = ref_state3
            e.change_setpoint(ref_state=ref_state, keepsolve=False, wantparams=False, tag="sp", iter_max=700)
            e.compute_QR_nmpc(n=-1)


        stat = e.solve_dyn(e.PlantSample, stop_if_nopt=False, tag="plant", keepsolve=keepsolve, wantparams=wantparams)
        if stat == 1:
            e.noisy_plant_manager(action="remove")
            e.solve_dyn(e.PlantSample, stop_if_nopt=True, tag="plant", keepsolve=keepsolve,
                        wantparams=wantparams)  #: Try again (without noise)

        e.update_state_real()  # update the current state
        e.update_soi_sp_nmpc()
        e.update_noise_meas(m_cov)  #: the noise it is not being added
        e.update_measurement()
        e.compute_y_offset()  #: Get the offset for y

        #: !!!!
        #: Do dot sens
        #: !!!!
        #: Note that any change in the model at this point is irrelevant for sens_update

        if stat_nmpc == 0:
            if i > 1:
                e.sens_dot_mhe()  #: Do sensitivity update for mhe
            e.update_state_mhe(as_nmpc_mhe_strategy=True)  #: Get offset for x
            if i > 1:
                e.sens_dot_nmpc()
            e.update_u(e.olnmpc)  #: Get the resulting input for k+1
        else:
            e.update_u(e.SteadyRef2)  #: Default

        e.print_r_mhe()
        e.print_r_dyn()

        e.preparation_phase_mhe(as_strategy=True)


        try:
            stat_mhe = e.solve_dyn(e.lsmhe,
                               skip_update=False, iter_max=500,
                               jacobian_regularization_value=1e-01,
                               max_cpu_time=600,
                               tag="lsmhe",
                               keepsolve=False,
                               wantparams=False)
        except ValueError:
            try:
                stat_mhe = e.solve_dyn(e.lsmhe,
                                   skip_update=False, iter_max=500,
                                   jacobian_regularization_value=1e-01,
                                   max_cpu_time=1200,
                                   tag="lsmhe",
                                   keepsolve=False,
                                   wantparams=False)
            except ValueError:
                stat_mhe = 299


        if stat_mhe != 0:
            f = open("errors_mhe", "a")
            info_s = "iter\t" + str(i) + "\tstat\t" + str(stat_mhe) + "\n"
            f.write(info_s)
            f.close()
        else:
            e.sens_k_aug_mhe()  # sensitivity matrix for mhe
            #: Prior-phase and arrival cost
            e.update_state_mhe()  #: get the state from mhe
            e.prior_phase()

        e.print_r_mhe()
        e.print_r_dyn()
        if stat_mhe != 0:
            stat_nmpc = 500 #: MHE failed
        else:
            #: Control NMPC
            e.preparation_phase_nmpc(as_strategy=False, make_prediction=False)
            try: #: First try
                stat_nmpc = e.solve_dyn(e.olnmpc,
                                        skip_update=False,
                                        max_cpu_time=300,
                                        tag="olnmpc")
            except ValueError:
                e.preparation_phase_nmpc(as_strategy=False, make_prediction=False)
                try: #: Second Try
                    stat_nmpc = e.solve_dyn(e.olnmpc,
                                            skip_update=False,
                                            max_cpu_time=300,
                                            jacobian_regularization_value=1e-04,
                                            tag="olnmpc")
                except ValueError:
                    stat_nmpc = 299

        e.print_r_nmpc()

        if stat_nmpc == 0:
            e.sens_k_aug_nmpc()  # sensitivity matrix for nmpc
        else:
            f = open("errors_nmpc", "a")
            info_s = "iter\t" + str(i) + "\tstat\t" + str(stat_nmpc) + "\n"
            f.write(info_s)
            f.close()

        #: Plant cycle
        e.cycleSamPlant(plant_step=True)
        e.plant_uinject(e.PlantSample, src_kind="dict", skip_homotopy=True)
        e.noisy_plant_manager(sigma=0.0001, action="apply", update_level=True)
        #: 0.001 is a good level


if __name__ == '__main__':
    e = main()
