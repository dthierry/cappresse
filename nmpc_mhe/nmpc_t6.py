from __future__ import division
from __future__ import print_function
from pyomo.environ import *
from nmpc_mhe.dync.NMPCGen import NmpcGen
from nmpc_mhe.mods.bfb.bfb_abs7momdt_ht2 import bfb_dae
from snap_shot import snap
import sys
from numpy.random import normal as npm
from pyutilib.common._exceptions import ApplicationError
states = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "mom"]
u = ["u1"]
u_bounds = {"u1":(162.183495794 * 0.0005, 162.183495794 * 10000)}
ref_state = {("c_capture", ((),)): 0.63}
c = NmpcGen(d_mod=bfb_dae,
            u=u, states=states,
            ref_state=ref_state,
            u_bounds=u_bounds,
            nfe_t=5,
            ncp_t=1,
            _t=100)

c.ss.dref = snap

c.load_iguess_ss()
# sys.exit()
# c.ss.lydot.display()
# sys.exit()
c.solve_ss()
c.solve_d(c.ss)
c.ss.write_nl()
# sys.exit()
c.load_d_s(c.d1)
c.d1.create_bounds()
c.solve_d(c.d1, halt_on_ampl_error=True)
c.update_state_real()  # update the current state

c.find_target_ss()
with open("mult_boundsss2.txt", "w") as f:
    c.ss2.ipopt_zL_out.pprint(ostream=f)
    f.close()
c.create_nmpc()
c.update_targets_nmpc()
c.compute_QR_nmpc(n=-1)
c.new_weights_olnmpc(100, 1e+03)


# q_cov = {}
# for i in range(1, nfet):
#     for j in range(1, ntrays + 1):
#         q_cov[("x", (j,)), ("x", (j,)), i] = 1e-05
#         q_cov[("M", (j,)), ("M", (j,)), i] = 0.5
#
# m_cov = {}
# for i in range(1, nfet + 1):
#     for j in range(1, ntrays + 1):
#         m_cov[("T", (j,)), ("T", (j,)), i] = 6.25e-2
#     for j in range(2, 42):
#         m_cov[("Mv", (j,)), ("Mv", (j,)), i] = 10e-08
#     m_cov[("Mv1", ((),)), ("Mv1", ((),)), i] = 10e-08
#     m_cov[("Mvn", ((),)), ("Mvn", ((),)), i] = 10e-08
#
# u_cov = {}
# for i in tfe:
#     u_cov["u1", i] = 7.72700925775773761472464684629813E-01 * 0.01
#     u_cov["u2", i] = 1.78604740940007800236344337463379E+06 * 0.001
#
#
# c.set_covariance_meas(m_cov)
# c.set_covariance_disturb(q_cov)
# c.set_covariance_u(u_cov)
#
# # Preparation phase
# c.init_lsmhe_prep(c.d1)
#
# c.shift_mhe()
dum = c.d_mod(1, c.ncp_t, _t=c.hi_t)
#
# c.init_step_mhe(dum, c.nfe_t)
# c.solve_d(c.lsmhe, skip_update=False)  #: Pre-loaded mhe solve
#
# c.create_rh_sfx()  #: Reduced hessian computation
#
# c.check_active_bound_noisy()
# c.load_covariance_prior()
# c.set_state_covariance()
#
# c.regen_objective_fun()  #: Regen erate the obj fun
# c.deact_icc_mhe()  #: Remove the initial conditions
#
# c.set_prior_state_from_prior_mhe()  #: Update prior-state
c.find_target_ss()  #: Compute target-steady state (beforehand)


# For ideal nmpc
for i in range(1, 1000):
    try:
        statd1 = c.solve_d(c.d1, stop_if_nopt=False, halt_on_ampl_error=True)
    except ApplicationError:
        print("Ipopt FAIL", file=sys.stderr)
        c.d1.write_nl()
        c.d1.snap_shot(filename="baddie.py")
        c.d1.report_zL(filename="bad_bounds")
        sys.exit()
    if statd1 != 0:
        c.d1.write_nl(name="d1fail.nl")
        statd1 = c.solve_d(c.d1, stop_if_nopt=True, halt_on_ampl_error=True, iter_max=6000)
    c.update_state_real()  # update the current state
    c.update_soi_sp_nmpc()

    # c.update_noise_meas(c.d1, m_cov)
    # c.load_input_mhe("mod", src=c.d1, fe=c.nfe_t)  #: The inputs must coincide



    # c.patch_meas_mhe(c.nfe_t, src=c.d1, noisy=True)  #: Get the measurement
    # c.compute_y_offset()

    # c.init_step_mhe(dum, c.nfe_t)  # Initialize next time-slot
    # c.update_state_mhe()

    # Prior-Covariance stuff
    # c.check_active_bound_noisy()
    # c.load_covariance_prior()
    # c.set_state_covariance()
    # c.regen_objective_fun()
    # Update prior-state
    # c.set_prior_state_from_prior_mhe()
    #
    # c.print_r_mhe()

    # Compute the controls
    c.initialize_olnmpc(c.d1, "real")
    c.load_init_state_nmpc(src_kind="state_dict", state_dict="real")  # for good measure
    # if i == 5:
    #     with open("somefilc.txt", "w") as f:
    #         c.olnmpc.R_nmpc.display(ostream=f)
    #         c.olnmpc.Q_nmpc.display(ostream=f)
    #         f.close()
    c.d1.report_zL(filename="whatevs.txt")
    stat_nmpc = c.solve_d(c.olnmpc, skip_update=False, max_cpu_time=700)
    if stat_nmpc != 0:
        c.olnmpc.write_nl(name='nmpc_failed.nl')
        stat_nmpc = c.solve_d(c.olnmpc, stop_if_nopt=True, skip_update=False, max_cpu_time=70000)
    c.update_u(c.olnmpc)
    c.print_r_nmpc()

    # c.shift_mhe()
    # c.shift_measurement_input_mhe()

    c.cycle_ics(plant_step=True)
    c.plant_input_gen(c.d1, src_kind="dict")

