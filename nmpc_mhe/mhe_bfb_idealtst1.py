from __future__ import print_function
from pyomo.environ import *
from pyomo.opt import ProblemFormat
from nmpc_mhe.dync.MHEGen import MheGen
from nmpc_mhe.mods.bfb.bfb_abs1 import *
import sys, os
import itertools, sys

states = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "Ws"]
x_noisy = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "Ws"]
u = ["u1", "u2"]
u_bounds = {"u1":(0.0001, 99.9999), "u2":(0.0001, 99.99)}
ref_state = {("c_capture", ((),)): 0.5}
# Known targets 0.38, 0.4, 0.5
# Let's roll with the Temperature of the gas-emulsion, pressure and gas_velocity

y = ["Tge", "P"]
nfet = 5
ncpx = 3
nfex = 5
tfe = [i for i in range(1, nfet + 1)]
lfe = [i for i in range(1, nfex + 1)]
lcp = [i for i in range(1, ncpx + 1)]
lc = ['c', 'h', 'n']

y_vars = {"Tge": [i for i in itertools.product(lfe, lcp)],
          "P": [i for i in itertools.product(lfe, lcp)],
          # "vg": [i for i in itertools.product(lfe, lcp)]
          }

x_vars = {"Ngb": [i for i in itertools.product(lfe, lcp, lc)],
          "Hgb": [i for i in itertools.product(lfe, lcp)],
          "Ngc": [i for i in itertools.product(lfe, lcp, lc)],
          "Hgc": [i for i in itertools.product(lfe, lcp)],
          "Nsc": [i for i in itertools.product(lfe, lcp, lc)],
          "Hsc": [i for i in itertools.product(lfe, lcp)],
          "Nge": [i for i in itertools.product(lfe, lcp, lc)],
          "Hge": [i for i in itertools.product(lfe, lcp)],
          "Nse": [i for i in itertools.product(lfe, lcp, lc)],
          "Hse": [i for i in itertools.product(lfe, lcp)],
          "Ws": [i for i in itertools.product(lfe, lcp)]}

# States -- (5 * 3 + 6) * fe_x * cp_x.
# For fe_x = 5 and cp_x = 3 we will have 315 differential-states.

e = MheGen(d_mod=bfb_dae,
           y=y,
           x_noisy=x_noisy,
           y_vars=y_vars,
           x_vars=x_vars,
           states=states,
           u=u,
           ref_state=ref_state,
           u_bounds=u_bounds,
           diag_QR=True)

e.load_iguess_ss()
# sys.exit()
e.ss.create_bounds()
e.solve_ss()
e.load_d_s(e.d1)
e.d1.create_bounds()
e.solve_d(e.d1)

q_cov = {}
for i in tfe:
    if i < nfet:
        for j in itertools.product(lfe, lcp, lc):
            # "Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "Ws"
            # q_cov[("Ngb", j), ("Ngb", j), i] = 1.e-05
            # q_cov[("Ngc", j), ("Ngc", j), i] = 1.e-05
            # q_cov[("Nsc", j), ("Nsc", j), i] = 10.0
            # q_cov[("Nge", j), ("Nge", j), i] = 0.01
            # q_cov[("Nse", j), ("Nse", j), i] = 10.0
            q_cov[("Ngb", j), ("Ngb", j), i] = 100000.
            q_cov[("Ngc", j), ("Ngc", j), i] = 100000.
            q_cov[("Nsc", j), ("Nsc", j), i] = 100000.
            q_cov[("Nge", j), ("Nge", j), i] = 100000.
            q_cov[("Nse", j), ("Nse", j), i] = 100000.
for i in tfe:
    if i < nfet:
        for j in itertools.product(lfe, lcp):
            # q_cov[("Hgb", j), ("Hgb", j), i] = 10.
            # q_cov[("Hgc", j), ("Hgc", j), i] = 5.
            # q_cov[("Hsc", j), ("Hsc", j), i] = 10.
            # q_cov[("Hge", j), ("Hge", j), i] = 10.
            # q_cov[("Hse", j), ("Hse", j), i] = 100.
            # q_cov[("Ws", j), ("Ws", j), i] = 0.1
            q_cov[("Hgb", j), ("Hgb", j), i] = 100000.
            q_cov[("Hgc", j), ("Hgc", j), i] = 100000.
            q_cov[("Hsc", j), ("Hsc", j), i] = 100000.
            q_cov[("Hge", j), ("Hge", j), i] = 100000.
            q_cov[("Hse", j), ("Hse", j), i] = 100000.
            q_cov[("Ws", j), ("Ws", j), i] = 100000.


m_cov = {}
for i in lfe:
    for j in itertools.product(lfe, lcp):
        m_cov[("Tge", j), ("Tge", j), i] = 10.0
        m_cov[("P", j), ("P", j), i] = 1e-03
        # m_cov[("vg", j), ("vg", j), i] = 1e-05


u_cov = {}
for i in tfe:
    u_cov["u1", i] = 10
    u_cov["u2", i] = 10


e.set_covariance_meas(m_cov)
e.set_covariance_disturb(q_cov)
e.set_covariance_u(u_cov)
e.create_rh_sfx()  #: Reduced hessian computation

# Preparation phase
e.init_lsmhe_prep(e.d1)

e.shift_mhe()
dum = e.d_mod(1, e.ncp_t, _t=e.hi_t)


e.init_step_mhe(dum, e.nfe_t)

e.deb_alg_sys_dyn()
tst = e.solve_d(e.d1, skip_update=False)  #: Pre-loaded mhe solve
e.d1.write(filename="heorqie.nl",
              format=ProblemFormat.nl,
              io_options={"symbolic_solver_labels": True})
print(os.getcwd())
sys.exit()


tst = e.solve_d(e.lsmhe, skip_update=False, iter_max=10)  #: Pre-loaded mhe solve
# if tst != 0:
#     e.k_aug.solve(e.lsmhe, tee=True)
#     sys.exit()
# e.lsmhe.pprint(filename="somefile.model")

e.check_active_bound_noisy()
e.load_covariance_prior()
e.set_state_covariance()

e.regen_objective_fun()  #: Regen erate the obj fun
e.deact_icc_mhe()  #: Remove the initial conditions

e.set_prior_state_from_prior_mhe()  #: Update prior-state
e.find_target_ss()  #: Compute target-steady state (beforehand)
# For ideal nmpc
for i in range(1, 15):
    print(str(i) + "--"*20, file=sys.stderr)
    print("*"*100)

    if i == 3:
        e.plant_input_gen(e.d1, "mod", src=e.ss2)

    e.solve_d(e.d1)
    if i == 3:
        e.d1.display(filename="plant.txt")
    e.update_noise_meas(e.d1, m_cov)
    e.load_input_mhe("mod", src=e.d1, fe=e.nfe_t)  #: The inputs must coincide
    some_val = value(e.lsmhe.u1[e.nfe_t]) - value(e.d1.u1[1])
    print(some_val, "Value of the offset")
    e.patch_meas_mhe(e.nfe_t, src=e.d1, noisy=True)  #: Get the measurement
    e.compute_y_offset()

    e.init_step_mhe(dum, e.nfe_t)  # Initialize next time-slot
    with open("file_src.txt", "w") as f:
        e.d1.u1.display(ostream=f)
        e.d1.u2.display(ostream=f)
        f.close()

    with open("file1.txt", "w") as f:
        e.lsmhe.u1.display(ostream=f)
        e.lsmhe.u2.display(ostream=f)
        f.close()
    stat = e.solve_d(e.lsmhe, skip_update=False)
    if stat == 1:
        stat = e.solve_d(e.lsmhe, skip_update=False, iter_max=250, stop_if_nopt=True)


    # Prior-Covariance stuff
    e.check_active_bound_noisy()
    e.load_covariance_prior()
    e.set_state_covariance()
    e.regen_objective_fun()
    # Update prior-state
    e.set_prior_state_from_prior_mhe()

    e.print_r_mhe()

    # Compute the controls

    e.shift_mhe()
    e.shift_measurement_input_mhe()



    e.cycle_ics(plant_step=True)
