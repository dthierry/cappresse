from __future__ import print_function
from pyomo.environ import *
from nmpc_mhe.dync.MHEGen import MheGen
from nmpc_mhe.mods.distl.dist_col import *
import sys
import itertools, sys

states = ["x", "M"]
x_noisy = ["x", "M"]
u = ["u1", "u2"]
ref_state = {("T", (29,)): 343.15, ("T", (14,)): 361.15}
u_bounds = {"u1": (0.0001, 9.9999e-1), "u2": (0, 1e+08)}

# Known targets 0.38, 0.4, 0.5
# Let's roll with the Temperature of the gas-emulsion, pressure and gas_velocity

y = ["T", "Mv", "Mv1", "Mvn"]

ntrays = 42
y_vars = {"T": [(i,) for i in range(1, ntrays + 1)],
          "Mv": [(i,) for i in range(2, ntrays)],
          "Mv1": [((),)],
          "Mvn": [((),)]}

x_vars = {"x": [(i,) for i in range(1, ntrays + 1)],
          "M": [(i,) for i in range(1, ntrays + 1)]}
nfet = 10
tfe = [i for i in range(1, nfet + 1)]
# States -- (5 * 3 + 6) * fe_x * cp_x.
# For fe_x = 5 and cp_x = 3 we will have 315 differential-states.

e = MheGen(d_mod=DistDiehlNegrete,
           y=y,
           x_noisy=x_noisy,
           y_vars=y_vars,
           x_vars=x_vars,
           states=states,
           u=u,
           ref_state=ref_state,
           u_bounds=u_bounds,
           diag_QR=True,
           nfe_t=nfet)

e.load_iguess_ss()
e.solve_ss()
e.load_d_s(e.d1)
e.solve_d(e.d1)

q_cov = {}
for i in range(1, nfet):
    for j in range(1, ntrays + 1):
        q_cov[("x", (j,)), ("x", (j,)), i] = 1e-05
        q_cov[("M", (j,)), ("M", (j,)), i] = 1

m_cov = {}
for i in range(1, nfet + 1):
    for j in range(1, ntrays + 1):
        m_cov[("T", (j,)), ("T", (j,)), i] = 6.25e-2
    for j in range(2, 42):
        m_cov[("Mv", (j,)), ("Mv", (j,)), i] = 10e-08
    m_cov[("Mv1", ((),)), ("Mv1", ((),)), i] = 10e-08
    m_cov[("Mvn", ((),)), ("Mvn", ((),)), i] = 10e-08

u_cov = {}
for i in tfe:
    u_cov["u1", i] = 7.72700925775773761472464684629813E-01 * 0.01
    u_cov["u2", i] = 1.78604740940007800236344337463379E+06 * 0.01


e.set_covariance_meas(m_cov)
e.set_covariance_disturb(q_cov)
e.set_covariance_u(u_cov)

# Preparation phase
e.init_lsmhe_prep(e.d1)

e.shift_mhe()
dum = e.d_mod(1, e.ncp_t, _t=e.hi_t)

e.init_step_mhe(dum, e.nfe_t)
e.solve_d(e.lsmhe, skip_update=False)  #: Pre-loaded mhe solve
# e.lsmhe.pprint(filename="somefile.model")

e.create_rh_sfx()  #: Reduced hessian computation

e.check_active_bound_noisy()
e.load_covariance_prior()
e.set_state_covariance()

e.regen_objective_fun()  #: Regen erate the obj fun
e.deact_icc_mhe()  #: Remove the initial conditions

e.set_prior_state_from_prior_mhe()  #: Update prior-state
e.find_target_ss()  #: Compute target-steady state (beforehand)
# For ideal nmpc
for i in range(1, 40):
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
