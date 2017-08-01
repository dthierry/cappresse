from __future__ import print_function
from pyomo.environ import *
from nmpc_mhe.dync.MHEGen import MheGen
from nmpc_mhe.mods.distl.dist_col import DistDiehlNegrete
import sys

y = ["T", "Mv", "Mv1", "Mvn"]
u = ["u1", "u2"]
ref_state = {("T", (29,)): 343.15, ("T", (14,)): 361.15}
u_bounds = {"u1":(0.0001, 9.9999e-1), "u2":(0, 1e+08)}

states = ["x", "M"]
x_noisy = ["x", "M"]

ntrays = 42
y_vars = {"T": [(i,) for i in range(1, ntrays + 1)],
          "Mv": [(i,) for i in range(2, ntrays)],
          "Mv1": [((),)],
          "Mvn": [((),)]}

x_vars = {"x": [(i,) for i in range(1, ntrays + 1)],
          "M": [(i,) for i in range(1, ntrays + 1)]}

e = MheGen(d_mod=DistDiehlNegrete,
           y=y,
           x_noisy=x_noisy,
           y_vars=y_vars,
           x_vars=x_vars,
           states=states,
           u=u,
           ref_state=ref_state,
           u_bounds=u_bounds)

e.solve_ss()
e.load_d_s(e.d1)
e.solve_d(e.d1)


q_cov = {}
for i in range(1, 5):
    for j in range(1, 43):
        q_cov[("x", (j,)), ("x", (j,)), i] = 1e-05
        q_cov[("M", (j,)), ("M", (j,)), i] = 1

m_cov = {}
for i in range(1, 5 + 1):
    for j in range(1, ntrays + 1):
        m_cov[("T", (j,)), ("T", (j,)), i] = 6.25e-2
    for j in range(2, 42):
        m_cov[("Mv", (j,)), ("Mv", (j,)), i] = 10e-08
    m_cov[("Mv1", ((),)), ("Mv1", ((),)), i] = 10e-08
    m_cov[("Mvn", ((),)), ("Mvn", ((),)), i] = 10e-08

e.set_covariance_meas(m_cov)
e.set_covariance_disturb(q_cov)
e.init_lsmhe_prep(e.d1)
e.shift_mhe()
dum = e.d_mod(1, e.ncp_t, _t=e.hi_t)

e.init_step_mhe(dum, e.nfe_t)
e.solve_d(e.lsmhe, skip_update=False)

e.create_rh_sfx()

e.check_active_bound_noisy()
e.load_covariance_prior()
e.set_state_covariance()

e.regen_objective_fun()
e.deact_icc_mhe()

e.set_prior_state_from_prior_mhe()
e.find_target_ss()




for i in range(1, 2):
    print(str(i) + "--"*20, file=sys.stderr)
    print(i)
    print("*"*100)


    if i == 10:
        e.plant_input_gen(e.ss2, 1)
        # e.lsmhe.pprint(filename="somefile.txt")


    e.solve_d(e.d1)
    e.update_noise_meas(e.d1, m_cov)

    e.patch_meas_mhe(e.nfe_t, src=e.d1, noisy=True)
    e.load_inputsmhe(src=e.d1, fe=e.nfe_t)

    e.compute_y_offset()

    e.create_sens_suffix()
    print("testing \n\n\n" * 4)

    e.sens_k_aug_mhe()
    e.sens_dot_mhe()


    e.init_step_mhe(dum, e.nfe_t, patch_y=True)
    e.solve_d(e.lsmhe, skip_update=False)
    e.check_active_bound_noisy()
    e.load_covariance_prior()
    e.set_state_covariance()
    e.regen_objective_fun()

    e.set_prior_state_from_prior_mhe()

    e.print_r_mhe()
    e.shift_mhe()
    # e.lsmhe.u1.display(ostream=f0)
    # e.lsmhe.u2.display(ostream=f0)
    e.shift_measurement()
    # e.lsmhe.u1.display(ostream=f1)
    # e.lsmhe.u2.display(ostream=f1)




