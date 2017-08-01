from __future__ import print_function
from pyomo.environ import *
from nmpc_mhe.dync.MHEGen import MheGen
from nmpc_mhe.mods.bfb.bfb_abs import *
import sys
import itertools, sys

states = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "Ws"]
x_noisy = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "Ws"]
u = ["u1", "u2"]
u_bounds = {"u1":(0.0001, 99.9999), "u2":(0.0001, 99.99)}
ref_state = {("c_capture", ((),)): 0.45}

# Let's roll with the Temperature of the gas-emulsion and pressure

y = ["Tge", "P", "vg"]
nfet = 5
ncpx = 3
nfex = 5
tfe = [i for i in range(1, nfet + 1)]
lfe = [i for i in range(1, nfex + 1)]
lcp = [i for i in range(1, ncpx + 1)]
lc = ['c', 'h', 'n']

y_vars = {"Tge": [i for i in itertools.product(lfe, lcp)],
          "P": [i for i in itertools.product(lfe, lcp)],
          "vg": [i for i in itertools.product(lfe, lcp)]}

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

e = MheGen(d_mod=bfb_dae,
           y=y,
           x_noisy=x_noisy,
           y_vars=y_vars,
           x_vars=x_vars,
           states=states,
           u=u,
           ref_state=ref_state,
           u_bounds=u_bounds)
e.load_iguess_ss()
e.solve_ss()
e.load_d_s(e.d1)
e.solve_d(e.d1)


q_cov = {}
for i in tfe:
    if i < nfet:
        for j in itertools.product(lfe, lcp, lc):
            # "Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "Ws"
            q_cov[("Ngb", j), ("Ngb", j), i] = 1.e-05
            q_cov[("Ngc", j), ("Ngc", j), i] = 1.e-05
            q_cov[("Nsc", j), ("Nsc", j), i] = 10.0
            q_cov[("Nge", j), ("Nge", j), i] = 0.01
            q_cov[("Nse", j), ("Nse", j), i] = 10.0
for i in tfe:
    if i < nfet:
        for j in itertools.product(lfe, lcp):
            q_cov[("Hgb", j), ("Hgb", j), i] = 10.
            q_cov[("Hgc", j), ("Hgc", j), i] = 5.
            q_cov[("Hsc", j), ("Hsc", j), i] = 100.
            q_cov[("Hge", j), ("Hge", j), i] = 10.
            q_cov[("Hse", j), ("Hse", j), i] = 100.
            q_cov[("Ws", j), ("Ws", j), i] = 10.


m_cov = {}
for i in lfe:
    for j in itertools.product(lfe, lcp):
        m_cov[("Tge", j), ("Tge", j), i] = 10.0
        m_cov[("P", j), ("P", j), i] = 1e-03
        m_cov[("vg", j), ("vg", j), i] = 1e-04

e.set_covariance_meas(m_cov)
e.set_covariance_disturb(q_cov)
e.init_lsmhe_prep(e.d1)

e.shift_mhe()
dum = e.d_mod(1, e.ncp_t, _t=e.hi_t)

e.init_step_mhe(dum, e.nfe_t)
e.solve_d(e.lsmhe, skip_update=False)

e.create_rh_sfx()

# e.check_active_bound_noisy()
# e.load_covariance_prior()
# e.set_state_covariance()
#
# e.regen_objective_fun()
# e.deact_icc_mhe()
#
# e.set_prior_state_from_prior_mhe()
# e.find_target_ss()
#
#
#
#
# for i in range(1, 2):
#     print(str(i) + "--"*20, file=sys.stderr)
#     print(i)
#     print("*"*100)
#
#
#     if i == 10:
#         e.plant_input_gen(e.ss2, 1)
#         # e.lsmhe.pprint(filename="somefile.txt")
#
#
#     e.solve_d(e.d1)
#     e.update_noise_meas(e.d1, m_cov)
#
#     e.patch_meas_mhe(e.nfe_t, src=e.d1, noisy=True)
#     e.load_inputsmhe(src=e.d1, fe=e.nfe_t)
#
#     e.compute_y_offset()
#
#     e.create_sens_suffix()
#     print("testing \n\n\n" * 4)
#
#     e.sens_k_aug_mhe()
#     e.sens_dot_mhe()
#
#
#     e.init_step_mhe(dum, e.nfe_t, patch_y=True)
#     e.solve_d(e.lsmhe, skip_update=False)
#     e.check_active_bound_noisy()
#     e.load_covariance_prior()
#     e.set_state_covariance()
#     e.regen_objective_fun()
#
#     e.set_prior_state_from_prior_mhe()
#
#     e.print_r_mhe()
#     e.shift_mhe()
#     e.shift_measurement()
#
#
#
#
