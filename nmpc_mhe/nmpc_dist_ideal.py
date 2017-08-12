from __future__ import print_function
from pyomo.environ import *
# from nmpc_mhe.dync.MHEGen import MheGen
from nmpc_mhe.dync.NMPCGen import NmpcGen
from nmpc_mhe.mods.distl.dist_col import *
import sys
import itertools, sys

states = ["x", "M"]
x_noisy = ["x", "M"]
u = ["u1", "u2"]
ref_state = {("T", (29,)): 343.15, ("T", (14,)): 361.15}
u_bounds = {"u1": (0.0001, 9.9999e-1), "u2": (0, 1e+08)}
# weights =  {("T", (29,)): 1000., ("T", (14,)): 1000.}
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

# e = MheGen(d_mod=DistDiehlNegrete,
#            y=y,
#            x_noisy=x_noisy,
#            y_vars=y_vars,
#            x_vars=x_vars,
#            states=states,
#            u=u,
#            ref_state=ref_state,
#            u_bounds=u_bounds,
#            diag_QR=True,
#            nfe_t=nfet)

c = NmpcGen(d_mod=DistDiehlNegrete,
            u=u,
            states=states,
            ref_state=ref_state,
            u_bounds=u_bounds)


c.load_iguess_ss()
c.solve_ss()
c.load_d_s(c.d1)
c.solve_d(c.d1)

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
    u_cov["u2", i] = 1.78604740940007800236344337463379E+06 * 0.001


# e.set_covariance_meas(m_cov)
# e.set_covariance_disturb(q_cov)
# e.set_covariance_u(u_cov)

# Preparation phase
# e.init_lsmhe_prep(e.d1)

# e.shift_mhe()
# dum = e.d_mod(1, e.ncp_t, _t=e.hi_t)
c.find_target_ss()
c.create_nmpc()
c.update_targets_nmpc()
c.compute_QR_nmpc(n=-1)
c.initialize_olnmpc(c.d1)

c.solve_d(c.olnmpc)
c.initialize_olnmpc(c.d1)
c.solve_d(c.olnmpc)
c.update_u(c.olnmpc)
c.cycle_ics()
c.plant_input_gen(c.d1, src_kind="dict")

