from pyomo.environ import *
from nmpc_mhe.dync.MHEGen import MheGen
from nmpc_mhe.mods.distl.dist_col import DistDiehlNegrete

y = ["T", "Mv", "Mv1", "Mvn"]
u = ["u1", "u2"]

states = ["x", "M"]
x_noisy = ["x", "M"]

ntrays = 42
y_vars = {"T": [(i,) for i in range(1, ntrays + 1)],
          "Mv": [(i,) for i in range(2, ntrays)],
          "Mv1":[((),)],
          "Mvn":[((),)]}

x_vars = {"x": [(i,) for i in range(1, ntrays + 1)],
          "M": [(i,) for i in range(1, ntrays + 1)]}

e = MheGen(d_mod=DistDiehlNegrete, y=y, x_noisy=x_noisy, y_vars=y_vars, x_vars=x_vars, states=states, u=u)

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
