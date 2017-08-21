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


c = NmpcGen(d_mod=DistDiehlNegrete,
            u=u,
            states=states,
            ref_state=ref_state,
            u_bounds=u_bounds)


c.load_iguess_ss()
c.solve_ss()
c.load_d_s(c.d1)
c.solve_d(c.d1)

c.update_state_real()  # update the current state

c.find_target_ss()
c.create_nmpc()
c.update_targets_nmpc()
c.compute_QR_nmpc(n=-1)
c.new_weights_olnmpc(10000, 1e+06)
c.d1.create_bounds()

c.create_predictor()
c.predictor_step(c.d1, "real")

q_cov = {}
for j in range(1, ntrays + 1):
    q_cov[("x", (j,))] = 1e-05
    q_cov[("M", (j,))] = 1


c.make_noisy(q_cov)
for i in range(1, 1000):
    c.solve_d(c.d1, stop_if_nopt=True, o_tee=True)

    # Dot_sens

    with open("debug1.txt", "w") as f:
        c.d1.w_pnoisy.display(ostream=f)
    c.randomize_noize(q_cov)
    c.update_state_real()  # update the current state
    c.update_soi_sp_nmpc()

    c.predictor_step(c.d1, "real")
    c.update_state_predicted()
    c.compute_offset_state("real")
    c.initialize_olnmpc(c.d2, "predicted")
    c.load_init_state_nmpc(src_kind="predicted")

    c.solve_d(c.olnmpc, stop_if_nopt=True, skip_update=False, iter_max=1000)
    c.update_u(c.olnmpc)
    c.print_r_nmpc()
    c.cycle_ics(plant_step=True)
    # c.cycle_ics_noisy()
    c.plant_input_gen(c.d1, src_kind="dict")



