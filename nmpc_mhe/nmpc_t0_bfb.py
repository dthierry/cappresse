from pyomo.environ import *
from nmpc_mhe.dync.NMPCGen import NmpcGen
from nmpc_mhe.mods.bfb.bfb_abs import *
import sys

states = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "Ws"]
u = ["u1", "u2"]
u_bounds = {"u1":(0.0001, 99.9999), "u2":(0.0001, 99.99)}
ref_state = {("c_capture", ((),)): 0.45}
c = NmpcGen(d_mod=bfb_dae, u=u, states=states, ref_state=ref_state, u_bounds=u_bounds)
c.load_iguess_ss()
c.solve_ss()
c.load_d_s(c.d1)
c.solve_d(c.d1, iter_max=2)

c.find_target_ss()
# c.plant_input_gen(c.ss2, 1)

c.create_nmpc()
c.update_targets_nmpc()
# sys.exit()
c.compute_QR_nmpc(n=-1)
c.initialize_olnmpc(c.d1)
c.olnmpc.display(filename="somefile0.txt")
c.solve_d(c.olnmpc)
c.initialize_olnmpc(c.d1)
c.olnmpc.display(filename="somefile1.txt")
c.solve_d(c.olnmpc)
c.update_u(src=c.olnmpc)
c.cycle_ics()
c.plant_input_gen(c.d1)
c.solve_d(c.d1)
c.update_state_real()
c.initialize_olnmpc(c.olnmpc, fe=5)
c.solve_d(c.olnmpc)

c.create_predictor()
c.predictor_step(c.d1)
c.update_state_predicted()
c.update_u(src=c.olnmpc)
c.update_soi_sp_nmpc()
# c.print_r_nmpc()

