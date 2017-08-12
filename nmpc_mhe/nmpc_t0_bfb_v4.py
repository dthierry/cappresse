from pyomo.environ import *
from nmpc_mhe.dync.NMPCGen import NmpcGen
from nmpc_mhe.mods.bfb.bfb_abs_v4 import *
import sys

states = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "Ws"]
u = ["u1", "u2"]
u_bounds = {"u1":(0.0001, 9937.98446662*10), "u2":(0.0001, 9937.98446662*10)}
ref_state = {("c_capture", ((),)): 0.44}
# weights = {("c_capture", ((),)): 1E+05}
c = NmpcGen(d_mod=bfb_dae, u=u, states=states, ref_state=ref_state, u_bounds=u_bounds)
c.load_iguess_ss()
c.solve_ss()
c.load_d_s(c.d1)
c.solve_d(c.d1)

c.find_target_ss()
c.ss2.write_nl()
# sys.exit(-1)
# c.plant_input_gen(c.ss2, 1)

c.create_nmpc()
c.update_targets_nmpc()
# sys.exit()
c.compute_QR_nmpc(n=-1)
c.initialize_olnmpc(c.d1)
# c.olnmpc.display(filename="somefile0.txt")
c.create_suffixes()
# c.solve_k_aug_nmpc()
# c.olnmpc.write_nl()
i = 1
k = 1
while k != 0:
    if i == 1:
        k = c.solve_d(c.olnmpc, max_cpu_time=60*i)
    else:
        k = c.solve_d(c.olnmpc, max_cpu_time=60 * i, warm_start=True)
    c.olnmpc.ipopt_zL_in.update(c.olnmpc.ipopt_zL_out)
    c.olnmpc.ipopt_zU_in.update(c.olnmpc.ipopt_zU_out)
    i += 1

c.initialize_olnmpc(c.d1)
c.olnmpc.display(filename="somefile1.txt")
c.solve_d(c.olnmpc, max_cpu_time=60*60)
c.update_u(src=c.olnmpc)
c.cycle_ics()
c.plant_input_gen(c.d1, src_kind="dict")
c.solve_d(c.d1)
c.update_state_real()
c.initialize_olnmpc(c.olnmpc, fe=5)
c.solve_d(c.olnmpc, max_cpu_time=60*60)

c.create_predictor()
c.predictor_step(c.d1)
c.update_state_predicted()
c.update_u(src=c.olnmpc)
c.update_soi_sp_nmpc()
# c.print_r_nmpc()

