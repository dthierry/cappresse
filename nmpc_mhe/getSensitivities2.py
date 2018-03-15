from __future__ import print_function
from pyomo.environ import *
from pyomo.core.base import Constraint, Objective, Suffix, minimize
from pyomo.opt import ProblemFormat, SolverFactory
from nmpc_mhe.dync.MHEGen2__obsolete import MheGen
from nmpc_mhe.mods.bfb.nob2 import bfb_dae
from snap_shot import snap
import sys, os
import itertools, sys
from numpy.random import normal as npm

states = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "mom"]
# x_noisy = ["Ngb", "Hgb", "Ngc", "Hgc", "Nsc", "Hsc", "Nge", "Hge", "Nse", "Hse", "mom"]
x_noisy = ["Hse"]
u = ["u1"]
u_bounds = {"u1":(162.183495794 * 0.0005, 162.183495794 * 10000)}

ref_state = {("c_capture", ((),)): 0.55}
# Known targets 0.38, 0.4, 0.5


y = ["Tgb", "vg"]
nfet = 5
ncpx = 3
nfex = 5
tfe = [i for i in range(1, nfet + 1)]
lfe = [i for i in range(1, nfex + 1)]
lcp = [i for i in range(1, ncpx + 1)]
lc = ['c', 'h', 'n']

y_vars = {
    "Tgb": [(1,1), (5,3)],
    "vg": [(1,1), (5,3)],
    }
x_vars = dict()
x_vars = {
          "Hse": [(1, 1)],
          }

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
           diag_QR=True,
           IgnoreProcessNoise=True)
e.ss.dref = snap

e.load_iguess_ss()
# sys.exit()
e.ss.create_bounds()
e.solve_ss()
e.load_d_s(e.d1)
e.d1.create_bounds()
e.solve_d(e.d1)

q_cov = {}
# for i in tfe:
#     if i < nfet:
#         for j in itertools.product(lfe, lcp, lc):
#             q_cov[("Ngb", j), ("Ngb", j), i] = 0.01*5.562535786e-05
#             q_cov[("Ngc", j), ("Ngc", j), i] = 0.01*0.000335771530697
#             q_cov[("Nsc", j), ("Nsc", j), i] = 0.01*739.786503718
#             q_cov[("Nge", j), ("Nge", j), i] = 0.01*0.0100570141164
#             q_cov[("Nse", j), ("Nse", j), i] = 0.01*641.425020561
for i in tfe:
    if i < nfet:
        for j in [(1,1)]:
            q_cov[("Hse", j), ("Hse", j), i] = 561353.476801 * 0.01


m_cov = {}
for i in lfe:
    for j in [(1,1), (5,3)]:
        m_cov[("Tgb", j), ("Tgb", j), i] = 2
        m_cov[("vg", j), ("vg", j), i] = 0.1


u_cov = {}
for i in tfe:
    u_cov["u1", i] = 5



e.set_covariance_meas(m_cov)
e.set_covariance_disturb(q_cov)
e.set_covariance_u(u_cov)
e.create_rh_sfx()  #: Reduced hessian computation

# Preparation phase
# e.init_lsmhe_prep(e.d1)

#e.shift_mhe()
dum = e.d_mod(1, e.ncp_t, _t=e.hi_t)

dum.create_bounds()
# e.init_step_mhe(dum, e.nfe_t)
# e.deb_alg_sys_dyn()
tst = e.solve_d(e.d1, skip_update=False)  #: Pre-loaded mhe solve
# with open("cons_first.txt", "w") as f:
#     for con in e.lsmhe.component_objects(Constraint, active=True):
#         con.pprint(ostream=f)
#     for obj in e.lsmhe.component_objects(Objective, active=True):
#         obj.pprint(ostream=f)
#     f.close()

# with open("cons_mhe.txt", "w") as f:
#     for con in e.lsmhe.component_objects(Constraint, active=True):
#         con.pprint(ostream=f)
#     for obj in e.lsmhe.component_objects(Objective, active=True):
#         obj.pprint(ostream=f)
#     f.close()
e.find_target_ss()  #: Compute target-steady state (beforehand)

# For ideal nmpc
for i in range(1, 2):
    print(str(i) + "--"*20, file=sys.stderr)
    print("*"*100)


    e.solve_d(e.d1)
    e.update_state_real()  # update the current state


    e.cycle_ics(plant_step=True)
    e.plant_input_gen(e.d1, "mod", src=e.ss2)
    # e.plant_input_gen(e.d1, src_kind="dict")

e.d1.dcdp = Suffix(direction=Suffix.EXPORT)
e.d1.var_order = Suffix(direction=Suffix.EXPORT)

e.d1.sens_init_constr = Suffix(direction=Suffix.EXPORT)
small_value = 1e-06
ii = 1
for x in e.states:
    con = getattr(e.d1, x + '_icc')
    con.set_suffix_value(e.d1.sens_init_constr, 1)
    var = getattr(e.d1, x)
    for key in var.keys():
        if key[1] == 0:
            if var[key].stale:
                continue
            setattr(e.d1, "sens_state_" + str(ii), Suffix(direction=Suffix.EXPORT))
            setattr(e.d1, "sens_state_value_" + str(ii), Suffix(direction=Suffix.EXPORT))
            setattr(e.d1, "sens_sol_state_" + str(ii), Suffix(direction=Suffix.IMPORT))
            iii = 1
            for xx in e.states:
                vv = getattr(e.d1, xx)
                for kk in vv.keys():
                    if kk[1] == 0:
                        if vv[kk].stale:
                            continue
                        sens_state = getattr(e.d1, "sens_state_" + str(ii))
                        sens_state_value = getattr(e.d1, "sens_state_value_" + str(ii))
                        vv[kk].set_suffix_value(sens_state, iii)
                        if ii == iii:
                            epsi = value(vv[kk]) * (1 + small_value)
                        else:
                            epsi = value(vv[kk])
                        vv[kk].set_suffix_value(sens_state_value, epsi)
                        iii += 1
            ii += 1

with open("ipopt.opt", "a") as f:
    f.write("\nn_sens_steps\t" + str(ii-1)+"\n")
    f.write("\nrun_sens\tyes\n")
    f.close()
for suf in e.d1.component_objects(Suffix):
    print(suf)
print("ipopt sens")
isens = SolverFactory("ipopt_sens",
                      executable="/home/dav0/Apps/Ipopt/build/bin/ipopt_sens")
e.d1.write_nl(name="sip.nl")
isens.solve(e.d1, tee=True)

with open("sensitivities_sipopt.dat", "w") as f:
    for x in e.states:
        var = getattr(e.d1, x)
        for k in var.keys():
            if k[1] == e.ncp_t:
                if var[k].stale:
                    continue
                ii = 1
                for xx in e.states:
                    vv = getattr(e.d1, xx)
                    for kk in vv.keys():
                        if kk[1] == e.ncp_t:
                            if vv[kk].stale:
                                continue
                            sens_sol_state = getattr(e.d1, "sens_sol_state_" + str(ii))
                            ds = var[k].get_suffix_value(sens_sol_state) - value(var[k])
                            dsdp = ds/small_value
                            s1 = str(dsdp)
                            f.write(s1 + "\t")
                            ii += 1
                f.write("\n")
                print(ii-1)
    f.close()

with open("states_sens.txt", "w") as f:
    f.close()

ii = 1
for x in e.states:
    con = getattr(e.d1, x + '_icc')
    with open("states_sens.txt", "a") as f:
        for key in con.keys():
            if not con[key].active:
                continue
            f.write(x + "\t" + str(key) + "\n")
        f.close()
    # con.set_suffix_value(e.d1.dcdp, 1)
    var = getattr(e.d1, x)
    for key in var.keys():
        if key[1] == e.ncp_t:
            if var[key].stale:
                continue
            con[key[2:]].set_suffix_value(e.d1.dcdp, ii)
            var[key].set_suffix_value(e.d1.var_order, ii)
            ii += 1
f = open("suf0.txt", "w")
e.d1.var_order.pprint(ostream=f)
f.close()
e.d1.dum_of = Objective(expr=1,sense=minimize)
e.d1.write_nl(name="whatevs0.nl")
kaug = SolverFactory("k_aug",
                     executable="/home/dav0/k2/KKT_matrix/src/kmatrix/k_aug")
kaug.options["compute_dsdp"] = ""
f = open("suf1.txt", "w")
e.d1.var_order.pprint(ostream=f)
f.close()
e.d1.write_nl(name="whatevs.nl")
kaug.solve(e.d1, tee=True)
# Turn off file determinism