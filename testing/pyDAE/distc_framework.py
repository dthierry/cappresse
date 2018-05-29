#!/usr/bin/env python
# -*- coding: utf-8
from __future__ import division
from __future__ import print_function

from nmpc_mhe.pyomo_dae.MHEGen_pyDAE import MheGen_DAE
from sample_mods.distc_pyDAE.distcpydaemod import mod
from nmpc_mhe.aux.utils import load_iguess, create_bounds
from pyomo.core.base import Var, Constraint, Param
from pyomo.opt import SolverFactory, ProblemFormat
from numpy import random
import sys

__author__ = "David Thierry"

def disp_vars(mod, file):
    if not file is None:
        with open(file, "w") as f:
            for i in mod.component_objects(Var):
                i.display(ostream=f)
    else:
        for i in mod.component_objects(Var):
            i.display()

def disp_cons(mod, file):
    if not file is None:
        with open(file, "w") as f:
            for i in mod.component_objects(Constraint):
                i.pprint(ostream=f)
    else:
        for i in mod.component_objects(Constraint):
            i.pprint()

def disp_params(mod, file):
    if not file is None:
        with open(file, "w") as f:
            for i in mod.component_objects(Param):
                i.pprint(ostream=f)
    else:
        for i in mod.component_objects(Param):
            i.pprint()


def main():
    states = ["x", "M"]
    state_bounds = {"M": (1.0, 1e+07),
                    "T": (200, 500),
                    "pm": (1.0, 5e+07),
                    "pn": (1.0, 5e+07),
                    "L": (0.0, 1e+03),
                    "V": (0.0, 1e+03),
                    "x": (0.0, 1.0),
                    "y": (0.0, 1.0),
                    "hl": (1.0, 1e+07),
                    "hv": (1.0, 1e+07),
                    "Qc": (0.0, 1e+08),
                    "D": (0.0, 1e+04),
                    "Vm": (0.0, 1e+04),
                    "Mv": (0.155 + 1e-06, 1e+04),
                    "Mv1": (8.5 + 1e-06, 1e+04),
                    "Mvn": (0.17 + 1e-06, 1e+04)
                    }

    measurements = ["T", "Mv", "Mv1", "Mvn"]
    controls = ["u1", "u2"]
    u_bounds = {"u1": (0000.1, 99.999), "u2": (0, None)}
    ref_state = {("T", (29,)): 343.15, ("T", (14,)): 361.15}

    e = MheGen_DAE(mod, 100, states, controls, states, measurements,
                   u_bounds=u_bounds,
                   ref_state=ref_state,
                   override_solver_check=True,
                   var_bounds=state_bounds,
                   k_aug_executable='/home/dav0/devzone/k_aug/cmake-build-k_aug/bin/k_aug',
                   dot_driver_executable='/home/dav0/devzone/k_aug/src/k_aug/dot_driver')
    Q = {}
    U = {}
    R = {}
    Q["x"] = 1e-05
    Q["M"] = 0.5
    R["T"] = 6.25e-02
    R["Mv"] = 10e-08
    R["Mv1"] = 10e-08
    R["Mvn"] = 10e-08
    U["u1"] = 7.72700925775773761472464684629813E-01 * 0.01
    U["u2"] = 1.78604740940007800236344337463379E+06 * 0.001

    e.set_covariance_disturb(Q)
    e.set_covariance_u(U)
    e.set_covariance_meas(R)
    e.create_rh_sfx()

    e.get_state_vars()

    create_bounds(e.SteadyRef, bounds=state_bounds)
    ipopt = SolverFactory('ipopt')
    for i in range(1, 1):
        continue
        ipopt.options["print_user_options"] = "yes"
        ipopt.options["OF_start_with_resto"] = "yes"
        ipopt.options["OF_bound_relax_factor"] = 1e-12
        ipopt.options["OF_honor_original_bounds"] = "no"
        ipopt.solve(e.SteadyRef, tee=True)
        disp_vars(e.SteadyRef, "my_vars")
        disp_cons(e.SteadyRef, "my_cons")
        # e.SteadyRef.Tdot.fix(0)gt
        ipopt.options["OF_start_with_resto"] = "no"
        ipopt.solve(e.SteadyRef, tee=True)
        bp = random.random(1)
        # ipopt.options["OF_bound_push"] = bp[0]

    # ipopt.options["OF_start_with_resto"] = "no"
    # ipopt.options["OF_bound_relax_factor"] = 1e-12
    # ipopt.options["OF_honor_original_bounds"] = "no"
    ipopt.options["bound_push"] = 1e-07
    ipopt.solve(e.SteadyRef, tee=True)
    # disp_vars(e.SteadyRef, "my_vars")
    # disp_cons(e.SteadyRef, "my_cons")

    # create_bounds(e.SteadyRef, bounds=dict(), clear=True)
    # ip2 = SolverFactory('ipopt')
    # ip2.options["halt_on_ampl_error"] = "yes"
    # ip2.solve(e.SteadyRef, tee=True, symbolic_solver_labels=True)


    e.load_iguess_steady()
    # load_iguess(e.SteadyRef, e.PlantSample, 0, 0)
    # e.cycleSamPlant()
    # reconcile_nvars_mequations(e.lsmhe)
    # create_bounds(e.PlantSample, bounds=e.var_bounds, clear=False)
    #
    # ipopt.options["halt_on_ampl_error"] = "yes"
    # ipopt.options["bound_push"] = 1e-07
    ipopt.solve(e.PlantSample,
                tee=True,
                symbolic_solver_labels=True)

    # disp_vars(e.PlantSample, "my_vars")
    # disp_cons(e.PlantSample, "my_cons")
    # disp_params(e.PlantSample, "my_params")

    # e.solve_dyn(e.PlantSample)
    # sys.exit()
    e.init_lsmhe_prep(e.PlantSample)
    e.shift_mhe()
    e.init_step_mhe()
    e.lsmhe.pprint(filename="lsmhe.txt")

    e.solve_dyn(e.lsmhe,
                skip_update=False,
                max_cpu_time=600,
                ma57_pre_alloc=5, tag="lsmhe")  #: Pre-loaded mhe solve
    disp_vars(e.lsmhe, "vars_mhe")

    e.lsmhe.write("lsmhe.nl", format=ProblemFormat.nl, io_options={'symbolic_solver_labels': True})

    ipopt.options["print_user_options"] = "yes"
    ipopt.options["OF_start_with_resto"] = "yes"
    ipopt.options["OF_honor_original_bounds"] = "no"
    ipopt.options["bound_push"] = 1e-02
    ipopt.solve(e.lsmhe, tee=True, symbolic_solver_labels=True)
    e.prior_phase()
    e.deact_icc_mhe()  #: Remove the initial conditions

    e.find_target_ss()

    #: NMPC
    e.create_nmpc()
    e.create_suffixes_nmpc()
    e.update_targets_nmpc()
    e.compute_QR_nmpc(n=-1)
    e.new_weights_olnmpc(1E-04, 1e+06)


if __name__ == '__main__':
    main()
