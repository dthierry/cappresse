#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, Set, Constraint, Expression, Param, Suffix, maximize
from pyomo.core.base import ConstraintList
from pyomo.core.kernel.numvalue import value
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
import numpy as np
import sys, time, re, os
from pyutilib.common._exceptions import ApplicationError
import datetime

__author__ = "David M Thierry @dthierry"
class LogfileError(RuntimeError):
    """Exception raised when the log file name is not well defined"""
    def __init__(self, arg):
        self.args = arg

class DynGen(object):
    """Default class for the Dynamic model"""
    def __init__(self, d_mod, hi_t, states, controls, **kwargs):

        # Base model
        if not d_mod:
            print("Warning no model declared")
        self.d_mod = d_mod

        self.nfe_t = kwargs.pop('nfe_t', 5)
        self.ncp_t = kwargs.pop('ncp_t', 3)
        self.k_aug_executable = kwargs.get('k_aug_executable', "/home/dav0/k2/KKT_matrix/src/kmatrix/k_aug")
        self.ipopt_executable = kwargs.get('ipopt_executable', None)

        self.hi_t = hi_t

        self._t = hi_t * self.nfe_t

        self.states = states
        self.u = controls

        # Values for the suffixes of input files
        self.int_file_mhe_suf = int()
        self.res_file_mhe_suf = str()

        self.int_file_nmpc_suf = int()
        self.res_file_nmpc_suf = str()

        self.res_file_suf = str(int(time.time()))


        # self.hi_t = self._t/self.nfe_t

        self.SteadyRef = self.d_mod(1, 1, steady=True)
        self.SteadyRef2 = object()
        self.PlantSample = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        self.PlantPred = object()
        self.SteadyRef.name = "SteadyRef"
        self.PlantSample.name = "PlantSample"
        if type(self.ipopt_executable) == str():
            self.ipopt = SolverFactory("ipopt",
                                       executable=self.ipopt_executable)
            self.asl_ipopt = SolverFactory("asl:ipopt",
                                           executable=self.ipopt_executable)
        else:
            self.ipopt = SolverFactory("ipopt")
            self.asl_ipopt = SolverFactory("asl:ipopt")

        self.k_aug = SolverFactory("k_aug",
                                   executable=self.k_aug_executable)
        self.k_aug_sens = SolverFactory("k_aug",
                                        executable=self.k_aug_executable)
        self.dot_driver = SolverFactory("dot_driver",
                                        executable="/home/dav0/k2/KKT_matrix/src/kmatrix/dot_driver/dot_driver")

        # self.k_aug.options["eig_rh"] = ""
        self.asl_ipopt.options["halt_on_ampl_error"] = "yes"

        # self.ipopt.options["print_user_options"] = "yes"
        # self.k_aug.options["deb_kkt"] = ""

        self.SteadyRef.ofun = Objective(expr=1, sense=minimize)
        self.dyn = object()
        self.l_state = []
        self.l_vals = []
        self._c_it = 0
        self.ccl = []
        self.iput = []
        self.sp = []

        self._kt_list = []
        self._dt_list = []
        self._ipt_list = []

        self._k_timing = ["0", "0", "0"]
        self._dot_timing = "0"
        self.ip_time = 0

        self._stall_iter = 0
        self._window_keep = self.nfe_t + 2

        self._u_plant = {}  #: key: (ui, time)
        for i in self.u:
            u = getattr(self.PlantSample, i)
            for t in range(0, self._window_keep):
                self._u_plant[(i, t)] = value(u[1])
        self.curr_u = dict.fromkeys(self.u, 0.0)

        self.state_vars = {}
        self.curr_estate = {}  #: Current estimated state (for the olnmpc)
        self.curr_rstate = {}  #: Current real state (for the olnmpc)


        self.curr_state_offset = {}  #: Current offset of measurement
        self.curr_pstate = {}  #: Current offset of measurement
        self.curr_state_noise = {}  #: Current noise of the state

        self.curr_state_target = {}  #: Current target state
        self.curr_u_target = {}  #: Current control state

        self.xp_l = []
        self.xp_key = {}


        with open("ipopt.opt", "w") as f:
            f.close()

    def load_iguess_steady(self):
        """"Call the method for loading initial guess from steady-state"""
        self.SteadyRef.init_steady_ref()

    def solve_steady_ref(self):
        """Solves steady state model (SteadyRef)
        Args:
            None
        Return:
            None"""
        # Create a separate set of options for Ipopt
        with open("ipopt.opt", "w") as f:
            f.write("max_iter 100\n")
            f.write("mu_init 1e-08\n")
            f.write("bound_push 1e-08\n")
            f.write("print_info_string yes\n")
            f.write("print_user_options yes\n")
            f.write("linear_solver ma57\n")
            f.close()
        ip = SolverFactory("ipopt")

        results = ip.solve(self.SteadyRef,
                           tee=True,
                           symbolic_solver_labels=True,
                           report_timing=True)

        self.SteadyRef.solutions.load_from(results)

        # Gather the keys for a given state and form the state_vars dictionary
        for x in self.states:
            self.state_vars[x] = []
            try:
                xv = getattr(self.SteadyRef, x)
            except AttributeError:  # delete this
                continue
            for j in xv.keys():
                if xv[j].stale:
                    continue
                if type(j[2:]) == tuple:
                    self.state_vars[x].append(j[2:])
                else:
                    self.state_vars[x].append((j[2:],))
        
        # Get values for reference states and controls
        for x in self.states:
            try:
                xvar = getattr(self.SteadyRef, x)
            except AttributeError:  # delete this
                continue
            for j in self.state_vars[x]:
                self.curr_state_offset[(x, j)] = 0.0
                self.curr_state_noise[(x, j)] = 0.0
                self.curr_estate[(x, j)] = value(xvar[1, 1, j])
                self.curr_rstate[(x, j)] = value(xvar[1, 1, j])
                self.curr_state_target[(x, j)] = value(xvar[1, 1, j])
        for u in self.u:
            uvar = getattr(self.SteadyRef, u)
            self.curr_u_target[u] = value(uvar[1])
            self.curr_u[u] = value(uvar[1])
        with open("res_dyn_label_" + self.res_file_suf + ".txt", "w") as f:
            for x in self.states:
                for j in self.state_vars[x]:
                    jth = (x, j)
                    jth = str(jth)
                    f.write(jth)
                    f.write('\t')
            f.close()
        self.journalist("I", self._c_it, "solve_steady_ref", "labels at " + self.res_file_suf)


    def load_d_s(self, dmod):
        """Loads the solution of the steady state model into the dynamic
        Args:
            dmod (pyomo.core.base.PyomoModel.ConcreteModel): Target model
        Return:
            None"""
        s = self.SteadyRef
        d = dmod
        for vs in s.component_objects(Var, active=True):
            vd = getattr(d, vs.getname())
            if vs.is_indexed():
                if len(vs.keys()) > 1:
                    # print(vs.getname())
                    for ks in vs.keys():
                        kj = ks[2:]
                        # for i in range(1, self.nfe_t + 1):
                        for j in range(1, self.ncp_t + 1):
                            vd[(1, j) + kj].set_value(value(vs[ks]))
                else:
                    for ks in vs.keys():
                        for kd in vd.keys():
                            vd[kd].set_value(value(vs[ks]))
        for x in self.states:
            try:
                dv = getattr(dmod, "d" + x + "_dt")
            except AttributeError:  # delete this
                continue
            for i in dv.itervalues():
                i.set_value(0.0)
        for i in self.states:
            pn = i + "_ic"
            p = getattr(d, pn)
            vs = getattr(s, i)
            vd0 = getattr(d, i)
            for ks in p.keys():
                p[ks].value = value(vs[(1, 1)+(ks,)])
                vd0[(1, 0)+(ks,)].set_value(value(vs[(1, 1)+(ks,)]))

    def solve_dyn(self, mod, **kwargs):
        """Solves a given dynamic model
        Args:
            mod (pyomo.core.base.PyomoModel.ConcreteModel): Target model
        Return:
            int: 0 if success 1 otw"""
        d = mod
        mu_init = 0.1
        iter_max = 3000
        max_cpu_time = 1e+06
        linear_solver = "ma57"
        ma57_pre_alloc = 1.05
        ma57_automatic_scaling = "no"
        ma57_small_pivot_flag = 0

        # o_tee = True
        stop_if_nopt = False
        want_stime = False  #: Updates internal time variable
        rep_timing = False
        stop_if_nopt = kwargs.pop("stop_if_nopt", False)
        # if kwargs.get("stop_if_nopt"):
        #     stop_if_nopt = kwargs["stop_if_nopt"]
        if kwargs.get("want_stime"):
            want_stime = kwargs["want_stime"]
        if kwargs.get("rep_timing"):
            rep_timing = kwargs["rep_timing"]
        if kwargs.get("stop_if_nopt"):
            stop_if_nopt = kwargs["stop_if_nopt"]
        if kwargs.get("iter_max"):
            iter_max = kwargs["iter_max"]

        if kwargs.get("linear_solver"):
            linear_solver = kwargs["linear_solver"]
        if kwargs.get("ma57_pre_alloc"):
            ma57_pre_alloc = kwargs["ma57_pre_alloc"]
        if kwargs.get("ma57_automatic_scaling"):
            ma57_automatic_scaling = kwargs["ma57_automatic_scaling"]
        if kwargs.get("ma57_small_pivot_flag"):
            ma57_small_pivot_flag = kwargs["ma57_small_pivot_flag"]

        max_cpu_time = kwargs.pop("max_cpu_time", 1e+06)

        o_tee = kwargs.pop("o_tee", True)
        skip_mult_update = kwargs.pop("skip_update", True)
        halt_on_ampl_error = kwargs.pop("halt_on_ampl_error", False)
        warm_start = kwargs.pop("warm_start", False)
        tol = kwargs.pop("tol", None)
        mu_init = kwargs.pop("mu_init", None)

        linear_scaling_on_demand = kwargs.pop("linear_scaling_on_demand", None)
        mu_strategy = kwargs.pop("mu_strategy", None)
        perturb_always_cd = kwargs.pop("perturb_always_cd", None)
        mu_target = kwargs.pop("mu_target", None)
        print_level = kwargs.pop("print_level", None)
        print_user_options = kwargs.pop("print_user_options", True)
        ma57_pivtol = kwargs.pop("ma57_pivtol", None)
        bound_push = kwargs.pop("bound_push", None)

        out_file = kwargs.pop("output_file", None)
        if out_file:
            if type(out_file) != str:
                self.journalist("E", self._c_it, "solve_dyn", "incorrect_output")
                print("output_file is not str", file=sys.stderr)
                sys.exit()

        tag = kwargs.pop("tag", None)

        if tag:
            if type(tag) == str:
                out_file = tag + "_log.log"
            else:
                raise LogfileError("Wrong type for the tag argument")



        jacRegVal = kwargs.pop("jacobian_regularization_value", None)
        jacRegExp = kwargs.pop("jacobian_regularization_exponent", None)
        if jacRegVal:
            if type(jacRegVal) != float:
                self.journalist("E", self._c_it, "solve_dyn", "incorrect_output")
                print("jacobian_regularization_value is not float", file=sys.stderr)

                sys.exit()
        if jacRegExp:
            if type(jacRegExp) != float:
                self.journalist("E", self._c_it, "solve_dyn", "incorrect_output")
                print("jacobian_regularization_exponent is not float", file=sys.stderr)
                sys.exit()
        if mu_strategy:
            if mu_strategy != "monotone" and mu_strategy != "adaptive":
                self.journalist("E", self._c_it, "solve_dyn", "incorrect_output(mu_strategy)")
                print(mu_strategy)
                sys.exit()
        if mu_target:
            if type(mu_target) != float:
                self.journalist("E", self._c_it, "solve_dyn", "incorrect_output")
                print("mu_target is not float", file=sys.stderr)
                sys.exit()
        if print_level:
            if type(print_level) != int:
                self.journalist("E", self._c_it, "solve_dyn", "incorrect_output")
                print("print_level is not int", file=sys.stderr)
                sys.exit()
        if ma57_pivtol:
            if type(ma57_pivtol) != float:
                self.journalist("E", self._c_it, "solve_dyn", "incorrect_output")
                print("ma57_pivtol is not float", file=sys.stderr)
                sys.exit()
        if bound_push:
            if type(bound_push) != float:
                self.journalist("E", self._c_it, "solve_dyn", "incorrect_output")
                print("bound_push is not float", file=sys.stderr)
                sys.exit()
            
        name = mod.name

        self.journalist("I", self._c_it, "Solving with IPOPT\t", name)

        with open("ipopt.opt", "w") as f:
            f.write("print_info_string\tyes\n")
            f.write("max_iter\t" + str(iter_max) + "\n")

            f.write("max_cpu_time\t" + str(max_cpu_time) + "\n")
            f.write("linear_solver\t" + linear_solver + "\n")
            f.write("ma57_pre_alloc\t" + str(ma57_pre_alloc) + "\n")
            f.write("ma57_automatic_scaling\t" + ma57_automatic_scaling + "\n")
            f.write("ma57_small_pivot_flag\t" + str(ma57_small_pivot_flag) + "\n")
            if warm_start:
                f.write("warm_start_init_point\t" + "yes" + "\n")
                f.write("warm_start_bound_push\t" + "1e-06" + "\n")
                f.write("mu_init\t" + "0.001" + "\n")
            if tol:
                f.write("tol\t" + str(tol) + "\n")
            if mu_init:
                f.write("mu_init\t" + str(mu_init) + "\n")
            if out_file:
                f.write("output_file\t" + out_file + "\n")
            if linear_scaling_on_demand:
                f.write("linear_scaling_on_demand\tyes\n")
            if jacRegVal:
                f.write("jacobian_regularization_value\t" + str(jacRegVal) + "\n")
            if jacRegExp:
                f.write("jacobian_regularization_exponent\t" + str(jacRegExp) + "\n")
            if mu_strategy:
                f.write("mu_strategy\t" + mu_strategy + "\n")
            if perturb_always_cd:
                f.write("perturb_always_cd\t" + "yes" + "\n")
            if mu_target:
                f.write("mu_target\t" + str(mu_target) + "\n")
            if print_level:
                f.write("print_level\t" + str(print_level) + "\n")
            if print_user_options:
                f.write("print_user_options\t" + "yes" + "\n")
            if ma57_pivtol:
                f.write("ma57_pivtol\t" + str(ma57_pivtol) + "\n")
            if bound_push:
                f.write("bound_push\t" + str(bound_push) + "\n")


            # f.write("\ncheck_derivatives_for_naninf yes\n")
            f.close()
            if halt_on_ampl_error:
                solver_ip = self.asl_ipopt
            else:
                solver_ip = self.ipopt
        # Solution attempt
        try:
            results = solver_ip.solve(d, tee=o_tee, symbolic_solver_labels=True, report_timing=rep_timing)
        except (ApplicationError, ValueError):
            stop_if_nopt = 1

        # Append to the logger
        if tag:
            with open("log_ipopt_" + tag + "_" + self.res_file_suf + ".txt", "a") as global_log:
                with open(out_file, "r") as filelog:
                    global_log.write("--\t" + str(self._c_it) + "\t" + str(datetime.datetime.now()) + "\t" + "-"*50)
                    for line in filelog:
                        global_log.write(line)
                    filelog.close()
                global_log.close()


        if (results.solver.status == SolverStatus.ok) and \
                (results.solver.termination_condition == TerminationCondition.optimal):
            self.journalist("I", self._c_it, "solve_dyn", " Model solved to optimality")
            d.solutions.load_from(results)
            self._stall_iter = 0
            if want_stime and rep_timing:
                self.ip_time = self.ipopt._solver_time_x
            if not skip_mult_update:
                mod.ipopt_zL_in.update(mod.ipopt_zL_out)
                mod.ipopt_zU_in.update(mod.ipopt_zU_out)

            return 0
        else:
            if stop_if_nopt:
                self.journalist("E", self._c_it, "solve_dyn", "Not-optimal. Stoping")
                sys.exit()
            self.journalist("W", self._c_it, "solve_dyn", "Not-optimal.")
            return 1

    def cycleSamPlant(self, plant_step=False):
        """Patches the initial conditions with the last result from the simulation
        Args:
            None
        Return
            None"""
        print("-" * 120)
        print("I[[cycleSamPlant]] Cycling initial state.")
        print("-" * 120)
        for x in self.states:
            x_ic = getattr(self.PlantSample, x + "_ic")
            v_tgt = getattr(self.PlantSample, x)
            for ks in x_ic.keys():
                if type(ks) != tuple:
                    ks = (ks,)
                x_ic[ks].value = value(v_tgt[(1, self.ncp_t) + ks])
                v_tgt[(1, 0) + ks].set_value(value(v_tgt[(1, self.ncp_t) + ks]))
        if plant_step:
            self._c_it += 1

    def load_iguess_dyndyn(self, src, tgt, fe, fe_src='d'):
        """Loads the solution of the src state model into the tgt, i.e. src-->tgt
        Args:
            src (pyomo.core.base.PyomoModel.ConcreteModel): Model with source values
            tgt (pyomo.core.base.PyomoModel.ConcreteModel): Model whose values are to be assigned
            fe (int): The tgt's finite element
            fe_src (str):
        Return:
            None"""
        #: Check cps
        cp_src = getattr(src, "cp_t")
        tgt_src = getattr(tgt, "cp_t")
        if len(cp_src.value) != len(tgt_src.value):
            print("These variables do not have the same number of cps")
            sys.exit(-1)
        cp = max(cp_src)
        for vs in src.component_objects(Var, active=True):
            if vs.getname()[-7:] == "_pnoisy":
                continue
            vd = getattr(tgt, vs.getname())
            # there are two cases: 1 key 1 elem, several keys 1 element
            vskeys = vs.keys()  #: keys of the source model
            if len(vskeys) == 1:
                #: One key
                for ks in vskeys:
                    for v in vd.keys():
                        vd[v].set_value(value(vs[ks]))
            else:
                k = 0
                for ks in vskeys:
                    if k == 0:  #: Do this only once per variable
                        if type(ks) != tuple:
                            #: Several keys of 1 element each!!
                            if fe_src == "d":
                                for ks1 in vskeys:
                                    vd[ks1].set_value(value(vs[ks1]))  #: This has got to be true
                                break
                            elif fe_src == "s":
                                for ks1 in vd.keys():
                                    try:
                                        vd[ks1].set_value(value(vs[fe]))  #: This has got to be true
                                    except AttributeError:
                                        vd[ks1].value = value(vs[fe])  #: This has got to be true
                                break
                        k += 1
                    ki = ks[:2]
                    kj = ks[2:]
                    if fe_src == 'd':
                        if vs.getname() in self.states:  #: States start at 0
                            for j in range(0, cp + 1):
                                vd[(fe, j) + kj].set_value(value(vs[ks]))
                        else:
                            for j in range(1, cp + 1):
                                vd[(fe, j) + kj].set_value(value(vs[ks]))
                    elif fe_src == 's':
                        if vs.getname() in self.states:  #: States start at 0
                            for j in range(0, cp + 1):
                                if ki == (fe, j):
                                    vd[(1, j) + kj].set_value(value(vs[ks]))
                                else:
                                    continue
                        else:
                            for j in range(1, cp + 1):
                                if ki == (fe, j):
                                    vd[(1, j) + kj].set_value(value(vs[ks]))
                                else:
                                    continue

    def create_dyn(self, initialize=True):
        print("-" * 120)
        print("I[[create_dyn]] Dynamic (full) model created.")
        print("-" * 120)
        self.dyn = self.d_mod(self.nfe_t, self.ncp_t, _t=self._t)
        self.dyn.name = "full_dyn"
        self.load_d_s(self.dyn)
        if initialize:
            self.load_d_s(self.PlantSample)
            for i in range(1, self.nfe_t + 1):
                self.solve_dyn(self.PlantSample, mu_init=1e-08, iter_max=10)
                self.cycleSamPlant()
                self.load_iguess_dyndyn(self.PlantSample, self.dyn, i)
            print("I[[create_dyn]] Dynamic (full) model initialized.")

    @staticmethod
    def journalist(flag, iter, phase, message):
        """Method that writes a little message
        Args:
            flag (str): The flag
            iter (int): The current iteration
            phase (str): The phase
            message (str): The text message to display
        Returns:
            None"""
        iter = str(iter)
        print("-" * 120)
        if flag == 'W':
            print(flag + iter + "[[" + phase + "]]" + message + ".", file=sys.stderr)
        else:
            print(flag + iter + "[[" + phase + "]]" + message + "." + "-" * 20)
        print("-" * 120)

    # NMPC or just dyn?
    def cycleSamPlant_noisy(self, sigma_bar=0.0001):
        """Patches the initial conditions with the last result from the simulation with noise.
        Args:
            sigma_bar (float): The variance.
        Return
            None"""
        print("-" * 120)
        print("I[[cycleSamPlant]] Cycling initial state -- NOISY.")
        print("-" * 120)
        s = np.random.normal(0, sigma_bar)
        for x in self.states:
            x_ic = getattr(self.PlantSample, x + "_ic")
            v_tgt = getattr(self.PlantSample, x)
            for ks in x_ic.keys():
                if type(ks) != tuple:
                    ks = (ks,)
                x_ic[ks].value = value(v_tgt[(1, self.ncp_t) + ks])
                sigma = value(v_tgt[(1, self.ncp_t) + ks]) * s

                self.curr_state_noise[(x, ks)] = sigma
                tst_val = value(v_tgt[(1, self.ncp_t) + ks]) + sigma
                if tst_val < 0:
                    print("error", tst_val, x, ks)
                x_ic[ks].value = value(v_tgt[(1, self.ncp_t) + ks]) + sigma
        self._c_it += 1

    def create_predictor(self):
        self.PlantPred = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        self.PlantPred.name = "Dynamic Predictor"

    def predictor_step(self, ref, state_dict, **kwargs):
        """Predicted-state computation by forward simulation.
        Args:
            ref (pyomo.core.base.PyomoModel.ConcreteModel): Reference model (mostly for initialization)
            state_dict (str): Source of state. For nmpc = real, mhe = estimated

        It always loads the input from the input dictionary"""


        fe = kwargs.pop("fe", 1)
        if fe > 1:
            fe_src = "s"
        else:
            fe_src = "d"
        self.journalist("I", self._c_it, "predictor_step", "Predictor step")
        self.load_iguess_dyndyn(ref, self.PlantPred, fe, fe_src=fe_src)  #: Load the initial guess
        self.load_init_state_gen(self.PlantPred, src_kind="dict", state_dict=state_dict)  #: Load the initial state
        self.plant_uinject(self.PlantPred, src_kind="dict")  #: Load the current control
        self.solve_dyn(self.PlantPred, stop_if_nopt=True)
        self.journalist("I", self._c_it, "predictor_step", "Predictor step - Success")
        sinopt = False


    def plant_uinject(self, d_mod, src_kind, nsteps=5, skip_homotopy=False, **kwargs):
        """Attempt to solve the dynamic model with some source model input
        Args:
            d_mod (pyomo.core.base.PyomoModel.ConcreteModel): Model to be updated
            src_kind (str): Kind of update (default=dict)
            nsteps (int): The number of continuation steps (default=5)
        Keyword Args:
            src (pyomo.core.base.PyomoModel.ConcreteModel): Source model
            src_fe (int): Finite element from the source model"""

        self.journalist("I", self._c_it, "plant_input", "Continuation_plant, src_kind=" + src_kind)
        #: Inputs
        target = {}
        current = {}
        ncont_steps = nsteps
        sinopt = False
        if src_kind == "mod":
            src = kwargs.pop("src", None)
            if src:
                src_fe = kwargs.pop("src_fe", 1)
                for u in self.u:
                    src_var = getattr(src, u)
                    plant_var = getattr(d_mod, u)
                    target[u] = value(src_var[src_fe])
                    current[u] = value(plant_var[1])
                    self.journalist("I", self._c_it,
                                     "plant_input",
                                     "Target {:f}, Current {:f}, n_steps {:d}".format(target[u], current[u],
                                                                                      ncont_steps))
            else:
                pass  #error
        else:
            for u in self.u:
                plant_var = getattr(d_mod, u)
                target[u] = self.curr_u[u]
                current[u] = value(plant_var[1])
                self.journalist("I", self._c_it,
                                 "plant_input",
                                 "Target {:f}, Current {:f}, n_steps {:d}".format(target[u], current[u],
                                                                                  ncont_steps))
        tn = sum(target[u] for u in self.u)**(1/len(self.u))
        cn = sum(current[u] for u in self.u)**(1/len(self.u))
        pgap = abs((tn-cn)/cn)
        print("Current Gap /\% {:f}".format(pgap*100))
        if pgap < 1e-05:
            ncont_steps = 2
        if skip_homotopy:
            pass
        else:
            for i in range(0, ncont_steps):
                for u in self.u:
                    plant_var = getattr(d_mod, u)
                    plant_var[1].value += (target[u]-current[u])/ncont_steps
                    pgap -= pgap/ncont_steps
                    print("Continuation {:d} :Current {:s}\t{:f}\t :Gap /\% {:f}".format(i, u, value(plant_var[1]), pgap*100))
                if i == ncont_steps-1:
                    sinopt = False
                    tstv = self.solve_dyn(d_mod,
                                         o_tee=True,
                                         stop_if_nopt=False,
                                         print_level=2,
                                         max_cpu_time=120,
                                         print_user_options=False)
                else:
                    tstv = self.solve_dyn(d_mod, o_tee=False,
                                         stop_if_nopt=False,
                                         print_level=2,
                                         max_cpu_time=120,
                                         print_user_options=False)
                if tstv != 0:
                    try:
                        self.solve_dyn(d_mod, o_tee=True,
                                     max_cpu_time = 240,
                                     halt_on_ampl_error = True,
                                     tol = 1e-03,
                                     output_file = "failed_homotopy_d1.txt",
                                     stop_if_nopt=sinopt)
                    except (ApplicationError, ValueError):
                        print("Ipopt FAIL", file=sys.stderr)
                        self.PlantSample.write_nl(name="baddie.nl")
                        self.PlantSample.pprint(filename="baddie.txt")
                        self.PlantSample.snap_shot(filename="baddie.py")
                        self.PlantSample.report_zL(filename="bad_bounds")
                        self.solve_dyn(d_mod, o_tee=True,
                                       halt_on_ampl_error=True,
                                       bound_push=0.1,
                                       tol=1e-03,
                                       output_file="failed_homotopy_d2.txt",
                                       stop_if_nopt=sinopt, ma57_pivtol=1e-12, ma57_pre_alloc=5,
                                       linear_scaling_on_demand=True)
        for u in self.u:
            plant_var = getattr(d_mod, u)
            plant_var[1].value = target[u]  #: To be sure


    def update_u(self, src, **kwargs):
        """Update the current control(input) vector
        Args:
            **kwargs: Arbitrary keyword arguments.
        Returns:
            None
        Keyword Args:
            mod (pyomo.core.base.PyomoModel.ConcreteModel): The reference model (default d1)
            fe (int): The required finite element """
        fe = kwargs.pop("fe", 1)
        for u in self.u:
            uvar = getattr(src, u)
            self.curr_u[u] = value(uvar[fe])

    def update_state_real(self):
        for x in self.states:
            xvar = getattr(self.PlantSample, x)
            for j in self.state_vars[x]:
                self.curr_rstate[(x, j)] = value(xvar[1, self.ncp_t, j])

    def update_state_predicted(self):
        """For the olnmpc"""
        for x in self.states:
            xvar = getattr(self.PlantPred, x)
            for j in self.state_vars[x]:
                self.curr_pstate[(x, j)] = value(xvar[1, self.ncp_t, j])

    def load_init_state_gen(self, dmod, src_kind="mod", **kwargs):
        """Loads ref state for a forward simulation.
        Args:
            **kwargs: Arbitrary keyword arguments.
        Returns:
            None
        Keyword Args:
            src_kind (str) : if == mod use reference model, otw use the internal dictionary
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model (default d1)
            fe (int): The required finite element
            cp (int): The required collocation point
        """
        # src_kind = kwargs.pop("src_kind", "mod")
        self.journalist("I", self._c_it, "load_init_state_gen", "Load State to nmpc src_kind=" + src_kind)
        ref = kwargs.pop("ref", None)

        if src_kind == "mod":
            fe_t = getattr(ref, "fe_t")
            nfe = max(fe_t)
            cp_t = getattr(ref, "cp_t")
            ncp = max(cp_t)
            fe = kwargs.pop("fe", nfe)
            cp = kwargs.pop("cp", ncp)
            if not ref:
                self.journalist("W", self._c_it, "load_init_state_gen", "No model was given")
                self.journalist("W", self._c_it, "load_init_state_gen", "No update on state performed")
                return
            for x in self.states:
                xic = getattr(dmod, x + "_ic")
                xvar = getattr(dmod, x)
                xsrc = getattr(ref, x)
                for j in self.state_vars[x]:
                    val_src = value(xsrc[(fe, cp) + j])
                    xic[j].value = val_src
                    xvar[(1, 0) + j].set_value(val_src)
        else:
            state_dict = kwargs.pop("state_dict", None)
            if state_dict == "real":  #: Load from the real state dict
                for x in self.states:
                    xic = getattr(dmod, x + "_ic")
                    xvar = getattr(dmod, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_rstate[(x, j)]
                        xvar[(1, 0) + j].set_value(self.curr_rstate[(x, j)])
            elif state_dict == "estimated":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(dmod, x + "_ic")
                    xvar = getattr(dmod, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_estate[(x, j)]
                        xvar[(1, 0) + j].set_value(self.curr_estate[(x, j)])
            elif state_dict == "predicted":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(dmod, x + "_ic")
                    xvar = getattr(dmod, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_pstate[(x, j)]
                        xvar[(1, 0) + j].set_value(self.curr_pstate[(x, j)])
            else:
                self.journalist("W", self._c_it, "load_init_state_gen", "No dict w/state was specified")
                self.journalist("W", self._c_it, "load_init_state_gen", "No update on state performed")
                return

    def make_noisy(self, cov_dict, conf_level=2):
        """Contrived noise introduction for the plant"""
        self.PlantSample.name = "Noisy plant (d1)"
        k = 0
        for x in self.states:
            s = getattr(self.PlantSample, x)  #: state
            xicc = getattr(self.PlantSample, x + "_icc")
            xicc.deactivate()
            for j in self.state_vars[x]:
                self.xp_l.append(s[(1, 0) + j])
                self.xp_key[(x, j)] = k
                k += 1

        self.PlantSample.xS_pnoisy = Set(initialize=[i for i in range(0, len(self.xp_l))])  #: Create set of noisy_states
        self.PlantSample.w_pnoisy = Var(self.PlantSample.xS_pnoisy, initialize=0.0)  #: Model disturbance
        self.PlantSample.Q_pnoisy = Param(self.PlantSample.xS_pnoisy, initialize=1, mutable=True)
        self.PlantSample.obj_fun_noisy = Objective(sense=maximize,
                                          expr=0.5 *
                                              sum(self.PlantSample.Q_pnoisy[k] * self.PlantSample.w_pnoisy[k]**2 for k in self.PlantSample.xS_pnoisy)
                                          )
        self.PlantSample.ics_noisy = ConstraintList()

        k = 0
        for x in self.states:
            s = getattr(self.PlantSample, x)  #: state
            xic = getattr(self.PlantSample, x + "_ic")
            for j in self.state_vars[x]:
                expr = s[(1, 1) + j] == xic[j] + self.PlantSample.w_pnoisy[k]
                self.PlantSample.ics_noisy.add(expr)
                k += 1

        for key in cov_dict:
            vni = key
            v_i = self.xp_key[vni]
            self.PlantSample.Q_pnoisy[v_i].value = cov_dict[vni]
            self.PlantSample.w_pnoisy[v_i].setlb(-conf_level * cov_dict[vni])
            self.PlantSample.w_pnoisy[v_i].setub(conf_level * cov_dict[vni])

        with open("debug.txt", "w") as f:
            self.PlantSample.Q_pnoisy.display(ostream=f)
            self.PlantSample.obj_fun_noisy.pprint(ostream=f)
            self.PlantSample.ics_noisy.pprint(ostream=f)
            self.PlantSample.w_pnoisy.display(ostream=f)

    def randomize_noize(self, cov_dict):
        conf_level = np.random.randint(1, high=4)
        print("Confidence level", conf_level)
        for key in cov_dict:
            vni = key
            v_i = self.xp_key[vni]
            self.PlantSample.w_pnoisy[v_i].setlb(-conf_level * cov_dict[vni])
            self.PlantSample.w_pnoisy[v_i].setub(conf_level * cov_dict[vni])


    def deb_alg_sys_dyn(self, ddt=False):
        """Debugging the algebraic system"""
        # Fix differential states
        # Deactivate ODEs de_
        # Deactivate FE cont cp_
        # Deactivate IC _icc
        # Deactivate coll dvar_t_

        # Deactivate hyk
        for i in self.states:
            if ddt:
                x = getattr(self.PlantSample, "d" + i + "_dt")
            else:
                x = getattr(self.PlantSample, i)
            x.fix()
            # cp_con = getattr(self.PlantSample, "cp_" + i)
            # cp_con.deactivate()
            de_con = getattr(self.PlantSample, "de_" + i)
            de_con.deactivate()
            icc_con = getattr(self.PlantSample, i + "_icc")
            icc_con.deactivate()
            dvar_con = getattr(self.PlantSample, "dvar_t_" + i)
            dvar_con.deactivate()


        # self.lsmhe.pprint(filename="algeb_mod.txt")

    def gradients_tool(self):
        self.journalist("E", self._c_it, "GradientsTool", "Begin")
        src = self.PlantSample
        src.dum_objfun = Objective(expr=1, sense=minimize)
        self.PlantSample.var_order = Suffix(direction=Suffix.EXPORT)
        self.PlantSample.con_order = Suffix(direction=Suffix.EXPORT)



        src.pprint(filename="first.txt")
        #: Fix/Deactivate irrelevant stuff
        for var in src.component_objects(Var, active=True):
            if not var.is_indexed():
                var.fix()
            for index in var.keys():
                if type(index) != tuple:
                    var.fix()
                    continue
                try:
                    if index[1] == self.ncp_t:
                        continue
                    var[index].fix()
                except IndexError:
                    var.fix()
                    print("Variable not indexed by time", var.name, file=sys.stderr)
        for con in src.component_objects(Constraint, active=True):
            if not con.is_indexed():
                con.deactivate()
            for index in con.keys():
                if type(index) != tuple:
                    con.deactivate()
                    continue
                try:
                    if index[1] == self.ncp_t:
                        continue
                    con[index].deactivate()
                except IndexError:
                    con.deactivate()
                    print("Constraint not indexed by time", con.name, file=sys.stderr)

        for i in self.states:  #: deactivate collocation related equations
            # con = getattr(src, "cp_" + i)  #: continuation
            # con.deactivate()
            con = getattr(src, "dvar_t_" + i)  #: derivative vars
            con.deactivate()
            con = getattr(src, i + "_icc")  #: initial-conditions
            con.deactivate()
            var = getattr(src, "d" + i + "_dt")
            var.fix()
            var = getattr(src, i)
            var.fix()  #: left with the av
            con = getattr(src, "de_" + i)  #: initial-conditions
            con.deactivate()
        # colcount = 0
        # for i in src.component_data_objects(Var, active=True):
        #     if i.is_fixed():
        #         continue
        #     print(i)
        #     colcount += 1
        #     i.set_suffix_value(src.var_order, colcount)
        # print(colcount)


        self.PlantSample.write_nl(name="dgy.nl")
        sfxdict = dict()
        self.parse_rc("dgy.row", sfxdict)
        colcount = 1
        for ob in sfxdict.keys():
            con = getattr(src, sfxdict[ob][0])
            con[sfxdict[ob][1]].set_suffix_value(src.con_order, colcount)
            colcount += 1

        sfxdict = dict()
        self.parse_rc("dgy.col", sfxdict)
        colcount = 1
        for ob in sfxdict.keys():
            var = getattr(src, sfxdict[ob][0])
            var[sfxdict[ob][1]].set_suffix_value(src.var_order, colcount)
            colcount += 1
        print("Colcount\t",str(colcount), file=sys.stderr)
        src.write_nl(name="dgy.nl")
        #: Now dgx
        for var in src.component_objects(Var):
            var.fix()  #: Fix everything
        for i in self.states:
            var = getattr(src, i)
            for index in var.keys():
                try:
                    if index[1] == self.ncp_t:
                        var[index].unfix()
                except IndexError:
                    print("Something whent wrong :(\t", var.name, file=sys.stderr)

        self.PlantSample.write_nl(name="dgx.nl")
        
        sfxdict = dict()
        self.parse_rc("dgx.col", sfxdict)
        colcount = 1
        for ob in sfxdict.keys():
            var = getattr(src, sfxdict[ob][0])
            var[sfxdict[ob][1]].set_suffix_value(src.var_order, colcount)
            colcount += 1
        src.write_nl(name="dgx.nl")

        #: Now dfy
        for var in src.component_objects(Var):
            var.unfix()
        for var in src.component_objects(Var):
            if not var.is_indexed():
                var.fix()
            for index in var.keys():
                if type(index) != tuple:
                    var.fix()
                    continue
                try:
                    if index[1] == self.ncp_t:
                        continue
                    var[index].fix()
                except IndexError:
                    var.fix()
                    print("Variable not indexed by time", var.name, file=sys.stderr)

        for con in src.component_objects(Constraint):
            con.deactivate()  #: deactivate everything

        for i in self.states:  #: deactivate collocation related terms
            var = getattr(src, "d" + i + "_dt")
            var.fix()
            var = getattr(src, i)
            var.fix()  #: left with the av
            con = getattr(src, "de_" + i)
            for index in con.keys():
                if index[1] == self.ncp_t:
                    con[index].activate()
        self.PlantSample.reconstruct()
        # self.PlantSample.write_nl(name="dfy.nl")
        
        sfxdict = dict()
        self.parse_rc("dgx.col", sfxdict)
        colcount = 1
        for ob in sfxdict.keys():
            col = getattr(src, "de_" + sfxdict[ob][0])
            col[sfxdict[ob][1]].set_suffix_value(src.con_order, colcount)
            colcount += 1
        src.write_nl(name="dfy.nl")

        #: Now dfx
        for var in src.component_objects(Var):
            var.fix()  #: Fix everything

        for i in self.states:
            var = getattr(src, i)
            for index in var.keys():
                try:
                    if index[1] == self.ncp_t:
                        var[index].unfix()
                except IndexError:
                    print("Something whent wrong :(\t", var.name, file=sys.stderr)
            # var.pprint()
        self.PlantSample.reconstruct()
        src.pprint(filename="second.txt")
        self.PlantSample.write_nl(name="dfx.nl")


    @staticmethod
    def parse_rc(file, dsfx):

        f = open(file, "r")
        lines = f.readlines()
        for l in lines:
            i = re.split('\[|\]', l.strip())
            index = list()
            if len(i) < 2:
                continue
            for k in i[1].split(','):
                try:
                    k = int(k)
                except ValueError:
                    pass
                index.append(k)
            index = tuple(index)
            dsfx[l] = [i[0], index]
        f.close()

    @staticmethod
    def load_iguess_single(src, tgt, src_fe=1, tgt_fe=1):
        """Loads a set of values from a src model to a tgt model,
           the loading assumes a single value for all collocation points of a variable"""
        cp_src = getattr(src, "cp_t")
        tgt_src = getattr(tgt, "cp_t")
        # First check if we can do direct assignment of all the collocation points
        if len(cp_src.value) == len(tgt_src.value):
            print("These variables have the same number of cps")
            # passed test of len
        # else:  #: If not grab the last collocation point from src and patch this value to all cp from tgt
        sncp = max(cp_src)
        for vtgt in tgt.component_objects(Var, active=True):
            if not vtgt.is_indexed:
                break  #: Not indexed
            vs = getattr(src, vtgt.name)
            for k in vtgt.keys():
                if type(k) == tuple:
                    if k[0] == tgt_fe:
                        try:
                            vtgt[k].set_value(value(vs[(src_fe, sncp) + k[2:]]))
                        except IndexError:  #: Not indexed by collocation point
                            print("Variables without index set cp_t", end="\t")
                            print(vtgt.name, end="\t")
                            vtgt[k].set_value(value(vs[(src_fe,) + k[1:]]))
                            break
                    else:
                        pass
                else:
                    break  #: Single index (perhaps)

    def print_r_dyn(self):
        self.journalist("I", self._c_it, "print_r_dyn", "Results at" + os.getcwd())
        self.journalist("I", self._c_it, "print_r_dyn", "Results suffix " + self.res_file_suf)
        with open("res_dyn_" + self.res_file_suf + ".txt", "a") as f:
            for x in self.states:
                for j in self.state_vars[x]:
                    val = self.curr_rstate[(x, j)]
                    xvs = str(val)
                    f.write(xvs)
                    f.write('\t')
            f.write('\n')
            f.close()