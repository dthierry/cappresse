# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, Set, Constraint, Expression, Param, Suffix, maximize
from pyomo.core.base import ConstraintList, ConcreteModel, TransformationFactory
from pyomo.core.expr.numvalue import value
from pyomo.dae import *
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition, ReaderFactory, ResultsFormat
from pyomo.opt.results.results_ import SolverResults
import numpy as np
# import sys, time, re, os
from pyutilib.common._exceptions import ApplicationError
import datetime
from shutil import copyfile
from nmpc_mhe.aux.utils import t_ij, load_iguess, augment_model, augment_steady
from nmpc_mhe.aux.utils import clone_the_model, aug_discretization, create_bounds
import sys
import time
import re
import os
__author__ = "David Thierry @dthierry"  #: March 2018



class LogfileError(RuntimeError):
    """Exception raised when the log file name is not well defined"""

    def __init__(self, arg):
        self.args = arg


class SolfileError(Exception):
    """Exception raised when the sol file solve didn't go okay"""

    def __init__(self, arg):
        self.args = arg


class DynSolWeAreDone(RuntimeError):
    """Exception raised when wave our hands in desperation"""

    def __init__(self, *args, **kwargs):
        pass


class UnexpectedOption(RuntimeError):
    """This is not a valid option"""

    def __init__(self, *args, **kwargs):
        pass


class DynGen_DAE(object):
    """Default class for the Dynamic model"""

    def __init__(self, d_mod, hi_t, states, controls, **kwargs):

        # Base model
        self.d_mod = d_mod
        #: Discretization info
        self.nfe_t = kwargs.pop('nfe_t', 5)
        self.ncp_t = kwargs.pop('ncp_t', 3)

        self.k_aug_executable = kwargs.get('k_aug_executable', None)
        self.ipopt_executable = kwargs.get('ipopt_executable', None)
        self.dot_driver_executable = kwargs.get('dot_driver_executable', None)
        override_solver_check = kwargs.get('override_solver_check', False)
        self.parfois_v = kwargs.get('parfois_v', None)


        self.var_bounds = kwargs.get("var_bounds", None)
        create_bounds(self.d_mod, pre_clear_check=True)

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
        self._reftime = time.time()

        self.SteadyRef = clone_the_model(self.d_mod)
        augment_steady(self.SteadyRef)

        self.SteadyRef2 = object()

        self.PlantSample = clone_the_model(self.d_mod)
        augment_model(self.PlantSample, 1, self.ncp_t, new_timeset_bounds=(0, self.hi_t))

        aug_discretization(self.PlantSample, nfe=1, ncp=self.ncp_t)
        create_bounds(self.PlantSample, bounds=self.var_bounds, clear=True)
        # discretizer = TransformationFactory('dae.collocation')
        # discretizer.apply_to(self.PlantSample, nfe=1, ncp=self.ncp_t, scheme="LAGRANGE-RADAU")

        self.PlantPred = None
        self.SteadyRef.name = "SteadyRef"
        self.PlantSample.name = "PlantSample"

        if isinstance(self.ipopt_executable, str):
            self.ipopt = SolverFactory("ipopt",
                                       executable=self.ipopt_executable)

            self.asl_ipopt = SolverFactory("asl:ipopt",
                                           executable=self.ipopt_executable)
        else:
            self.ipopt = SolverFactory("ipopt")
            self.asl_ipopt = SolverFactory("asl:ipopt")

        if self.k_aug_executable:
            self.k_aug = SolverFactory("k_aug",
                                       executable=self.k_aug_executable)
            self.k_aug_sens = SolverFactory("k_aug",
                                       executable=self.k_aug_executable)
            if self.k_aug.available():
                pass
            else:
                self.k_aug = SolverFactory("k_aug")
                self.k_aug_sens = SolverFactory("k_aug")
                if self.k_aug.available():
                    pass
                elif override_solver_check:
                    pass
                else:
                    raise RuntimeError("k_aug not found")

        if self.dot_driver_executable:
            self.dot_driver = SolverFactory("dot_driver",
                                       executable=self.dot_driver_executable)
            if self.dot_driver.available():
                pass
            else:
                self.dot_driver = SolverFactory("dot_driver")
                if self.dot_driver.available():
                    pass
                elif override_solver_check:
                    pass
                else:
                    raise RuntimeError("k_aug not found")

        # self.k_aug.options["eig_rh"] = ""
        self.asl_ipopt.options["halt_on_ampl_error"] = "yes"
        self.SteadyRef.ofun = Objective(expr=1.0, sense=minimize)
        self.dyn = object()
        self.l_state = []
        self.l_vals = []
        self._iteration_count = 0
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
                self._u_plant[(i, t)] = value(u[0])
        self.curr_u = dict.fromkeys(self.u, 0.0)

        self.state_vars = {}
        self.curr_estate = {}  #: Current estimated state (for the olnmpc)
        self.curr_rstate = {}  #: Current real state (for the olnmpc)

        self.curr_meas = {}  #: Current measurement (for mhe)
        self.curr_state_offset = {}  #: Current offset of measurement
        self.curr_pstate = {}  #: Current offset of measurement
        self.curr_state_noise = {}  #: Current noise of the state

        self.curr_state_target = {}  #: Current target state
        self.curr_u_target = {}  #: Current control state

        self.xp_l = []
        self.xp_key = {}
        self.WhatHappensNext = 0.0

        with open("ipopt.opt", "w") as f:
            f.close()

    def load_iguess_steady(self):
        """"Call the method for loading initial guess from steady-state"""
        retval = self.solve_dyn(self.SteadyRef, bound_push=1e-07, rho_0=10000000.0)
        load_iguess(self.SteadyRef, self.PlantSample, 0, 0)
        self.cycleSamPlant()
        if retval:
            raise RuntimeError("The solution of the Steady-state problem failed")

    def get_state_vars(self, skip_solve=False):
        """Solves steady state model (SteadyRef)
        Args:
            None
        Return:
            None"""
        # Create a separate set of options for Ipopt
        if skip_solve:
            with open("ipopt.opt", "w") as f:
                f.write("max_iter 100\n")
                f.write("mu_init 1e-08\n")
                f.write("bound_push 1e-08\n")
                f.write("print_info_string yes\n")
                f.write("print_user_options yes\n")
                f.write("linear_solver ma57\n")
                f.close()
            ip = SolverFactory("")

            results = ip.solve(self.SteadyRef,
                               tee=True,
                               symbolic_solver_labels=False,
                               report_timing=True)

            self.SteadyRef.solutions.load_from(results)

        if not hasattr(self.d_mod, "t"):
            raise RuntimeError("The base model is missing a t object for the time domain")
        if not isinstance(self.d_mod.t, ContinuousSet):
            raise RuntimeError("The t object is not ContinuousSet\nMake t ContinuousSet.")
        time_steady = getattr(self.SteadyRef, "t")
        # Gather the keys for a given state and form the state_vars dictionary
        for x in self.states:
            self.state_vars[x] = []
            try:
                xv = getattr(self.SteadyRef, x)
            except AttributeError:  # delete this
                raise RuntimeError("State {} does not exists as a Var".format(x))
            if xv._implicit_subsets is None:
                if not xv.index_set() is time_steady:
                    raise RuntimeError("Var {} does not have t as part of its index set\n"
                                       "It can not be a State".format(x))
                else:
                    self.state_vars[x] = ((),)
            else:
                if time_steady in xv._implicit_subsets:
                    pass
                else:
                    raise RuntimeError("State {} does not contain the ContinuousSet t")
                # BUG: Is this a tuple? A: Yes
                remaining_set = xv._implicit_subsets[1]
                for j in range(2, len(xv._implicit_subsets)):
                    remaining_set *= xv._implicit_subsets[j]
                for index in remaining_set:
                    if isinstance(index, tuple):
                        self.state_vars[x].append(index)
                    else:
                        self.state_vars[x].append((index,))

            # for j in xv.keys():
            #     #: pyomo.dae only has one index for time
            #     if xv[j].stale:
            #         continue
            #     if isinstance(j[1:], tuple):
            #         self.state_vars[x].append(j[1:])
            #     else:
            #         self.state_vars[x].append((j[1:],))
        print(self.state_vars)
        # Get values for reference states and controls
        for x in self.states:
            try:
                xvar = getattr(self.SteadyRef, x)
            except AttributeError:  # delete this
                continue
            for j in self.state_vars[x]:
                self.curr_state_offset[(x, j)] = 0.0
                self.curr_state_noise[(x, j)] = 0.0
                self.curr_estate[(x, j)] = value(xvar[1, j])  #: for SteadyRef the relevant time index starts at 1
                self.curr_rstate[(x, j)] = value(xvar[1, j])
                # print(self.curr_rstate[(x, j)])
                self.curr_state_target[(x, j)] = value(xvar[1, j])
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
        self.journalist("I", self._iteration_count, "solve_steady_ref", "labels at " + self.res_file_suf)

    def solve_dyn(self, mod, keepsolve=False, **kwargs):
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
        ma57_pre_alloc = 1.5
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
        bound_relax_fact = kwargs.pop("bound_relax_fact", None)

        out_file = kwargs.pop("output_file", None)
        #rho_init = kwargs.pop("l1exactpenalty_rho0", )
        rho_0 = kwargs.pop("rho_0", None)
        l1_ = kwargs.pop("l1_mode", False)
        if out_file:
            if type(out_file) != str:
                self.journalist("E", self._iteration_count, "solve_dyn", "incorrect_output")
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
                self.journalist("E", self._iteration_count, "solve_dyn", "incorrect_output")
                print("jacobian_regularization_value is not float", file=sys.stderr)

                sys.exit()
        if jacRegExp:
            if type(jacRegExp) != float:
                self.journalist("E", self._iteration_count, "solve_dyn", "incorrect_output")
                print("jacobian_regularization_exponent is not float", file=sys.stderr)
                sys.exit()
        if mu_strategy:
            if mu_strategy != "monotone" and mu_strategy != "adaptive":
                self.journalist("E", self._iteration_count, "solve_dyn", "incorrect_output(mu_strategy)")
                print(mu_strategy)
                sys.exit()
        if mu_target:
            if type(mu_target) != float:
                self.journalist("E", self._iteration_count, "solve_dyn", "incorrect_output")
                print("mu_target is not float", file=sys.stderr)
                sys.exit()
        if print_level:
            if type(print_level) != int:
                self.journalist("E", self._iteration_count, "solve_dyn", "incorrect_output")
                print("print_level is not int", file=sys.stderr)
                sys.exit()
        if ma57_pivtol:
            if type(ma57_pivtol) != float:
                self.journalist("E", self._iteration_count, "solve_dyn", "incorrect_output")
                print("ma57_pivtol is not float", file=sys.stderr)
                sys.exit()
        if bound_push:
            if type(bound_push) != float:
                self.journalist("E", self._iteration_count, "solve_dyn", "incorrect_output")
                print("bound_push is not float", file=sys.stderr)
                sys.exit()

        name = mod.name

        self.journalist("I", self._iteration_count, "Solving with IPOPT\t", name)

        with open("ipopt.opt", "w") as f:
            #f.write("start_with_resto\tyes\n")
            #f.write("expect_infeasible_problem\tyes\n")
            if l1_:
                f.write("l1exactpenalty_objective_type\tobjective_inv\n")
            if isinstance(rho_0, float):
                f.write("l1exactpenalty_rho0\t")
                f.write(str(rho_0))
                f.write("\n")
            f.write("print_info_string\tyes\n")
            if isinstance(bound_relax_fact, float):
                f.write("bound_relax_factor\t" + str(bound_relax_fact) + "\n")
            else:
                f.write("bound_relax_factor\t0.0\n")
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
            f.write("output_file\t" + "out_capresse.txt" + "\n")
            f.write("file_print_level\t" + "6" + "\n")
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

        keepfiles = kwargs.pop("keepfiles", False)
        loadsolve = kwargs.pop("loadsolve", False)
        if loadsolve:
            solfile = kwargs.pop("solfile", None)
            if not solfile:
                raise SolfileError("Missing .sol file")
            else:
                self.load_solfile(d, solfile)

        if keepsolve:
            keepfiles = True

        # Solution attempt

        results = None
        try:
            results = solver_ip.solve(d,
                                      tee=o_tee,
                                      symbolic_solver_labels=False,
                                      report_timing=rep_timing,
                                      keepfiles=keepfiles, load_solutions=False)
        except (ApplicationError, ValueError):
            stop_if_nopt = 1
            d.write(filename="failure_.nl", io_options={"symbolic_solver_labels": True})

        if isinstance(results, SolverResults):
            if tag == "plant":  #: If this is the plant, don't load the solutions if there is a failure
                if results.solver.status != SolverStatus.ok or \
                        results.solver.termination_condition != TerminationCondition.optimal:
                    pass
                else:
                    d.solutions.load_from(results)
            else:
                d.solutions.load_from(results)
        if keepsolve:
            self.write_solfile(d, tag, solve=False)  #: solve false otherwise it'll call sol_dyn again
        wantparams = kwargs.pop("wantparams", False)
        if wantparams:
            self.param_writer(d, tag)

        # Append to the logger
        if tag:
            with open("log_ipopt_" + tag + "_" + self.res_file_suf + ".txt", "a") as global_log:
                with open(out_file, "r") as filelog:
                    global_log.write(
                        "--\t" + str(self._iteration_count) + "\t" + str(datetime.datetime.now()) + "\t" + "-" * 50)
                    for line in filelog:
                        global_log.write(line)
                    filelog.close()
                global_log.close()

        if isinstance(results, SolverResults):
            #: Check termination
            if results.solver.status == SolverStatus.ok and results.solver.termination_condition == TerminationCondition.optimal:
                self.journalist("I", self._iteration_count, "solve_dyn", " Model solved to optimality")
                # d.solutions.load_from(results)
                self._stall_iter = 0
                if want_stime and rep_timing:
                    self.ip_time = self.ipopt._solver_time_x
                if not skip_mult_update:
                    mod.ipopt_zL_in.update(mod.ipopt_zL_out)
                    mod.ipopt_zU_in.update(mod.ipopt_zU_out)

                return 0
        if stop_if_nopt:
            d.write(filename="failure_done_.nl", io_options={"symbolic_solver_labels": True})
            self.journalist("E", self._iteration_count, "solve_dyn", "Not-optimal. Stoping")
            self.journalist("E", self._iteration_count, "solve_dyn",
                            "Problem\t" + d.name + "\t" + str(self._iteration_count))
            raise DynSolWeAreDone("Ipopt: we are done :(")
            # sys.exit()
        self.journalist("W", self._iteration_count, "solve_dyn", "Not-optimal.")
        return 1

    def cycleSamPlant(self, plant_step=False):
        """Patches the initial conditions with the last result from the simulation
        Args:
            None
        Return
            None"""

        print("-" * 120)
        print("I[[cycleSamPlant]] Cycling initial state. Iteration timing(s):", end="\t")
        newtime = time.time()
        print(str(newtime - self._reftime))
        self._reftime = newtime
        t = t_ij(self.PlantSample.t, 0, self.ncp_t)
        for x in self.states:
            x_ic = getattr(self.PlantSample, x + "_ic")
            v_tgt = getattr(self.PlantSample, x)
            if not x_ic.is_indexed():
                x_ic.value = value(v_tgt[t])  #: this has got to be true
            else:
                for ks in x_ic.keys():
                    if not isinstance(ks, tuple):
                        ks = (ks,)
                    x_ic[ks].value = value(v_tgt[(t,) + ks])
                    v_tgt[(0,) + ks].set_value(value(v_tgt[(t,) + ks]))
        if plant_step:
            self._iteration_count += 1

    def create_dyn(self, initialize=True):
        # type: (bool) -> None
        """
        Creates a dynamic simulation plant with self.nfe_t finite elements. Used mostly for debugging purposes.

        Args:
            initialize (bool): True if the marching-forward finite-per-finite element is desired.
        """
        print("-" * 120)
        print("I[[create_dyn]] Dynamic (full) model created.")
        print("-" * 120)

        self.dyn = clone_the_model(self.d_mod) #(self.nfe_t, self.ncp_t, _t=self._t)
        augment_model(self.dyn, self.nfe_t, self.ncp_t, new_timeset_bounds=(0, self._t))
        self.dyn.name = "full_dyn"
        # self.load_d_s(self.dyn)
        load_iguess(self.SteadyRef, self.PlantSample, 0, 0)
        aug_discretization(self.dyn, nfe=self.nfe_t, ncp=self.ncp_t)
        # discretizer = TransformationFactory('dae.collocation')
        # discretizer.apply_to(self.dyn, nfe=self.nfe_t, ncp=self.ncp_t, scheme="LAGRANGE-RADAU")

        if initialize:
            # self.load_d_s(self.PlantSample)
            load_iguess(self.SteadyRef, self.PlantSample, 0, 0)

            for i in range(0, self.nfe_t):
                self.solve_dyn(self.PlantSample, mu_init=1e-08, iter_max=10, o_tee=True)
                self.cycleSamPlant()
                print(i, "Current finite element")
                print(self.dyn.nfe_t)
                load_iguess(self.PlantSample, self.dyn, 0, i)
            print("I[[create_dyn]] Dynamic (full) model initialized.")

    @staticmethod
    def journalist(flag, i, phase, message):
        """Method that writes a little message
        Args:
            flag (str): The flag
            i (int): The current iteration
            phase (str): The phase
            message (str): The text message to display
        Returns:
            None"""
        i = str(i)
        print("-==-" * 15)

        if flag == 'W':
            print(flag + i + "[[" + phase + "]]" + message + ".", file=sys.stderr)
        # print to file warning
        elif flag == 'E':
            print("Fatal error", file=sys.stderr)
            print(flag + i + "[[" + phase + "]]" + message + "." + "-" * 20)
        else:
            print(flag + i + "[[" + phase + "]]" + message + "." + "-" * 20)
        # print("-" * 120)

    def create_predictor(self):
        self.PlantPred = clone_the_model(self.d_mod)  # (1, self.ncp_t, _t=self.hi_t)
        augment_model(self.PlantPred, 1, self.ncp_t, new_timeset_bounds=(0, self.hi_t))

        self.PlantPred.name = "Dynamic Predictor"
        aug_discretization(self.PlantPred, nfe=1, ncp=self.ncp_t)

    def predictor_step(self, ref, state_dict, **kwargs):
        """Predicted-state computation by forward simulation.
        Args:
            ref (ConcreteModel): Reference model (mostly for initialization)
            state_dict (str): Source of state. For nmpc = real, mhe = estimated

        It always loads the input from the input dictionary"""

        fe = kwargs.pop("fe", 0)
        if fe > 0:
            fe_src = "s"
        else:
            fe_src = "d"
        self.journalist("I", self._iteration_count, "predictor_step", "Predictor step")
        load_iguess(ref, self.PlantPred, fe, 0)  #: Load the initial guess
        self.load_init_state_gen(self.PlantPred, src_kind="dict", state_dict=state_dict)  #: Load the initial state
        self.plant_uinject(self.PlantPred, src_kind="dict")  #: Load the current control
        self.solve_dyn(self.PlantPred, stop_if_nopt=True)
        self.journalist("I", self._iteration_count, "predictor_step", "Predictor step - Success")
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
        self.journalist("I", self._iteration_count, "plant_input", "Continuation_plant, src_kind=" + src_kind)
        #: Inputs
        target = {}
        current = {}
        ncont_steps = nsteps
        sinopt = False
        if src_kind == "mod":
            src = kwargs.pop("src", None)
            if src:
                src_fe = kwargs.pop("src_fe", 0)
                for u in self.u:
                    src_var = getattr(src, u)
                    plant_var = getattr(d_mod, u)
                    target[u] = value(src_var[src_fe])
                    current[u] = value(plant_var[0])
                    self.journalist("I", self._iteration_count,
                                    "plant_input",
                                    "Target {:f}, Current {:f}, n_steps {:d}".format(target[u], current[u],
                                                                                     ncont_steps))
            else:
                raise ValueError("Unexpeted src_kind %s" % src_kind)
        elif src_kind == "dict":
            for u in self.u:
                plant_var = getattr(d_mod, u)
                target[u] = self.curr_u[u]  #: Value to be injected
                current[u] = value(plant_var[0])
                self.journalist("I", self._iteration_count,
                                "plant_input",
                                "Target {:f}, Current {:f}, n_steps {:d}".format(target[u], current[u],
                                                                                 ncont_steps))
        else:
            raise UnexpectedOption("src_kind is not not valid")

        tn = sum(target[u] for u in self.u) ** (1 / len(self.u))
        cn = sum(current[u] for u in self.u) ** (1 / len(self.u))
        pgap = abs((tn - cn) / cn)
        print("Current Gap /\% {:f}".format(pgap * 100))
        if pgap < 1e-05:
            ncont_steps = 2
        if skip_homotopy:
            pass
        else:
            for i in range(0, ncont_steps):
                for u in self.u:
                    plant_var = getattr(d_mod, u)
                    plant_var[0].value += (target[u] - current[u]) / ncont_steps
                    pgap -= pgap / ncont_steps
                    print("Continuation {:d} :Current {:s}\t{:f}\t :Gap /\% {:f}".format(i, u, value(plant_var[0]),
                                                                                         pgap * 100))
                if i == ncont_steps - 1:
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
                                       max_cpu_time=240,
                                       halt_on_ampl_error=True,
                                       tol=1e-03,
                                       output_file="failed_homotopy_d1.txt",
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
                                       stop_if_nopt=sinopt,
                                       ma57_pivtol=1e-12,
                                       ma57_pre_alloc=5,
                                       linear_scaling_on_demand=True)
        for u in self.u:
            plant_var = getattr(d_mod, u)
            for key in plant_var.keys():
                plant_var[key].value = target[u]  #: To be sure

    def update_u(self, src, **kwargs):
        """Update the current control(input) vector
        Args:
            **kwargs: Arbitrary keyword arguments.
        Returns:
            None
        Keyword Args:
            mod (pyomo.core.base.PyomoModel.ConcreteModel): The reference model (default d1)
            fe (int): The required finite element """
        stat = 0
        if hasattr(src, "is_steady"):
            if src.is_steady:
                fe = 1
            else:
                fe = 0
        else:
            fe = kwargs.pop("fe", 0)
        for u in self.u:
            uvar = getattr(src, u)
            vu = value(uvar[fe])
            self.curr_u[u] = vu
            if uvar[fe].lb is None:
                pass
            else:
                if vu < uvar[fe].lb:
                    stat = 1
            if uvar[fe].ub is None:
                pass
            else:
                if vu > uvar[fe].ub:
                    stat = 1
        return stat
            

    def update_state_real(self):
        for x in self.states:
            xvar = getattr(self.PlantSample, x)
            t = t_ij(self.PlantSample.t, 0, self.ncp_t)
            for j in self.state_vars[x]:
                self.curr_rstate[(x, j)] = value(xvar[t, j])

    def update_state_predicted(self, src="estimated"):
        """Make a prediction for the next state"""

        if self.PlantPred:
            print(self.PlantPred)
        else:
            print(self.PlantPred)
            self.create_predictor()
            load_iguess(self.SteadyRef, self.PlantSample, 0, 0)
        if src == "estimated":
            self.load_init_state_gen(self.PlantPred, src_kind="dict", state_dict="estimated")  #: Load the initial state
        else:
            self.load_init_state_gen(self.PlantPred, src_kind="dict", state_dict="real")  #: Load the initial state
        #: See if this works
        stat = self.solve_dyn(self.PlantPred, skip_update=True,
                              iter_max=250,
                              stop_if_nopt=True,
                              jacobian_regularization_value=1e-02,
                              linear_scaling_on_demand=True, tag="lsmhe")
        for x in self.states:
            xvar = getattr(self.PlantSample, x)
            t = t_ij(self.PlantSample.t, 0, self.ncp_t)
            for j in self.state_vars[x]:
                self.curr_pstate[(x, j)] = value(xvar[t, j])

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
        self.journalist("I", self._iteration_count, "load_init_state_gen", "Load State to nmpc src_kind=" + src_kind)
        ref = kwargs.pop("ref", None)

        if src_kind == "mod":
            fe_t = getattr(ref, "nfe_t")
            cp_t = getattr(ref, "ncp_t")
            #: in place if a particular set of fe and cp is desired
            fe = kwargs.pop("fe", fe_t)
            cp = kwargs.pop("cp", cp_t)
            if not ref:
                self.journalist("W", self._iteration_count, "load_init_state_gen", "No model was given")
                self.journalist("W", self._iteration_count, "load_init_state_gen", "No update on state performed")
                return
            tgt_tS = getattr(ref, "t")
            t = t_ij(tgt_tS, fe, cp)
            for x in self.states:
                xic = getattr(dmod, x + "_ic")
                xvar = getattr(dmod, x)
                xsrc = getattr(ref, x)
                for j in self.state_vars[x]:
                    val_src = value(xsrc[(t,) + j])
                    xic[j].value = val_src
                    xvar[(0,) + j].set_value(val_src)
        else:
            state_dict = kwargs.pop("state_dict", None)
            if state_dict == "real":  #: Load from the real state dict
                for x in self.states:
                    xic = getattr(dmod, x + "_ic")
                    xvar = getattr(dmod, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_rstate[(x, j)]
                        xvar[(0,) + j].set_value(self.curr_rstate[(x, j)])
            elif state_dict == "estimated":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(dmod, x + "_ic")
                    xvar = getattr(dmod, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_estate[(x, j)]
                        xvar[(0,) + j].set_value(self.curr_estate[(x, j)])
            elif state_dict == "predicted":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(dmod, x + "_ic")
                    xvar = getattr(dmod, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_pstate[(x, j)]
                        xvar[(0,) + j].set_value(self.curr_pstate[(x, j)])
            else:
                self.journalist("W", self._iteration_count, "load_init_state_gen", "No dict w/state was specified")
                self.journalist("W", self._iteration_count, "load_init_state_gen", "No update on state performed")
                return

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

    def print_r_dyn(self):
        self.journalist("I", self._iteration_count, "print_r_dyn", "Results at" + os.getcwd())
        self.journalist("I", self._iteration_count, "print_r_dyn", "Results suffix " + self.res_file_suf)
        with open("res_dyn_" + self.res_file_suf + ".txt", "a") as f:
            for x in self.states:
                for j in self.state_vars[x]:
                    val = self.curr_rstate[(x, j)]
                    xvs = str(val)
                    f.write(xvs)
                    f.write('\t')
            f.write('\n')
            f.close()
        with open("res_noiselvl_" + self.res_file_suf + ".txt", "a") as f:
            f.write(str(self.WhatHappensNext))
            f.write('\n')
            f.close()

        if self.parfois_v:
            for v in self.parfois_v:
                var = getattr(self.PlantSample, v)
                if var.is_indexed() and var.index_set().dimen > 1:
                    s = var.index_set().set_tuple[1:]
                    ss = s[0]
                    if len(s) > 1:
                        for j in s[:1]:
                            ss *= j
                    for j in ss:
                        idx = ((self.PlantSample.t.last()),)
                        if not isinstance(j, tuple):
                            j = ((j),)
                        idx = idx + j
                        var_val = value(var[idx])
                        with open("res_parfois_" + self.res_file_suf + ".txt", "a") as f:
                            f.write(str(var_val))
                            f.write('\t')
            with open("res_parfois_" + self.res_file_suf + ".txt", "a") as f:
                f.write('\n')






    @staticmethod
    def which(str_program):
        """Literally from stackoverflow. Returns true if program is in path"""

        def is_exe(fpath):
            return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

        fpath, fname = os.path.split(str_program)
        if fpath:
            if is_exe(str_program):
                return str_program
        else:
            for path in os.environ["PATH"].split(os.pathsep):
                exe_file = os.path.join(path, str_program)
                if is_exe(exe_file):
                    return exe_file

    def param_writer(self, mod, tag):
        """Writes the current mutable parameters to a dict for reloading them later"""
        # with open(filename, "w") as tgt:
        #     tgt.write("# Params\n\n")
        #     for p in mod.component_objects(Param):
        #         if p._mutable:
        #             if p.keys()[0] or type(p.keys()[0]) == int:
        #                 for k in p.keys():
        #                     pn = p.name
        #                     pv = value(p[k])
        #                     tgt.write(pn + "\t" + str(k) + "\t" + str(pv) + "\n")
        #             else:
        #                 pn = p.name
        #                 print(p.name)
        #                 print(p.keys())
        #                 pv = value(p)
        #                 tgt.write(pn + "\t" + str(pv) + "\n")
        #     tgt.close()
        # method 2
        filename = tag.replace(" ", "") + \
                   "_" + self.res_file_suf + "_" + \
                   str(self._iteration_count) + ".json"
        params = dict()
        for p in mod.component_objects(Param):
            if p._mutable:
                # if p.keys()[0] or type(p.keys()[0]) == int:
                for k in p.keys():
                    pn = p.name
                    pv = value(p[k])
                    key = (pn + "," + str(k))
                    params[key] = pv
        import json
        with open(filename, "w") as f:
            json.dump(params, f)
            f.close()
        path = os.getcwd()
        f = path + "/" + filename
        self.journalist("I", self._iteration_count, "param_writer", "name:\t\n\t" + filename)
        self.journalist("I", self._iteration_count, "param_writer", "path:\t\n\t" + f)

    @staticmethod
    def param_reader(mod, filename):
        import json
        with open(filename, "r") as f:
            params = json.load(f)
            for p in mod.component_objects(Param):
                if p._mutable:
                    for k in p.keys():
                        pn = p.name
                        # pv = value(p[k])
                        key = (pn + "," + str(k))
                        p[k].value = params[key]

    def write_solfile(self, mod, tag, solve=True, **kwargs):
        """Attempts to write the sol file from a particular run"""

        filename = tag.replace(" ", "") + \
                   "_" + self.res_file_suf + "_" + \
                   str(self._iteration_count) + ".sol"
        path = os.getcwd()
        f = path + "/" + filename
        if solve:
            tst = self.solve_dyn(mod, keepfiles=True, **kwargs)
            if tst:
                raise SolfileError("Solution was not optimal")
        solf = self.ipopt._soln_file
        copyfile(solf, f)
        self.journalist("I", self._iteration_count, "write_solfile", "name:\t\n\t" + filename)
        self.journalist("I", self._iteration_count, "write_solfile", "path:\t\n\t" + f)

    def load_solfile(self, mod, solfilename):
        """Attempts to read the solfile and load it into the corresponding mod"""
        _, smap_id = mod.write("dummy.nl", format=ProblemFormat.nl)
        os.remove("dummy.nl")
        smap = mod.solutions.symbol_map[smap_id]
        reader = ReaderFactory(ResultsFormat.sol)
        self.journalist("I", self._iteration_count, "load_solfile", "name:\t\n\t" + solfilename)
        results = reader(solfilename)
        results._smap = smap
        mod.solutions.load_from(results)

    def set_iteration_count(self, iteration_count=0):
        """Change the iteration count"""
        self._iteration_count = iteration_count
        self.journalist("I", self._iteration_count, "set_iteration_count",
                        "The iteration count has changed")

    def noisy_plant_manager(self, sigma=0.01, action="apply", update_level=False):
        """A simple way to introduce noise to the plant"""
        #: Notice that this is one possible way to introduce it
        if update_level:
            self.WhatHappensNext = np.random.normal(0, sigma)
        for state in self.states:
            x_ic = getattr(self.PlantSample, state + "_ic")
            for key in x_ic.keys():
                if action == "apply":
                    x_ic[key].value += x_ic[key].value * self.WhatHappensNext
                elif action == "remove":
                    x_ic[key].value -= x_ic[key].value * self.WhatHappensNext
                else:
                    raise ValueError("Unexpected action %s" % action)
        if action == "remove":
            self.journalist("I", self._iteration_count, "noisy_plant_manager", "Noise removed")


