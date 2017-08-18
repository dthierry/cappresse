#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix, maximize
from pyomo.core.base import ConstraintList
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition

import numpy as np
import sys, time

__author__ = "David M Thierry @dthierry"


class DynGen(object):
    def __init__(self, **kwargs):
        self.int_file_mhe_suf = int(time.time())
        self.res_file_mhe_suf = str(self.int_file_mhe_suf)
        self.d_mod = kwargs.pop('d_mod', None)

        self.nfe_t = kwargs.pop('nfe_t', 5)
        self.ncp_t = kwargs.pop('nfe_t', 3)

        self._t = kwargs.pop('_t', 100)
        self.states = kwargs.pop('states', [])
        self.u = kwargs.pop('u', [])  #: The inputs (controls)

        self.hi_t = self._t/self.nfe_t
        self.ss = self.d_mod(1, 1, steady=True)
        self.ss2 = object
        self.d1 = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        self.d2 = object()
        self.ss.name = "ss"
        self.d1.name = "d1"

        self.ipopt = SolverFactory("ipopt")
        self.asl_ipopt = SolverFactory("asl:ipopt")
        self.k_aug = SolverFactory("k_aug",
                                   executable="/home/dav0/k2/KKT_matrix/src/kmatrix/k_aug")
        self.k_aug_sens = SolverFactory("k_aug",
                                        executable="/home/dav0/k2/KKT_matrix/src/kmatrix/k_aug")
        self.dot_driver = SolverFactory("dot_driver",
                                        executable="/home/dav0/k2/KKT_matrix/src/kmatrix/dot_driver/dot_driver")

        # self.k_aug.options["eig_rh"] = ""
        self.asl_ipopt.options["halt_on_ampl_error"] = "yes"

        self.ipopt.options["print_user_options"] = "yes"
        # self.k_aug.options["deb_kkt"] = ""

        self.ss.ofun = Objective(expr=1, sense=minimize)
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
            u = getattr(self.d1, i)
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

    def load_iguess_ss(self):
        """"Call the method for loading initial guess from steady-state"""
        self.ss.init_steady_ref()

    def solve_ss(self):
        """Solves steady state model
        Args:
            None
        Return:
            None"""
        # self.k_aug.solve(self.ss, tee=True, symbolic_solver_labels=True)
        with open("ipopt.opt", "w") as f:
            f.write("max_iter 100\n")
            f.write("mu_init 1e-08\n")
            f.write("bound_push 1e-08\n")
            f.close()
        ip = SolverFactory("asl:ipopt")
        ip.options["halt_on_ampl_error"] = "yes"
        ip.options["print_user_options"] = "yes"
        ip.options["linear_solver"] = "ma57"
        results = ip.solve(self.ss, tee=True, symbolic_solver_labels=True, report_timing=True)
        self.ss.solutions.load_from(results)
        for x in self.states:
            self.state_vars[x] = []
            try:
                xv = getattr(self.ss, x)
            except AttributeError:  # delete this
                continue
            for j in xv.keys():
                if xv[j].stale:
                    continue
                if type(j[2:]) == tuple:
                    self.state_vars[x].append(j[2:])
                else:
                    self.state_vars[x].append((j[2:],))

        for x in self.states:
            try:
                xvar = getattr(self.ss, x)
            except AttributeError:  # delete this
                continue
            for j in self.state_vars[x]:
                self.curr_state_offset[(x, j)] = 0.0
                self.curr_state_noise[(x, j)] = 0.0
                self.curr_estate[(x, j)] = value(xvar[1, 1, j])
                self.curr_rstate[(x, j)] = value(xvar[1, 1, j])
                self.curr_state_target[(x, j)] = value(xvar[1, 1, j])
        for u in self.u:
            uvar = getattr(self.ss, u)
            self.curr_u_target[u] = value(uvar[1])
            self.curr_u[u] = value(uvar[1])



        # self.ss.write(filename="ss.nl",format=ProblemFormat.nl, io_options={"symbolic_solver_labels":True})

    def load_d_s(self, dmod):
        """Loads the solution of the steady state model into the dynamic
        Args:
            dmod (pyomo.core.base.PyomoModel.ConcreteModel): Target model
        Return:
            None"""
        s = self.ss
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

    def solve_d(self, mod, **kwargs):
        """Solves dynamic model
        Args:
            mod (pyomo.core.base.PyomoModel.ConcreteModel): Target model
        Return:
            int: 0 if success 1 otw"""
        d = mod
        mu_init = 0.1
        iter_max = 100
        max_cpu_time = 1e+06
        linear_solver = "ma57"
        ma57_pre_alloc = 1.05
        ma57_automatic_scaling = "no"
        ma57_small_pivot_flag = 0

        # o_tee = True
        stop_if_nopt = False
        want_stime = False  #: Updates internal time variable
        rep_timing = False

        if kwargs.get("stop_if_nopt"):
            stop_if_nopt = kwargs["stop_if_nopt"]
        if kwargs.get("want_stime"):
            want_stime = kwargs["want_stime"]
        if kwargs.get("rep_timing"):
            rep_timing = kwargs["rep_timing"]
        if kwargs.get("stop_if_nopt"):
            stop_if_nopt = kwargs["stop_if_nopt"]
        if kwargs.get("iter_max"):
            iter_max = kwargs["iter_max"]
        if kwargs.get("mu_init"):
            mu_init = kwargs["mu_init"]


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


        name = mod.name

        self.journalizer("I", self._c_it, "Solving with IPOPT", name)

        with open("ipopt.opt", "w") as f:
            f.write("print_info_string\tyes\n")
            f.write("max_iter\t" + str(iter_max) + "\n")
            f.write("mu_init\t" + str(mu_init) + "\n")
            f.write("max_cpu_time\t" + str(max_cpu_time) + "\n")
            f.write("linear_solver\t" + linear_solver + "\n")
            f.write("ma57_pre_alloc\t" + str(ma57_pre_alloc) + "\n")
            f.write("ma57_automatic_scaling\t" + ma57_automatic_scaling + "\n")
            f.write("ma57_small_pivot_flag\t" + str(ma57_small_pivot_flag) + "\n")
            if warm_start:
                f.write("warm_start_init_point\t" + "yes" + "\n")
                f.write("warm_start_bound_push\t" + "1e-06" + "\n")
                f.write("mu_init\t" + "0.001" + "\n")
            # f.write("mu_init 1e-08\n")
            # f.write("halt_on_ampl_error yes")
            f.close()
            # self.ipopt.options["halt_on_ampl_error"] = "yes"
            if halt_on_ampl_error:
                solver_ip = self.asl_ipopt
            else:
                solver_ip = self.ipopt
        results = solver_ip.solve(d, tee=o_tee, symbolic_solver_labels=True, report_timing=rep_timing)
        if (results.solver.status == SolverStatus.ok) and \
                (results.solver.termination_condition == TerminationCondition.optimal):
            self.journalizer("I", self._c_it, "solve_d", " Model solved to optimality")
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
                self.journalizer("E", self._c_it, "solve_d", "Not-optimal. Stoping")
                sys.exit()
            self.journalizer("W", self._c_it, "solve_d", "Not-optimal.")
            return 1

    def cycle_ics(self, plant_step=False):
        """Patches the initial conditions with the last result from the simulation
        Args:
            None
        Return
            None"""
        print("-" * 120)
        print("I[[cycle_ics]] Cycling initial state.")
        print("-" * 120)
        for x in self.states:
            x_ic = getattr(self.d1, x + "_ic")
            v_tgt = getattr(self.d1, x)
            for ks in x_ic.keys():
                if type(ks) != tuple:
                    ks = (ks,)
                x_ic[ks].value = value(v_tgt[(1, self.ncp_t) + ks])
                v_tgt[(1, self.ncp_t) + ks].set_value(value(v_tgt[(1, self.ncp_t) + ks]))
        if plant_step:
            self._c_it += 1

    def load_d_d(self, src, d_mod, fe, fe_src='d'):
        """Loads the solution of the src state model into the d_mod, i.e. src-->d_mod
        Args:
            src (pyomo.core.base.PyomoModel.ConcreteModel): Model with source values
            d_mod (pyomo.core.base.PyomoModel.ConcreteModel): Model whose values are to be assigned
            fe (int): The d_mod's finite element
            fe_src (str):
        Return:
            None"""
        for vs in src.component_objects(Var, active=True):
            if vs.getname()[-7:] == "_pnoisy":
                continue
            vd = getattr(d_mod, vs.getname())
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
                            for j in range(0, self.ncp_t + 1):
                                vd[(fe, j) + kj].set_value(value(vs[ks]))
                        else:
                            for j in range(1, self.ncp_t + 1):
                                vd[(fe, j) + kj].set_value(value(vs[ks]))
                    elif fe_src == 's':
                        if vs.getname() in self.states:  #: States start at 0
                            for j in range(0, self.ncp_t + 1):
                                if ki == (fe, j):
                                    vd[(1, j) + kj].set_value(value(vs[ks]))
                                else:
                                    continue
                        else:
                            for j in range(1, self.ncp_t + 1):
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
            self.load_d_s(self.d1)
            for i in range(1, self.nfe_t + 1):
                self.solve_d(self.d1, mu_init=1e-08, iter_max=10)
                self.cycle_ics()
                self.load_d_d(self.d1, self.dyn, i)
            print("I[[create_dyn]] Dynamic (full) model initialized.")

    @staticmethod
    def journalizer(flag, iter, phase, message):
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
    def cycle_ics_noisy(self, sigma_bar=0.0001):
        """Patches the initial conditions with the last result from the simulation with noise.
        Args:
            sigma_bar (float): The variance.
        Return
            None"""
        print("-" * 120)
        print("I[[cycle_ics]] Cycling initial state -- NOISY.")
        print("-" * 120)
        s = np.random.normal(0, sigma_bar)
        for x in self.states:
            x_ic = getattr(self.d1, x + "_ic")
            v_tgt = getattr(self.d1, x)
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
        self.d2 = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        self.d2.name = "Dynamic Predictor"

    def predictor_step(self, ref, **kwargs):
        """Predicted-state computation by forward simulation.
        Args:
            ref (pyomo.core.base.PyomoModel.ConcreteModel): Reference model (mostly for initialization)"""
        fe = kwargs.pop("fe", 1)
        if fe > 1:
            fe_src = "s"
        else:
            fe_src = "d"
        self.journalizer("I", self._c_it, "predictor_step", "Predictor step")
        self.load_d_d(ref, self.d2, fe, fe_src=fe_src)  #: Load the initial guess
        self.load_init_state_gen(self.d2, src_kind="dict", state_dict="estimated")  #: Load the initial state
        self.plant_input_gen(self.d2, src_kind="dict")  #: Load the current control
        self.solve_d(self.d2)
        self.journalizer("I", self._c_it, "predictor_step", "Predictor step - Success")

    def plant_input_gen(self, d_mod, src_kind, nsteps=5, **kwargs):
        """Attempt to solve the dynamic model with some source model input
        Args:
            d_mod (pyomo.core.base.PyomoModel.ConcreteModel): Model to be updated
            src_kind (str): Kind of update (default=dict)
            nsteps (int): The number of continuation steps (default=5)
        Keyword Args:
            src (pyomo.core.base.PyomoModel.ConcreteModel): Source model
            src_fe (int): Finite element from the source model"""

        self.journalizer("I", self._c_it, "plant_input", "Continuation_plant, src_kind=" + src_kind)
        #: Inputs
        target = {}
        current = {}
        ncont_steps = nsteps
        if src_kind == "mod":
            src = kwargs.pop("src", None)
            if src:
                src_fe = kwargs.pop("src_fe", 1)
                for u in self.u:
                    src_var = getattr(src, u)
                    plant_var = getattr(d_mod, u)
                    target[u] = value(src_var[src_fe])
                    current[u] = value(plant_var[1])
                    self.journalizer("I", self._c_it,
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
                self.journalizer("I", self._c_it,
                                 "plant_input",
                                 "Target {:f}, Current {:f}, n_steps {:d}".format(target[u], current[u],
                                                                                  ncont_steps))
        for i in range(0, ncont_steps):
            for u in self.u:
                plant_var = getattr(d_mod, u)
                plant_var[1].value += (target[u]-current[u])/ncont_steps
                print("Continuation :Current {:s}\t{:f}".format(u, value(plant_var[1])))
            if i == ncont_steps:
                self.solve_d(d_mod, o_tee=False, stop_if_nopt=True)
            else:
                tstv = self.solve_d(d_mod, o_tee=True, stop_if_nopt=False)
                if tstv != 0:
                    for x in self.states:
                        x_ic = getattr(self.d1, x + "_ic")
                        for ks in x_ic.keys():
                            print(ks, value(x_ic[ks]))

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
            xvar = getattr(self.d1, x)
            for j in self.state_vars[x]:
                self.curr_rstate[(x, j)] = value(xvar[1, self.ncp_t, j])

    def update_state_predicted(self):
        """For the olnmpc"""
        for x in self.states:
            xvar = getattr(self.d2, x)
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
        self.journalizer("I", self._c_it, "load_init_state_gen", "Load State to nmpc src_kind=" + src_kind)
        ref = kwargs.pop("ref", None)
        fe = kwargs.pop("fe", self.nfe_t)
        cp = kwargs.pop("cp", self.ncp_t)
        if src_kind == "mod":
            if not ref:
                self.journalizer("W", self._c_it, "load_init_state_gen", "No model was given")
                self.journalizer("W", self._c_it, "load_init_state_gen", "No update on state performed")
                return
            for x in self.states:
                xic = getattr(dmod, x + "_ic")
                xvar = getattr(dmod, x)
                xsrc = getattr(ref, x)
                for j in self.state_vars[x]:
                    xic[j].value = value(xsrc[(fe, cp) + j])
                    xvar[(1, 0) + j].set_value(value(xsrc[(fe, cp) + j]))
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
                self.journalizer("W", self._c_it, "load_init_state_gen", "No dict w/state was specified")
                self.journalizer("W", self._c_it, "load_init_state_gen", "No update on state performed")
                return

    def make_noisy(self, cov_dict, conf_level=2):
        self.d1.name = "Noisy plant (d1)"
        k = 0
        for x in self.states:
            s = getattr(self.d1, x)  #: state
            xicc = getattr(self.d1, x + "_icc")
            xicc.deactivate()
            for j in self.state_vars[x]:
                self.xp_l.append(s[(1, 0) + j])
                self.xp_key[(x, j)] = k
                k += 1

        self.d1.xS_pnoisy = Set(initialize=[i for i in range(0, len(self.xp_l))])  #: Create set of noisy_states
        self.d1.w_pnoisy = Var(self.d1.xS_pnoisy, initialize=0.0)  #: Model disturbance
        self.d1.Q_pnoisy = Param(self.d1.xS_pnoisy, initialize=1, mutable=True)
        self.d1.obj_fun_noisy = Objective(sense=maximize,
                                          expr=0.5 *
                                              sum(self.d1.Q_pnoisy[k] * self.d1.w_pnoisy[k]**2 for k in self.d1.xS_pnoisy)
                                          )
        self.d1.ics_noisy = ConstraintList()

        k = 0
        for x in self.states:
            s = getattr(self.d1, x)  #: state
            xic = getattr(self.d1, x + "_ic")
            for j in self.state_vars[x]:
                expr = s[(1, 1) + j] == xic[j] + self.d1.w_pnoisy[k]
                self.d1.ics_noisy.add(expr)
                k += 1

        for key in cov_dict:
            vni = key
            v_i = self.xp_key[vni]
            self.d1.Q_pnoisy[v_i].value = cov_dict[vni]
            self.d1.w_pnoisy[v_i].setlb(-conf_level * cov_dict[vni])
            self.d1.w_pnoisy[v_i].setub(conf_level * cov_dict[vni])

        with open("debug.txt", "w") as f:
            self.d1.Q_pnoisy.display(ostream=f)
            self.d1.obj_fun_noisy.pprint(ostream=f)
            self.d1.ics_noisy.pprint(ostream=f)
            self.d1.w_pnoisy.display(ostream=f)

    def randomize_noize(self, cov_dict):
        conf_level = np.random.randint(1, high=4)
        print("Confidence level", conf_level)
        for key in cov_dict:
            vni = key
            v_i = self.xp_key[vni]
            self.d1.w_pnoisy[v_i].setlb(-conf_level * cov_dict[vni])
            self.d1.w_pnoisy[v_i].setub(conf_level * cov_dict[vni])

