#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition

import numpy as np
import sys

__author__ = "David M Thierry @dthierry"


class DynGen(object):
    def __init__(self, **kwargs):
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
        self.k_aug = SolverFactory("k_aug",
                                   executable="/home/dav0/k2/KKT_matrix/src/kmatrix/k_aug")
        self.dot_driver = SolverFactory("dot_driver",
                                        executable="/home/dav0/k2/KKT_matrix/src/kmatrix/dot_driver/dot_driver")

        # self.k_aug.options["eig_rh"] = ""
        # self.ipopt.options["halt_on_ampl_error"] = "yes"

        self.ipopt.options["print_user_options"] = "yes"
        # self.k_aug.options["deb_kkt"] = ""

        self.ss.ofun = Objective(expr=1, sense=minimize)
        self.dyn = object()
        self.l_state = []
        self.l_vals = []
        self.w_ = {}
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
            f.write("max_iter 10\n")
            f.write("mu_init 1e-08\n")
            f.close()
        results = self.ipopt.solve(self.ss, tee=True, symbolic_solver_labels=True, report_timing=True)
        self.ss.solutions.load_from(results)
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
                    print(vs.getname())
                    for ks in vs.iterkeys():
                        kj = ks[2:]
                        # for i in range(1, self.nfe_t + 1):
                        for j in range(1, self.ncp_t + 1):
                            vd[(1, j) + kj].set_value(value(vs[ks]))
                else:
                    for ks in vs.iterkeys():
                        for kd in vd.keys():
                            vd[kd].set_value(value(vs[ks]))
        for x in self.states:
            dv = getattr(dmod, "d" + x + "_dt")
            for i in dv.itervalues():
                i.set_value(0.0)
        for i in self.states:
            pn = i + "_ic"
            p = getattr(d, pn)
            vs = getattr(s, i)
            vd0 = getattr(d, i)
            for ks in p.iterkeys():
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
        if kwargs.get("max_cpu_time"):
            max_cpu_time = kwargs["max_cpu_time"]
        if kwargs.get("linear_solver"):
            linear_solver = kwargs["linear_solver"]
        if kwargs.get("ma57_pre_alloc"):
            ma57_pre_alloc = kwargs["ma57_pre_alloc"]
        if kwargs.get("ma57_automatic_scaling"):
            ma57_automatic_scaling = kwargs["ma57_automatic_scaling"]
        if kwargs.get("ma57_small_pivot_flag"):
            ma57_small_pivot_flag = kwargs["ma57_small_pivot_flag"]

        o_tee = kwargs.pop("o_tee", True)
        skip_mult_update = kwargs.pop("skip_update", True)

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
            # f.write("mu_init 1e-08\n")
            # f.write("halt_on_ampl_error yes")
            f.close()
        results = self.ipopt.solve(d, tee=o_tee, symbolic_solver_labels=True, report_timing=rep_timing)
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

    def cycle_ics(self):
        """Patches the initial conditions with the last result from the simulation
        Args:
            None
        Return
            None"""
        print("-" * 120)
        print("I[[cycle_ics]] Cycling initial state.")
        print("-" * 120)
        for i in self.states:
            pn = i + "_ic"
            p = getattr(self.d1, pn)
            vs = getattr(self.d1, i)
            for ks in p.iterkeys():
                p[ks].value = value(vs[(1, self.ncp_t) + ks])

    def load_d_d(self, src, tgt, i):
        """Loads the solution of the src state model into the tgt
        Args:
            src (pyomo.core.base.PyomoModel.ConcreteModel):
            trg (pyomo.core.base.PyomoModel.ConcreteModel):
            i (int): The target's finite element
        Return:
            None"""
        for vs in src.component_objects(Var, active=True):
            vd = getattr(tgt, vs.getname())
            # there are two cases: 1 key 1 elem, several keys 1 element
            vskeys = vs.keys()
            if len(vskeys) == 1:
                #: One key
                for ks in vskeys:
                    for v in vd.itervalues():
                        v.set_value(value(vs[ks]))
            else:
                k = 0
                for ks in vskeys:
                    if k == 0:
                        if type(ks) != tuple:
                            #: Several keys of 1 element each!!
                            for ks1 in vskeys:
                                vd[ks1].set_value(value(vs[ks1]))  #: This has got to be true
                            break
                        k += 1
                    kj = ks[2:]
                    if vs.getname() in self.states:  #: States start at 0
                        for j in range(0, self.ncp_t + 1):
                            vd[(i, j) + kj].set_value(value(vs[ks]))
                    else:
                        for j in range(1, self.ncp_t + 1):
                            vd[(i, j) + kj].set_value(value(vs[ks]))

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

    @staticmethod
    def _xk_r_con(m, i, s, sl, vl, cp):
        """Rule for the residual constraint
        Args:
            m (pyomo.core.base.PyomoModel.ConcreteModel): The parent model
            i (int): The current finite element
            s (int): The current state number
            sl (list): The list of state names tuple
            vl (list): The list of values of state
            cp (int): The desired collocation point"""
        v = getattr(m, sl[s][0])  #: State
        return v[(i, cp) + sl[s][1]] == vl[s] + m.xk_res[i, s]

    @staticmethod
    def _xk_r_lsq(m, i, s, sl, vl, cp):
        """Rule for the residual constraint
        Args:
            m (pyomo.core.base.PyomoModel.ConcreteModel): The parent model
            i (int): The current finite element
            s (int): The current state number
            sl (list): The list of state names tuple
            vl (list): The list of values of state
            cp (int): The desired collocation point"""
        v = getattr(m, sl[s][0])  #: State
        return (v[(i, cp) + sl[s][1]] - vl[s])**2

    @staticmethod
    def _xk_irule(m, i, s, sl, vl, cp):
        v = getattr(m, sl[s][0])
        return (abs(value(v[(i, cp) + sl[s][1]]) - vl[s])**0.5)/vl[s]

    def print_cc(self):
        """print a state/measurement of interest"""

        self.journalizer("I",
                         self._c_it,
                         "print_cc",
                         "i = {:d} Current cco2 {:f}".format(self._c_it, value(self.d1.c_capture[1, self.ncp_t])))

        self.ccl.append(value(self.d1.c_capture[1, self.ncp_t]))
        self.sp.append(value(self.ss2.c_capture[1, 1]))
        self.iput.append([value(self.d1.per_opening2[1]), value(self.d1.per_opening1[1])])
        self._ipt_list.append(self.ip_time)
        self._dt_list.append(self._dot_timing)
        self._kt_list.append(self._k_timing)
        with open("results_dot_driver.txt", "w") as f:
            for i in range(0, len(self.ccl)):
                c = []
                o = str(self.ccl[i])
                f.write(o)
                for j in range(0, len(self.iput[i])):
                    c.append(str(self.iput[i][j]))
                    f.write("\t" + c[j])
                f.write("\t" + str(self.sp[i]))
                f.write("\t" + str(self._ipt_list[i]))
                for j in range(0, len(self._kt_list[i])):
                    f.write("\t" + self._kt_list[i][j])
                f.write("\t" + self._dt_list[i])
                f.write("\n")
            f.close()

    # NMPC or just dyn?
    def cycle_ics_noisy(self, sigma_bar=0.01):
        """Patches the initial conditions with the last result from the simulation with noise.
        Args:
            sigma_bar (float): The variance.
        Return
            None"""
        self.w_ = {}
        print("-" * 120)
        print("I[[cycle_ics]] Cycling initial state -- NOISY.")
        print("-" * 120)
        with open("noisy_" + str(self._c_it) + ".state", "w") as noi:
            with open("nominal_" + str(self._c_it) + ".state", "w") as nom:
                for i in self.states:
                    pn = i + "_ic"
                    p = getattr(self.d1, pn)
                    vs = getattr(self.d1, i)
                    for ks in p.iterkeys():
                        if vs[(1, self.ncp_t) + ks].stale:
                            continue
                        p[ks].value = value(vs[(1, self.ncp_t) + ks])
                    p.display(ostream=nom)
                    for ks in p.iterkeys():
                        if vs[(1, self.ncp_t) + ks].stale:
                            continue
                        sigma = value(vs[(1, self.ncp_t) + ks]) * sigma_bar
                        s = np.random.normal(0, sigma)
                        self.w_[i, ks] = s
                        p[ks].value = value(vs[(1, self.ncp_t) + ks]) + s
                    p.display(ostream=noi)
                nom.close()
            noi.close()
        self._c_it += 1

    def create_predictor(self):
        self.d2 = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        self.d2.name = "Dynamic Predictor_2"

    def predictor_step(self, ref):
        """Step for the nominal model.
        Args:
            ref (pyomo.core.base.PyomoModel.ConcreteModel): """
        self.journalizer("I", self._c_it, "predictor_step", "Predictor step")
        self.load_d_d(ref, self.d2, 1)
        self.d2.per_opening2[1].value = value(ref.per_opening2[1])
        self.d2.per_opening1[1].value = value(ref.per_opening1[1])
        for i in self.states:
            pn = i + "_ic"
            p = getattr(self.d2, pn)
            vs = getattr(self.d2, i)
            for ks in p.iterkeys():
                p[ks].value = value(vs[(1, self.ncp_t) + ks])
        self.solve_d(self.d2)
        self.journalizer("I", self._c_it, "predictor_step", "Predictor step - Success")

    def plant_input_gen(self, src, src_fe, nsteps=5):
        """Attempt to solve the dynamic model with some source model input
        Args:
            src (pyomo.core.base.PyomoModel.ConcreteModel): Source model
            src_fe (int): Finite element from the source model
            nsteps (int): The number of continuation steps (default=5)"""

        self.journalizer("I", self._c_it, "plant_input", "Continuation_plant")
        d1 = self.d1
        #: Inputs
        target = {}
        current = {}
        ncont_steps = nsteps

        for u in self.u:
            src_var = getattr(src, u)
            tgt_var = getattr(d1, u)

            target[u] = value(src_var[src_fe])
            current[u] = value(tgt_var[1])
            self.journalizer("I", self._c_it,
                             "plant_input",
                             "Target {:f}, Current {:f}, n_steps {:d}".format(target[u], current[u], ncont_steps))

        # print("Target {:f}, Current {:f}, n_steps {:d}".format(target[0], current[0], ncont_steps))
        for i in range(0, ncont_steps):
            for u in self.u:
                tgt_var = getattr(d1, u)
                tgt_var[1].value += (target[u]-current[u])/ncont_steps
                print("Continuation :Current {:s}\t{:f}".format(u, value(tgt_var[1])))
            if i == ncont_steps:
                self.solve_d(d1, o_tee=False, stop_if_nopt=True)
            else:
                self.solve_d(d1, o_tee=False, stop_if_nopt=False)
