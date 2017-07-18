#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from nmpc_mhe.dync.DynGen import DynGen
import numpy as np
import sys

__author__ = "David M Thierry @dthierry"

"""Not quite."""


class NmpcGen(DynGen):
    def __init__(self):
        DynGen.__init__(self)

    def create_ocp(self):
        print("-" * 120)
        print("I[[create_ocp]] OcP (full) model created.")
        print("-" * 120)
        self.ocp = self.d_mod(self.nfe_t, self.ncp_t, _t=self._t)
        self.ocp.name = "ocp (Optimal Control)"
        self.ocp.create_bounds()
        # self.ctrl = ["per_opening2"] Don't know how to do this properly
        self.ocp.del_component(self.ocp.per_opening2)
        self.ocp.per_opening2 = Var(self.ocp.fe_t, initialize=50., bounds=(5, 95))
        self.ocp.v4.reconstruct()

        self.ocp.del_component(self.ocp.per_opening1)
        self.ocp.per_opening1 = Var(self.ocp.fe_t, initialize=85., bounds=(5, 95))
        self.ocp.v1.reconstruct()

    def initialize_ocp(self, ref):
        """Initializes the ocp
        Args
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model"""
        self.journalizer("I", self._c_it, "initialize_ocp", "Attempting to initialize OCP")
        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.name = "Dummy I"
        #: Load current solution
        self.load_d_d(ref, dum, 1)
        dum.per_opening2[1].value = value(ref.per_opening2[1])
        dum.per_opening1[1].value = value(ref.per_opening1[1])
        #: Patching of finite elements
        for finite_elem in range(1, self.nfe_t + 1):
            #: Cycle ICS
            for i in self.states:
                pn = i + "_ic"
                p = getattr(dum, pn)
                vs = getattr(dum, i)
                for ks in p.iterkeys():
                    p[ks].value = value(vs[(1, self.ncp_t) + ks])
            if finite_elem == 1:
                for i in self.states:
                    pn = i + "_ic"
                    p = getattr(self.ocp, pn)  #: Target
                    vs = getattr(dum, i)  #: Source
                    for ks in p.iterkeys():
                        p[ks].value = value(vs[(1, self.ncp_t) + ks])
                    # with open("ic.txt", "a") as f:
                    #     p.display(ostream=f)
                    #     f.close()

            #: Solve
            # ref.pprint(filename="ref.txt")
            # dum.pprint(filename="dum.txt")
            self.solve_d(dum, o_tee=False)
            #: Patch
            self.load_d_d(dum, self.ocp, finite_elem)
            self.ocp.per_opening2[finite_elem].set_value(value(dum.per_opening2[1]))
            self.ocp.per_opening1[finite_elem].set_value(value(dum.per_opening1[1]))
        self.journalizer("I", self._c_it, "initialize_ocp", "Attempting to initialize OCP Done")

    def load_reference_state(self, ref, state_weight=1e-14, control_weight=1e+01):
        """Loads ref state for set-point
        Args:
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference Model
            state_weight (float): Designated weight factor for the state term in the objective
            control_weight (float): Designated weight factor for the control term in the objective"""
        try:
            self.ocp.del_component(self.ocp.xk)
        except AttributeError:
            pass
        try:
            self.ocp.del_component(self.ocp.xk_ref)
        except AttributeError:
            pass
        try:
            self.ocp.del_component(self.ocp.O_F)
        except AttributeError:
            pass
        try:
            self.ocp.del_component(self.ocp.Qk)
        except AttributeError:
            pass
        try:
            self.ocp.del_component(self.ocp.nfe_tm1)
        except AttributeError:
            pass
        try:
            self.ocp.del_component(self.ocp.s_w)
            self.ocp.del_component(self.ocp.c_w)
        except AttributeError:
            pass
        self.ocp.xk = Set(initialize=[i for i in range(0, len(self.l_state))])
        self.ocp.Qk = Param(self.ocp.xk, initialize=1., mutable=True)

        self.ocp.s_w = Param(initialize=state_weight, mutable=True)
        self.ocp.c_w = Param(initialize=control_weight, mutable=True)

        self.ocp.nfe_tm1 = Set(initialize=[i for i in range(1, self.nfe_t)])
        # self.ocp.xk_res = Var(self.ocp.nfe_tm1, self.ocp.xk,
        #                       initialize=lambda m, i, x: self._xk_irule(m, i, x, self.l_state, self.l_vals, self.ncp_t))
        # self.ocp.xk_ref = Constraint(self.ocp.nfe_tm1, self.ocp.xk,
        #                              rule=lambda m, i, x: self._xk_r_con(m, i, x, self.l_state, self.l_vals, self.ncp_t))
        self.load_qk()
        # self.ocp.O_F = Objective(sense=minimize,
        #                          expr=sum(
        #                              sum(self.ocp.Qk[s] * self.ocp.xk_res[i, s]**2 for s in self.ocp.xk) +
        #                              (self.ocp.per_opening2[i] - value(ref.per_opening2[1]))**2
        #                              for i in self.ocp.nfe_tm1))

        self.ocp.O_F = Objective(sense=minimize,
                                 expr=sum(
                                     self.ocp.s_w * sum(self.ocp.Qk[s] *
                                         self._xk_r_lsq(self.ocp, i, s, self.l_state, self.l_vals, self.ncp_t)
                                         for s in self.ocp.xk) +
                                     self.ocp.c_w * (self.ocp.per_opening2[i] - value(ref.per_opening2[1])) ** 2 +
                                     self.ocp.c_w * (self.ocp.per_opening1[i] - value(ref.per_opening1[1])) ** 2
                                     for i in self.ocp.nfe_tm1))
        with open("xkof.txt", "w") as f:
            self.ocp.xk.pprint(ostream=f)
            # self.ocp.xk_res.pprint(ostream=f)
            # self.ocp.xk_ref.pprint(ostream=f)
            self.ocp.Qk.display(ostream=f)
            self.ocp.O_F.pprint(ostream=f)
            f.close()

    def load_qk(self, max_qval=1e+04, min_qval=1e-06):
        """Loads values to the weight matrix Qk with the last fe as a reference"""
        for i in self.ocp.xk:
            vl = self._xk_irule(self.ocp, self.nfe_t, i, self.l_state, self.l_vals, self.ncp_t)
            # if vl < 1e-08:
            #     vl = 1e-06
            if vl > max_qval:
                vl = max_qval
            if vl < min_qval:
                vl = min_qval
            self.ocp.Qk[i].value = vl**2

    def new_weights_ocp(self, state_weight, control_weight):
        self.ocp.c_w = control_weight
        self.ocp.s_w = state_weight


    def create_suffixes(self):
        """Creates the requiered suffixes for the OCP problem"""
        if hasattr(self.ocp, "npdp"):
            pass
        else:
            self.ocp.npdp = Suffix(direction=Suffix.EXPORT)
        if hasattr(self.ocp, "dof_v"):
            pass
        else:
            self.ocp.dof_v = Suffix(direction=Suffix.EXPORT)
        # for i in self.states:
        #     con_name = "ic_" + i.lower()
        #     con_ = getattr(self.ocp, con_name)
        #     print(self.w_)
        #     for ks in con_.iterkeys():
        #         con_[ks].set_suffix_value(self.ocp.npdp, self.w_[i, ks])
        #: Control 1
        for key in self.ocp.per_opening1.iterkeys():
            if self.ocp.per_opening1[key].stale:
                continue
            self.ocp.per_opening1[key].set_suffix_value(self.ocp.dof_v, 1)
        #: Control 2
        for key in self.ocp.per_opening2.iterkeys():
            if self.ocp.per_opening2[key].stale:
                continue
            self.ocp.per_opening2[key].set_suffix_value(self.ocp.dof_v, 1)


    def solve_dot_dri(self):
        for i in self.states:
            con_name = "ic_" + i.lower()
            con_ = getattr(self.ocp, con_name)
            for ks in con_.iterkeys():
                con_[ks].set_suffix_value(self.ocp.npdp, self.w_[i, ks])
        self.journalizer("I", self._c_it, "solve_dot_driver", self.ocp.name)
        with open("r1.txt", "w") as r1:
            self.ocp.per_opening1.pprint(ostream=r1)
            self.ocp.per_opening2.pprint(ostream=r1)
            r1.close()
        results = self.dot_driver.solve(self.ocp, tee=True, symbolic_solver_labels=True)
        self.ocp.solutions.load_from(results)
        with open("r2.txt", "w") as r2:
            self.ocp.per_opening1.pprint(ostream=r2)
            self.ocp.per_opening2.pprint(ostream=r2)
            r2.close()
        ftiming = open("timings_dot_driver.txt", "r")
        s = ftiming.readline()
        ftiming.close()
        k = s.split()
        self._dot_timing = k[0]

    def solve_k_aug(self):
        self.ocp.ipopt_zL_in.update(self.ocp.ipopt_zL_out)
        self.ocp.ipopt_zU_in.update(self.ocp.ipopt_zU_out)
        self.journalizer("I", self._c_it, "solve_k_aug", self.ocp.name)
        results = self.k_aug.solve(self.ocp, tee=True, symbolic_solver_labels=True)
        self.ocp.solutions.load_from(results)
        ftimings = open("timings_k_aug.txt", "r")
        s = ftimings.readline()
        ftimings.close()
        self._k_timing = s.split()


    def plant_input_ocp(self, nsteps=5):
        self.journalizer("I", self._c_it, "plant_input", "Continuation")
        ocp = self.ocp
        d1 = self.d1
        #: Inputs
        target = [value(ocp.per_opening2[1]), value(ocp.per_opening1[1])]
        current = [value(d1.per_opening2[1]), value(d1.per_opening1[1])]
        ncont_steps = nsteps
        self.journalizer("I", self._c_it,
                         "plant_input","Target {:f}, Current {:f}, n_steps {:d}".format(target[0], current[0], ncont_steps))
        print("Target {:f}, Current {:f}, n_steps {:d}".format(target[0], current[0], ncont_steps))
        for i in range(0, ncont_steps):
            for pi in range(0, len(current)):
                current[pi] += (target[pi]-current[pi])/ncont_steps
                print("Continuation :Current {:d}\t{:f}".format(pi, current[pi]))
            d1.per_opening2[1].value = current[0]
            d1.per_opening1[1].value = current[1]
            self.solve_d(d1, o_tee=False)



    def stall_strategy(self, strategy, cmv=1e-04, **kwargs):
        """Suggested three strategies: Change weights, change matrices, change linear algebra"""
        self._stall_iter += 1
        self.journalizer("I", self._c_it, "stall_strategy", "Solver Stalled. " + str(self._stall_iter) + " Times")
        if strategy == "increase_weights":
            spf = 0
            ma57_as = "no"
            sw = self.ocp.s_w
            cw = self.ocp.c_w
            sw.value += sw.value
            cw.value += cw.value
            if sw.value > 1e06 or cw.value > 1e06:
                return 1
        elif strategy == "recompute_matrices":
            cmv += 1e04 * 5
            self.load_qk(max_qval=cmv)
        elif strategy == "linear_algebra":
            spf = 1
            ma57_as = "yes"

        retval = self.solve_d(self.ocp, max_cpu_time=300,
                              small_pivot_flag=spf,
                              ma57_automatic_scaling=ma57_as,
                              want_stime=True,
                              rep_timing=True)
        if retval == 0:
            return 0
        else:
            if self._stall_iter > 10:
                self.journalizer("I", self._c_it, "stall_strategy",
                                 "Max number of tries reached")
                sys.exit()
            self.stall_strategy("increase_weights")

    def find_target_ss(self, target):
        """Attempt to find a second steady state
        Args:
            target (float): The desired carbon capture
        Returns
            None"""
        print("-" * 120)
        print("I[[find_target_ss]] Attempting to find steady state")
        print("-" * 120)
        del self.ss2
        self.ss2 = self.d_mod(1, 1, steady=True)
        self.ss2.name = "ss2 (reference)"

        self.ss2.del_component(self.ss2.per_opening1)
        self.ss2.del_component(self.ss2.per_opening2)

        self.ss2.per_opening1 = Var(self.ss2.fe_t, initialize=85., bounds=(0, 95))
        self.ss2.per_opening2 = Var(self.ss2.fe_t, initialize=50., bounds=(0, 95))
        # self.ss2.per_opening3 = Var(self.ss2.fe_t, initialize=50., bounds=(0, 100))
        self.ss2.v1.reconstruct()
        # self.ss2.v3.reconstruct()
        self.ss2.v4.reconstruct()
        self.ss2.create_bounds()

        for vt in self.ss2.component_objects(Var, active=True):
            vs = getattr(self.ss, vt.getname())
            for ks in vs.iterkeys():
                vt[ks].set_value(value(vs[ks]))
        self.ss2.create_bounds()
        self.ss2.ob = Objective(expr=1e+02 * (target - self.ss2.c_capture[1, 1]) ** 2, sense=minimize)
        self.solve_d(self.ss2, iter_max=500)

        print("-" * 120)
        print("I[[find_target_ss]] Target: solve done")
        print("-" * 120)

        self.l_state = []
        self.l_vals = []
        ref = self.ss2
        for i in self.states:
            vs = getattr(ref, i)
            for key in vs.iterkeys():
                if vs[key].stale:
                    continue
                self.l_state.append((i, key[2:]))
                self.l_vals.append(value(vs[key]))