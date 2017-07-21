#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from nmpc_mhe.dync.DynGen import DynGen
import numpy as np
import sys
from six import iterkeys
__author__ = "David M Thierry @dthierry"

"""Not quite."""


class NmpcGen(DynGen):
    def __init__(self, **kwargs):
        DynGen.__init__(self, **kwargs)
        # We need a list of the relevant controls smth like u = [u1, u2, ..., un]
        # We need a list of tuples that contain the bounds of u


        print("-" * 120)
        print("I[[create_olnmpc]] olnmpc (full) model created.")
        print("-" * 120)

        self.olnmpc = self.d_mod(self.nfe_t, self.ncp_t, _t=self._t)
        self.olnmpc.name = "olnmpc (Open-Loop NMPC)"
        self.olnmpc.create_bounds()

        for u in self.u:
            cv = getattr(self.olnmpc, u)  #: Get the param
            c_val = [value(cv[i]) for i in cv.iterkeys()]  #: Current value
            self.olnmpc.del_component(cv)  #: Delete the param
            self.olnmpc.add_component(u + "_", Var(self.olnmpc.fe_t, initialize=lambda m, i: c_val[i-1]))
            cc = getattr(self.olnmpc, u + "_c")  #: Get the constraint
            ce = getattr(self.olnmpc, u + "_e")  #: Get the expression
            cv = getattr(self.olnmpc, u + "_")  #: Get the new variable
            cc.clear()
            cc.rule = lambda m, i: cv[i] == ce[i]
            cc.reconstruct()



    def initialize_olnmpc(self, ref):
        """Initializes the olnmpc
        Args
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model"""
        self.journalizer("I", self._c_it, "initialize_olnmpc", "Attempting to initialize olnmpc")
        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.name = "Dummy I"
        #: Load current solution
        self.load_d_d(ref, dum, 1)
        for u in self.u:  #: Initialize controls
            cv = getattr(u, dum)
            cv_ref = getattr(u, ref)
            for i in cv.iterkeys():
                cv[i].value = value(cv_ref[i])

        # dum.per_opening2[1].value = value(ref.per_opening2[1])
        # dum.per_opening1[1].value = value(ref.per_opening1[1])
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
                    p = getattr(self.olnmpc, pn)  #: Target
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
            self.load_d_d(dum, self.olnmpc, finite_elem)

            for u in self.u:
                cv = getattr(u, self.olnmpc)  #: set controls for open-loop nmpc
                cv_dum = getattr(u, dum)
                # works only for fe_t index
                cv[finite_elem].set_value(value(cv_dum[1]))


            # self.olnmpc.per_opening2[finite_elem].set_value(value(dum.per_opening2[1]))
            # self.olnmpc.per_opening1[finite_elem].set_value(value(dum.per_opening1[1]))
        self.journalizer("I", self._c_it, "initialize_olnmpc", "Attempting to initialize olnmpc Done")

    def load_reference_state(self, ref, state_weight=1e-14, control_weight=1e+01):
        """Loads ref state for set-point
        Args:
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference Model
            state_weight (float): Designated weight factor for the state term in the objective
            control_weight (float): Designated weight factor for the control term in the objective"""
        try:
            self.olnmpc.del_component(self.olnmpc.xk)
        except AttributeError:
            pass
        try:
            self.olnmpc.del_component(self.olnmpc.xk_ref)
        except AttributeError:
            pass
        try:
            self.olnmpc.del_component(self.olnmpc.O_F)
        except AttributeError:
            pass
        try:
            self.olnmpc.del_component(self.olnmpc.Qk)
        except AttributeError:
            pass
        try:
            self.olnmpc.del_component(self.olnmpc.nfe_tm1)
        except AttributeError:
            pass
        try:
            self.olnmpc.del_component(self.olnmpc.s_w)
            self.olnmpc.del_component(self.olnmpc.c_w)
        except AttributeError:
            pass
        self.olnmpc.xk = Set(initialize=[i for i in range(0, len(self.l_state))])
        self.olnmpc.Qk = Param(self.olnmpc.xk, initialize=1., mutable=True)

        self.olnmpc.s_w = Param(initialize=state_weight, mutable=True)
        self.olnmpc.c_w = Param(initialize=control_weight, mutable=True)

        self.olnmpc.nfe_tm1 = Set(initialize=[i for i in range(1, self.nfe_t)])
        # self.olnmpc.xk_res = Var(self.olnmpc.nfe_tm1, self.olnmpc.xk,
        #                       initialize=lambda m, i, x: self._xk_irule(m, i, x, self.l_state, self.l_vals, self.ncp_t))
        # self.olnmpc.xk_ref = Constraint(self.olnmpc.nfe_tm1, self.olnmpc.xk,
        #                              rule=lambda m, i, x: self._xk_r_con(m, i, x, self.l_state, self.l_vals, self.ncp_t))
        self.load_qk()
        # self.olnmpc.O_F = Objective(sense=minimize,
        #                          expr=sum(
        #                              sum(self.olnmpc.Qk[s] * self.olnmpc.xk_res[i, s]**2 for s in self.olnmpc.xk) +
        #                              (self.olnmpc.per_opening2[i] - value(ref.per_opening2[1]))**2
        #                              for i in self.olnmpc.nfe_tm1))

        self.olnmpc.O_F = Objective(sense=minimize,
                                 expr=sum(
                                     self.olnmpc.s_w * sum(self.olnmpc.Qk[s] *
                                         self._xk_r_lsq(self.olnmpc, i, s, self.l_state, self.l_vals, self.ncp_t)
                                         for s in self.olnmpc.xk) +
                                     self.olnmpc.c_w * (self.olnmpc.per_opening2[i] - value(ref.per_opening2[1])) ** 2 +
                                     self.olnmpc.c_w * (self.olnmpc.per_opening1[i] - value(ref.per_opening1[1])) ** 2
                                     for i in self.olnmpc.nfe_tm1))
        with open("xkof.txt", "w") as f:
            self.olnmpc.xk.pprint(ostream=f)
            # self.olnmpc.xk_res.pprint(ostream=f)
            # self.olnmpc.xk_ref.pprint(ostream=f)
            self.olnmpc.Qk.display(ostream=f)
            self.olnmpc.O_F.pprint(ostream=f)
            f.close()

    def load_qk(self, max_qval=1e+04, min_qval=1e-06):
        """Loads values to the weight matrix Qk with the last fe as a reference"""
        for i in self.olnmpc.xk:
            vl = self._xk_irule(self.olnmpc, self.nfe_t, i, self.l_state, self.l_vals, self.ncp_t)
            # if vl < 1e-08:
            #     vl = 1e-06
            if vl > max_qval:
                vl = max_qval
            if vl < min_qval:
                vl = min_qval
            self.olnmpc.Qk[i].value = vl**2

    def new_weights_olnmpc(self, state_weight, control_weight):
        self.olnmpc.c_w = control_weight
        self.olnmpc.s_w = state_weight


    def create_suffixes(self):
        """Creates the requiered suffixes for the olnmpc problem"""
        if hasattr(self.olnmpc, "npdp"):
            pass
        else:
            self.olnmpc.npdp = Suffix(direction=Suffix.EXPORT)
        if hasattr(self.olnmpc, "dof_v"):
            pass
        else:
            self.olnmpc.dof_v = Suffix(direction=Suffix.EXPORT)
        # for i in self.states:
        #     con_name = "ic_" + i.lower()
        #     con_ = getattr(self.olnmpc, con_name)
        #     print(self.w_)
        #     for ks in con_.iterkeys():
        #         con_[ks].set_suffix_value(self.olnmpc.npdp, self.w_[i, ks])
        #: Control 1
        for key in self.olnmpc.per_opening1.iterkeys():
            if self.olnmpc.per_opening1[key].stale:
                continue
            self.olnmpc.per_opening1[key].set_suffix_value(self.olnmpc.dof_v, 1)
        #: Control 2
        for key in self.olnmpc.per_opening2.iterkeys():
            if self.olnmpc.per_opening2[key].stale:
                continue
            self.olnmpc.per_opening2[key].set_suffix_value(self.olnmpc.dof_v, 1)


    def solve_dot_dri(self):
        for i in self.states:
            con_name = "ic_" + i.lower()
            con_ = getattr(self.olnmpc, con_name)
            for ks in con_.iterkeys():
                con_[ks].set_suffix_value(self.olnmpc.npdp, self.w_[i, ks])
        self.journalizer("I", self._c_it, "solve_dot_driver", self.olnmpc.name)
        with open("r1.txt", "w") as r1:
            self.olnmpc.per_opening1.pprint(ostream=r1)
            self.olnmpc.per_opening2.pprint(ostream=r1)
            r1.close()
        results = self.dot_driver.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        with open("r2.txt", "w") as r2:
            self.olnmpc.per_opening1.pprint(ostream=r2)
            self.olnmpc.per_opening2.pprint(ostream=r2)
            r2.close()
        ftiming = open("timings_dot_driver.txt", "r")
        s = ftiming.readline()
        ftiming.close()
        k = s.split()
        self._dot_timing = k[0]

    def solve_k_aug(self):
        self.olnmpc.ipopt_zL_in.update(self.olnmpc.ipopt_zL_out)
        self.olnmpc.ipopt_zU_in.update(self.olnmpc.ipopt_zU_out)
        self.journalizer("I", self._c_it, "solve_k_aug", self.olnmpc.name)
        results = self.k_aug.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        ftimings = open("timings_k_aug.txt", "r")
        s = ftimings.readline()
        ftimings.close()
        self._k_timing = s.split()


    def plant_input_olnmpc(self, nsteps=5):
        self.journalizer("I", self._c_it, "plant_input", "Continuation")
        olnmpc = self.olnmpc
        d1 = self.d1
        #: Inputs
        target = [value(olnmpc.per_opening2[1]), value(olnmpc.per_opening1[1])]
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
            sw = self.olnmpc.s_w
            cw = self.olnmpc.c_w
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

        retval = self.solve_d(self.olnmpc, max_cpu_time=300,
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