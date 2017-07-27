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

        self.ref_state = kwargs.pop("ref_state", [])

        # We need a list of tuples that contain the bounds of u
        self.olnmpc = object()

        self.curr_state_offset = {}  #: Current offset of measurement
        self.curr_state_noise = {}  #: Current noise of the state
        for x in self.states:
            for j in self.state_vars[x]:
                self.curr_state_offset[(x, j)] = 0.0
                self.curr_state_noise[(x, j)] = 0.0


    def create_nmpc(self):
        self.olnmpc = self.d_mod(self.nfe_t, self.ncp_t, _t=self._t)
        self.olnmpc.name = "olnmpc (Open-Loop NMPC)"
        self.olnmpc.create_bounds()

        for u in self.u:
            cv = getattr(self.olnmpc, u)  #: Get the param
            c_val = [value(cv[i]) for i in cv.iterkeys()]  #: Current value
            self.olnmpc.del_component(cv)  #: Delete the param
            self.olnmpc.add_component(u, Var(self.olnmpc.fe_t, initialize=lambda m, i: c_val[i-1]))
            cc = getattr(self.olnmpc, u + "_c")  #: Get the constraint
            ce = getattr(self.olnmpc, u + "_e")  #: Get the expression
            cv = getattr(self.olnmpc, u)  #: Get the new variable
            cc.clear()
            cc.rule = lambda m, i: cv[i] == ce[i]
            cc.reconstruct()

        self.xmpc_l = []
        self.xmpc_key = {}
        k = 0
        for x in self.states:
            n_s = getattr(self.olnmpc, x)  #: Noisy-state
            for j in self.state_vars[x]:
                self.xmpc_l.append(n_s[(1, 0) + j])
                self.xmpc_key[(x, j)] = k
                k += 1

        self.olnmpc.xmpcS_nmpc = Set(initialize=[i for i in range(0, len(self.xmpc_l))])  #: Create set of noisy_states
        self.olnmpc.xmpc_ref_nmpc = Param(self.olnmpc.xmpcS_nmpc, initialize=0.0, mutable=True)
        self.olnmpc.Q_nmpc = Param(self.olnmpc.xmpcS_nmpc, initialize=1, mutable=True)  #: Control-weight

        self.umpc_l = []
        for u in self.u:
            uvar = getattr(self.olnmpc, u)
            for k in uvar.keys():
                self.umpc_l.append(uvar[k])

        self.olnmpc.umpcS_nmpc = Set(initialize=[i for i in range(0, len(self.umpc_l))])
        self.olnmpc.umpc_ref_nmpc = Param(self.olnmpc.umpcS_nmpc, initialize=0.0, mutable=True)
        self.olnmpc.R_nmpc = Param(self.olnmpc.umpcS_nmpc, initialize=1, mutable=True)  #: Control-weight


    def initialize_olnmpc(self, ref):
        """Initializes the olnmpc from a reference state
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
            self.solve_d(dum, o_tee=False)
            #: Patch
            self.load_d_d(dum, self.olnmpc, finite_elem)

            for u in self.u:
                cv = getattr(u, self.olnmpc)  #: set controls for open-loop nmpc
                cv_dum = getattr(u, dum)
                # works only for fe_t index
                cv[finite_elem].set_value(value(cv_dum[1]))
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

        # Set of states for control
        # weights for the states (objective)
        # set of controls for control
        # weights for the control (objective)

        self.olnmpc.xk = Set(initialize=[i for i in range(0, len(self.l_state))])
        self.olnmpc.Qk = Param(self.olnmpc.xk, initialize=1., mutable=True)

        self.olnmpc.s_w = Param(initialize=state_weight, mutable=True)
        self.olnmpc.c_w = Param(initialize=control_weight, mutable=True)

        self.olnmpc.nfe_tm1 = Set(initialize=[i for i in range(1, self.nfe_t)])

        self.load_qk()

        self.olnmpc.O_F = Objective(sense=minimize,
                                 expr=sum(
                                     self.olnmpc.s_w * sum(self.olnmpc.Qk[s] *
                                         self._xk_r_lsq(self.olnmpc, i, s, self.l_state, self.l_vals, self.ncp_t)
                                         for s in self.olnmpc.xk) +
                                     self.olnmpc.c_w * (self.olnmpc.per_opening2[i] - value(ref.per_opening2[1])) ** 2 +
                                     self.olnmpc.c_w * (self.olnmpc.per_opening1[i] - value(ref.per_opening1[1])) ** 2
                                     for i in self.olnmpc.nfe_tm1))


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
        """Creates the required suffixes for the olnmpc problem"""
        if hasattr(self.olnmpc, "npdp"):
            pass
        else:
            self.olnmpc.npdp = Suffix(direction=Suffix.EXPORT)
        if hasattr(self.olnmpc, "dof_v"):
            pass
        else:
            self.olnmpc.dof_v = Suffix(direction=Suffix.EXPORT)

        for u in self.u:
            uv = getattr(self.olnmpc, u)
            uv[1].set_suffix_value(self.olnmpc.dof_v, 1)


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

    def find_target_ss(self):
        """Attempt to find a second steady state
        Args:
            None
        Returns
            None"""
        print("-" * 120)
        print("I[[find_target_ss]] Attempting to find steady state")
        print("-" * 120)
        del self.ss2
        self.ss2 = self.d_mod(1, 1, steady=True)
        self.ss2.name = "ss2 (reference)"
        for u in self.u:
            cv = getattr(self.ss2, u)  #: Get the param
            c_val = [value(cv[i]) for i in cv.iterkeys()]  #: Current value
            self.ss2.del_component(cv)  #: Delete the param
            self.ss2.add_component(u, Var(self.ss2.fe_t, initialize=lambda m, i: c_val[i-1]))
            cc = getattr(self.ss2, u + "_c")  #: Get the constraint
            ce = getattr(self.ss2, u + "_e")  #: Get the expression
            cv = getattr(self.ss2, u)  #: Get the new variable
            cc.clear()
            cc.rule = lambda m, i: cv[i] == ce[i]
            cc.reconstruct()

        self.ss2.create_bounds()
        self.ss2.equalize_u(direction="r_to_u")

        for vs in self.ss.component_objects(Var, active=True):  #: Load_guess
            vt = getattr(self.ss2, vs.getname())
            for ks in vs.iterkeys():
                vt[ks].set_value(value(vs[ks]))
        ofexp = 0
        for i in self.ref_state.keys():
            v = getattr(self.ss2, i[0])
            vkey = i[1]
            ofexp += (v[(1, 1) + vkey] - self.ref_state[i])**2
        self.ss2.obfun_ss2 = Objective(expr=ofexp, sense=minimize)

        self.solve_d(self.ss2, iter_max=500, stop_if_nopt=True)

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