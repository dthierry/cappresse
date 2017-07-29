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

        self.ref_state = kwargs.pop("ref_state", None)
        self.u_bounds = kwargs.pop("u_bounds", None)

        # We need a list of tuples that contain the bounds of u
        self.olnmpc = object()

        self.curr_soi = {}  #: Values that we would like to keep track
        for k in self.ref_state.keys():
            self.curr_soi[k] = 0.0

    def create_nmpc(self):
        self.olnmpc = self.d_mod(self.nfe_t, self.ncp_t, _t=self._t)
        self.olnmpc.name = "olnmpc (Open-Loop NMPC)"
        self.olnmpc.create_bounds()

        for u in self.u:
            cv = getattr(self.olnmpc, u)  #: Get the param
            c_val = [value(cv[i]) for i in cv.keys()]  #: Current value
            self.olnmpc.del_component(cv)  #: Delete the param
            self.olnmpc.add_component(u, Var(self.olnmpc.fe_t, initialize=lambda m, i: c_val[i-1]))
            self.olnmpc.equalize_u(direction="r_to_u")
            cc = getattr(self.olnmpc, u + "_c")  #: Get the constraint
            ce = getattr(self.olnmpc, u + "_e")  #: Get the expression
            cv = getattr(self.olnmpc, u)  #: Get the new variable
            for k in cv.keys():
                cv[k].setlb(self.u_bounds[u][0])
                cv[k].setub(self.u_bounds[u][1])
            cc.clear()
            cc.rule = lambda m, i: cv[i] == ce[i]
            cc.reconstruct()

        self.xmpc_l = {}

        self.xmpc_key = {}

        self.xmpc_l[1] = []

        k = 0
        for x in self.states:
            n_s = getattr(self.olnmpc, x)  #: State
            for j in self.state_vars[x]:
                self.xmpc_l[1].append(n_s[(1, self.ncp_t) + j])
                self.xmpc_key[(x, j)] = k
                k += 1

        for t in range(2, self.nfe_t + 1):
            self.xmpc_l[t] = []
            for x in self.states:
                n_s = getattr(self.olnmpc, x)  #: State
                for j in self.state_vars[x]:
                    self.xmpc_l[t].append(n_s[(t, self.ncp_t) + j])

        self.olnmpc.xmpcS_nmpc = Set(initialize=[i for i in range(0, len(self.xmpc_l[1]))])
        #: Create set of noisy_states
        self.olnmpc.xmpc_ref_nmpc = Param(self.olnmpc.xmpcS_nmpc, initialize=0.0, mutable=True)
        self.olnmpc.Q_nmpc = Param(self.olnmpc.xmpcS_nmpc, initialize=1, mutable=True)  #: Control-weight
        # (diagonal Matrix)
        self.olnmpc.Q_w_nmpc = Param(self.olnmpc.fe_t, initialize=1e-4, mutable=True)
        self.olnmpc.R_w_nmpc = Param(self.olnmpc.fe_t, initialize=1e2, mutable=True)

        self.olnmpc.xQ_expr_nmpc = Expression(expr=sum(
            sum(self.olnmpc.Q_w_nmpc[fe] *
                self.olnmpc.Q_nmpc[k] * (self.xmpc_l[fe][k] - self.olnmpc.xmpc_ref_nmpc[k])**2 for k in self.olnmpc.xmpcS_nmpc)
                for fe in range(1, self.nfe_t+1)))

        self.umpc_l = {}
        for t in range(1, self.nfe_t + 1):
            self.umpc_l[t] = []
            for u in self.u:
                uvar = getattr(self.olnmpc, u)
                self.umpc_l[t].append(uvar[t])

        self.olnmpc.umpcS_nmpc = Set(initialize=[i for i in range(0, len(self.umpc_l[1]))])
        self.olnmpc.umpc_ref_nmpc = Param(self.olnmpc.umpcS_nmpc, initialize=0.0, mutable=True)
        self.olnmpc.R_nmpc = Param(self.olnmpc.umpcS_nmpc, initialize=1, mutable=True)  #: Control-weight
        self.olnmpc.xR_expr_nmpc = Expression(expr=sum(
            sum(self.olnmpc.R_w_nmpc[fe] *
                self.olnmpc.R_nmpc[k] * (self.umpc_l[fe][k] - self.olnmpc.umpc_ref_nmpc[k]) ** 2 for k in
                self.olnmpc.umpcS_nmpc)
            for fe in range(1, self.nfe_t + 1)))
        self.olnmpc.objfun_nmpc = Objective(expr=self.olnmpc.xQ_expr_nmpc + self.olnmpc.xR_expr_nmpc)

    def initialize_olnmpc(self, ref, fe=1, predicted=False):
        # The reference is always a model
        # The source of the state might be different
        # The source might be a predicted-state from forward simulation
        """Initializes the olnmpc from a reference state, loads the state into the olnmpc
        Args
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model
        Returns:
            """
        self.journalizer("I", self._c_it, "initialize_olnmpc", "Attempting to initialize olnmpc")
        self.load_init_state_nmpc(src_kind="mod", ref=ref, fe=1, cp=self.ncp_t)
        if predicted:
            self.load_init_state_nmpc(src_kind="dict", state_dict="predicted")

        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.name = "Dummy I"
        #: Load current solution
        self.load_d_d(ref, dum, fe, fe_src="s")  #: This is supossed to work
        for u in self.u:  #: Initialize controls dummy model
            cv_dum = getattr(dum, u)
            cv_ref = getattr(ref, u)
            for i in cv_dum.keys():
                cv_dum[i].value = value(cv_ref[fe])
        #: Patching of finite elements
        for finite_elem in range(1, self.nfe_t + 1):
            if predicted and finite_elem==1:
                self.load_init_state_gen(dum, src_kind="dict", state_dict="predicted")
            elif finite_elem == 1:
                self.load_init_state_gen(dum, src_kind="mod", ref=ref, fe=1)
            else:
                self.load_init_state_gen(dum, src_kind="mod", ref=dum, fe=1)
            # for i in self.states:  #: Cycle ICs for dum
            #     xdum_ic = getattr(dum, i + "_ic")
            #     xdum = getattr(dum, i)
            #     for ks in xdum_ic.keys():
            #         xdum_ic[ks].value = value(xdum[(1, self.ncp_t) + (ks,)])
            #         xdum[(1, 0) + (ks,)].set_value(value(xdum[(1, self.ncp_t) + (ks,)]))
            self.solve_d(dum, o_tee=False)
            #: Patch
            self.load_d_d(dum, self.olnmpc, finite_elem)

            for u in self.u:
                cv_nmpc = getattr(self.olnmpc, u)  #: set controls for open-loop nmpc
                cv_dum = getattr(dum, u)
                # works only for fe_t index
                cv_nmpc[finite_elem].set_value(value(cv_dum[1]))
        self.journalizer("I", self._c_it, "initialize_olnmpc", "Done")

    def load_init_state_nmpc(self, **kwargs):
        """Loads ref state for set-point
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
        src_kind = kwargs.pop("src_kind", "mod")
        self.journalizer("I", self._c_it, "load_init_state_nmpc", "Load State to nmpc src_kind=" + src_kind)
        ref = kwargs.pop("ref", None)
        fe = kwargs.pop("fe", self.nfe_t)
        cp = kwargs.pop("cp", self.ncp_t)
        if src_kind == "mod":
            if not ref:
                self.journalizer("W", self._c_it, "load_init_state_nmpc", "No model was given")
                self.journalizer("W", self._c_it, "load_init_state_nmpc", "No update on state performed")
                return
            for x in self.states:
                xic = getattr(self.olnmpc, x + "_ic")
                xvar = getattr(self.olnmpc, x)
                xsrc = getattr(ref, x)
                for j in self.state_vars[x]:
                    xic[j].value = value(xsrc[(fe, cp) + j])
                    xvar[(1, 0) + j].set_value(value(xsrc[(fe, cp) + j]))
        else:
            state_dict = kwargs.pop("state_dict", None)
            if state_dict == "real":  #: Load from the real state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_rstate[(x, j)]
                        xvar[(1, 0) + j].set_value(self.curr_rstate[(x, j)])
            elif state_dict == "estimated":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_estate[(x, j)]
                        xvar[(1, 0) + j].set_value(self.curr_estate[(x, j)])
            elif state_dict == "predicted":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_pstate[(x, j)]
                        xvar[(1, 0) + j].set_value(self.curr_pstate[(x, j)])
            else:
                self.journalizer("W", self._c_it, "load_init_state_nmpc", "No dict w/state was specified")
                self.journalizer("W", self._c_it, "load_init_state_nmpc", "No update on state performed")
                return

    def compute_QR_nmpc(self, src="mhe", n=-1, **kwargs):
        """Using the current state & control targets, computes the Qk and Rk matrices (diagonal)
        Args:
            src (str): The source of the update (default mhe) (mhe or plant)
            n (int): The exponent of the weight"""
        check_values = kwargs.pop("check_values", False)
        if check_values:
            max_w_value = kwargs.pop("max_w_value", 1e+06)
            min_w_value = kwargs.pop("min_w_value", 0.0)
        self.update_targets_nmpc()
        if src == "mhe":
            for x in self.states:
                for j in self.state_vars[x]:
                    k = self.xmpc_key[(x, j)]
                    self.olnmpc.Q_nmpc[k].value = abs(self.curr_estate[(x, j)] - self.curr_state_target[(x, j)])**n
                    self.olnmpc.xmpc_ref_nmpc[k].value = self.curr_state_target[(x, j)]
        elif src == "plant":
            for x in self.states:
                for j in self.state_vars[x]:
                    k = self.xmpc_key[(x, j)]
                    self.olnmpc.Q_nmpc[k].value = abs(self.curr_rstate[(x, j)] - self.curr_state_target[(x, j)])**n
                    self.olnmpc.xmpc_ref_nmpc[k].value = self.curr_state_target[(x, j)]
        k = 0
        for u in self.u:
            self.olnmpc.R_nmpc[k].value = abs(self.curr_u[u] - self.curr_u_target[u])**n
            self.olnmpc.umpc_ref_nmpc[k].value = self.curr_u_target[u]
            k += 1
        if check_values:
            for k in self.olnmpc.xmpcS_nmpc:
                if value(self.olnmpc.Q_nmpc[k]) < min_w_value:
                    self.olnmpc.Q_nmpc[k].value = min_w_value
                if value(self.olnmpc.Q_nmpc[k]) > max_w_value:
                    self.olnmpc.Q_nmpc[k].value = max_w_value
            k = 0
            for u in self.u:
                if value(self.olnmpc.R_nmpc[k]) < min_w_value:
                    self.olnmpc.R_nmpc[k].value = min_w_value
                if value(self.olnmpc.R_nmpc[k]) > max_w_value:
                    self.olnmpc.R_nmpc[k].value = max_w_value
                k += 1

    def new_weights_olnmpc(self, state_weight, control_weight):
        if type(state_weight) == float:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[fe].value = state_weight
        elif type(state_weight) == dict:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[fe].value = state_weight[fe]

        if type(control_weight) == float:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.R_w_nmpc[fe].value = control_weight
        elif type(control_weight) == dict:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.R_w_nmpc[fe].value = control_weight[fe]

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
        for x in self.states:
            con_name = x + "_icc"
            con_ = getattr(self.olnmpc, con_name)
            for j in self.state_vars[x]:
                con_[j].set_suffix_value(self.olnmpc.npdp, self.curr_state_offset[(x, j)])
            # for ks in con_.keys():

        self.journalizer("I", self._c_it, "solve_dot_driver", self.olnmpc.name)
        # with open("r1.txt", "w") as r1:
        #     self.olnmpc.per_opening1.pprint(ostream=r1)
        #     self.olnmpc.per_opening2.pprint(ostream=r1)
        #     r1.close()
        results = self.dot_driver.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        # with open("r2.txt", "w") as r2:
        #     self.olnmpc.per_opening1.pprint(ostream=r2)
        #     self.olnmpc.per_opening2.pprint(ostream=r2)
        #     r2.close()
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

    def stall_strategy(self, strategy, cmv=1e-04, **kwargs):  # Fix the damn stall strategy
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

    def find_target_ss(self, ref_state=None):
        """Attempt to find a second steady state
        Args:
            None
        Returns
            None"""
        if ref_state:
            self.ref_state = ref_state
        else:
            if not self.ref_state:
                self.journalizer("W", self._c_it, "find_target_ss", "No reference state was given")
                sys.exit()

        self.journalizer("I", self._c_it, "find_target_ss", "Attempting to find steady state")

        del self.ss2
        self.ss2 = self.d_mod(1, 1, steady=True)
        self.ss2.name = "ss2 (reference)"
        for u in self.u:
            cv = getattr(self.ss2, u)  #: Get the param
            c_val = [value(cv[i]) for i in cv.keys()]  #: Current value
            self.ss2.del_component(cv)  #: Delete the param
            self.ss2.add_component(u, Var(self.ss2.fe_t, initialize=lambda m, i: c_val[i-1]))
            self.ss2.equalize_u(direction="r_to_u")
            cc = getattr(self.ss2, u + "_c")  #: Get the constraint
            ce = getattr(self.ss2, u + "_e")  #: Get the expression
            cv = getattr(self.ss2, u)  #: Get the new variable
            for k in cv.keys():
                cv[k].setlb(self.u_bounds[u][0])
                cv[k].setub(self.u_bounds[u][1])
            cc.clear()
            cc.rule = lambda m, i: cv[i] == ce[i]
            cc.reconstruct()

        self.ss2.create_bounds()
        self.ss2.equalize_u(direction="r_to_u")

        for vs in self.ss.component_objects(Var, active=True):  #: Load_guess
            vt = getattr(self.ss2, vs.getname())
            for ks in vs.keys():
                vt[ks].set_value(value(vs[ks]))
        ofexp = 0
        for i in self.ref_state.keys():
            v = getattr(self.ss2, i[0])
            vkey = i[1]
            ofexp += (v[(1, 1) + vkey] - self.ref_state[i])**2
        self.ss2.obfun_ss2 = Objective(expr=ofexp, sense=minimize)

        self.solve_d(self.ss2, iter_max=500, stop_if_nopt=True)
        self.journalizer("I", self._c_it, "find_target_ss", "Target: solve done")

    def update_targets_nmpc(self):
        """Use the reference model to update  the current state and control targets"""
        for x in self.states:
            xvar = getattr(self.ss2, x)
            for j in self.state_vars[x]:
                self.curr_state_target[(x, j)] = value(xvar[1, 1, j])
        for u in self.u:
            uvar = getattr(self.ss2, u)
            self.curr_u_target[u] = value(uvar[1])

    def change_setpoint(self, ref_state):
        """Change the update the ref_state dictionary, and attempt to find a new reference state"""
        if not ref_state:
            self.journalizer("W", self._c_it, "change_setpoint", "No reference state was given")
            return
        self.ref_state = ref_state
        ofexp = 0.0
        for i in self.ref_state.keys():
            v = getattr(self.ss2, i[0])
            vkey = i[1]
            ofexp += (v[(1, 1) + vkey] - self.ref_state[i])**2

        self.ss2.obfun_ss2.set_value(ofexp)
        self.solve_d(self.ss2, iter_max=500, stop_if_nopt=True)

    def compute_offset_state(self, src_kind="estimated"):
        if src_kind == "estimated":
            for x in self.states:
                for j in self.state_vars[x]:
                    self.curr_state_offset[(x, j)] = self.curr_pstate[(x, j)] - self.curr_estate[(x, j)]
        elif src_kind == "real":
            for x in self.states:
                for j in self.state_vars[x]:
                    self.curr_state_offset[(x, j)] = self.curr_pstate[(x, j)] - self.curr_rstate[(x, j)]

    def print_r_nmpc(self):
        self.journalizer("I", self._c_it, "print_r_nmpc", "Results")
        for x in self.states:
            elist = []
            rlist = []
            xe = getattr(self.lsmhe, x)
            xr = getattr(self.d1, x)
            for j in self.x_vars[x]:
                elist.append(value(xe[(self.nfe_t, self.ncp_t) + j]))
                rlist.append(value(xr[(1, self.ncp_t) + j]))
            self.s_estimate[x].append(elist)
            self.s_real[x].append(rlist)

        with open("res_nmpc_sp.txt", "w") as f:
            for x in self.x_noisy:
                for j in range(0, len(self.s_estimate[x][0])):
                    for i in range(0, len(self.s_estimate[x])):
                        xvs = str(self.s_estimate[x][i][j])
                        f.write(xvs)
                        f.write('\t')
                    f.write('\n')
            f.close()
        with open("res_nmpc_input.txt", "w") as f:
            for x in self.x_noisy:
                for j in range(0, len(self.s_real[x][0])):
                    for i in range(0, len(self.s_real[x])):
                        xvs = str(self.s_real[x][i][j])
                        f.write(xvs)
                        f.write('\t')
                    f.write('\n')
            f.close()

        for y in self.y:
            elist = []
            rlist = []
            nlist = []
            yklst = []
            ye = getattr(self.lsmhe, y)
            yr = getattr(self.d1, y)
            for j in self.y_vars[y]:
                elist.append(value(ye[(self.nfe_t, self.ncp_t) + j]))
                rlist.append(value(yr[(1, self.ncp_t) + j]))
                nlist.append(self.curr_m_noise[(y, j)])
                yklst.append(value(self.lsmhe.yk0_mhe[self.nfe_t, self.yk_key[(y, j)]]))
            self.y_estimate[y].append(elist)
            self.y_real[y].append(rlist)
            self.y_noise_jrnl[y].append(nlist)
            self.yk0_jrnl[y].append(yklst)


    def update_soi_val(self):
        """States-of-interest"""
        for k in self.ref_state.keys():
            vname = k[0]
            vkey = k[1]
            var = getattr(self.d1, vname)
            #: Assuming the variable is indexed by time
            self.curr_soi[k] = value(var[(1,self.ncp_t) + vkey])