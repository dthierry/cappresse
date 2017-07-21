#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix, ConstraintList
from pyomo.core.base.sets import SimpleSet
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from nmpc_mhe.dync.DynGen import DynGen
import numpy as np
from itertools import product
import sys

__author__ = "David M Thierry @dthierry"
"""Not yet."""


class MheGen(DynGen):
    def __init__(self, **kwargs):
        DynGen.__init__(self, **kwargs)

        # Need a list of relevant measurements y

        self.y = kwargs.pop('y', [])
        self.y_vars = kwargs.pop('y_vars', {})

        # Need a list or relevant noisy-states z

        self.x_noisy = kwargs.pop('x_noisy', [])
        self.x_vars = kwargs.pop('x_vars', {})
        self.deact_ics = kwargs.pop('del_ics', True)
        self.diag_Q_R = kwargs.pop('diag_QR', True)  #: By default use diagonal matrices for Q and R matrices
        self.u = kwargs.pop('u', [])

        print("-" * 120)
        print("I[[create_lsmhe]] lsmhe (full) model created.")
        print("-" * 120)

        self.lsmhe = self.d_mod(self.nfe_t, self.ncp_t, _t=self._t)
        self.lsmhe.name = "lsmhe (Least-Squares MHE)"
        self.lsmhe.create_bounds()
        #: create x_pi constraint

        #: Create list of noisy-states vars
        self.xkN_l = []
        self.xkN_exl = []
        self.xkN_key = {}
        k = 0
        for x in self.x_noisy:
            n_s = getattr(self.lsmhe, x)  #: Noisy-state
            for jth in self.x_vars[x]:  #: the jth variable
                self.xkN_l.append(n_s[(1, 0) + jth])
                self.xkN_exl.append(0)  #: exclusion list for active bounds
                self.xkN_key[(x, jth)] = k
                k += 1

        self.lsmhe.xkNk = Set(initialize=[i for i in range(0, len(self.xkN_l))])  #: Create set of noisy_states

        # self.lsmhe.x_pi = Var(self.lsmhe.xkNk, initialize=0.0)  #: Create variable x-x0 for objective
        # prior-state?
        self.lsmhe.x_0 = Param(self.lsmhe.xkNk, initialize=0.0, mutable=True)  #: Prior-state
        self.lsmhe.wk = Var(self.lsmhe.fe_t, self.lsmhe.xkNk, initialize=0.0)  #: Model disturbance
        self.lsmhe.PikN = Param(self.lsmhe.xkNk, self.lsmhe.xkNk,
                                initialize=lambda m, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Prior-Covariance
        self.lsmhe.Q_mhe = Param(range(1, self.nfe_t), self.lsmhe.xkNk, initialize=1, mutable=True) if self.diag_Q_R\
            else Param(range(1, self.nfe_t), self.lsmhe.xkNk, self.lsmhe.xkNk,
                             initialize=lambda m, t, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Disturbance-weight

        #: Create list of measurements vars
        self.yk_l = {}
        self.yk_key = {}
        k = 0
        self.yk_l[1] = []
        for y in self.y:
            m_v = getattr(self.lsmhe, y)  #: Measured "state"
            for jth in self.y_vars[y]:  #: the jth variable
                self.yk_l[1].append(m_v[(1, self.ncp_t) + jth])
                self.yk_key[(y, jth)] = k
                k += 1

        for t in range(2, self.nfe_t + 1):
            self.yk_l[t] = []
            for y in self.y:
                m_v = getattr(self.lsmhe, y)  #: Measured "state"
                for jth in self.y_vars[y]:  #: the jth variable
                    self.yk_l[t].append(m_v[(1, self.ncp_t) + jth])

        self.lsmhe.ykk = Set(initialize=[i for i in range(0, len(self.yk_l[1]))])  #: Create set of measured_vars
        self.lsmhe.nuk = Var(self.lsmhe.fe_t, self.lsmhe.ykk, initialize=0.0)   #: Measurement noise
        self.lsmhe.yk0 = Param(self.lsmhe.fe_t, self.lsmhe.ykk, initialize=1.0, mutable=True)
        self.lsmhe.hyk_c = Constraint(self.lsmhe.fe_t, self.lsmhe.ykk,
                                      rule=lambda mod, t, i: mod.yk0[t, i] - self.yk_l[t][i] - mod.nuk[t, i] == 0.0)
        self.lsmhe.R_mhe = Param(self.lsmhe.fe_t, self.lsmhe.ykk, initialize=1.0, mutable=True) if self.diag_Q_R else \
            Param(self.lsmhe.fe_t, self.lsmhe.ykk, self.lsmhe.ykk,
                             initialize=lambda mod, t, i, ii: 1.0 if i == ii else 0.0, mutable=True)

        #: Deactivate icc constraints
        if self.deact_ics:
            for i in self.states:
                self.lsmhe.del_component(i + "_icc")
        #: Maybe only for a subset of the states
        else:
            for i in self.states:
                if i in self.x_noisy:
                    ic_con = getattr(self.lsmhe, i + "_icc")
                    for k in ic_con.keys():
                        if k[2:] in self.x_vars[i]:
                            ic_con[k].deactivate()
        #: Put the noise in the continuation equations (finite-element)
        j = 0
        for i in self.x_noisy:
            cp_con = getattr(self.lsmhe, "cp_" + i)
            cp_exp = getattr(self.lsmhe, "noisy_" + i)
            for k in self.x_vars[i]:  #: This should keep the same order
                for t in range(1, self.nfe_t):
                    cp_con[t, k].set_value(cp_exp[t, k] == self.lsmhe.wk[t, j])
                j += 1

        #: Expressions for the objective function (least-squares)
        self.lsmhe.Q_expr = Expression(
            expr=sum(
                sum(
                    self.lsmhe.Q_mhe[i, k] * self.lsmhe.wk[i, k]**2 for k in self.lsmhe.xkNk)
                for i in range(1, self.nfe_t))) if self.diag_Q_R else Expression(
            expr=sum(sum(self.lsmhe.wk[i, j] *
                         sum(self.lsmhe.Q_mhe[i, j, k] * self.lsmhe.wk[i, k] for k in self.lsmhe.xkNk)
                         for j in self.lsmhe.xkNk) for i in range(1, self.nfe_t)))

        self.lsmhe.R_expr = Expression(
            expr=sum(
                sum(
                    self.lsmhe.R_mhe[i, k] * self.lsmhe.nuk[i, k]**2 for k in self.lsmhe.xkNk)
                for i in self.lsmhe.fe_t)) if self.diag_Q_R else Expression(
            expr=sum(sum(self.lsmhe.nuk[i, j] *
                         sum(self.lsmhe.R_mhe[i, j, k] * self.lsmhe.nuk[i, k] for k in self.lsmhe.xkNk)
                         for j in self.lsmhe.xkNk) for i in self.lsmhe.fe_t))

        self.lsmhe.Arrival_expr = Expression(
            expr=sum((self.xkN_l[j] - self.lsmhe.x_0[j]) *
                     sum(self.lsmhe.PikN[j, k] * (self.xkN_l[k] - self.lsmhe.x_0[k]) for k in self.lsmhe.xkNk)
                     for j in self.lsmhe.xkNk))

        self.lsmhe.mhe_obfun_dum = Objective(sense=minimize,
                                             expr=self.lsmhe.Q_expr + self.lsmhe.R_expr)
        self.lsmhe.mhe_obfun_dum.deactivate()


        self.lsmhe.mhe_obfun = Objective(sense=minimize,
                                         expr=self.lsmhe.Arrival_expr + self.lsmhe.Q_expr + self.lsmhe.R_expr)



        self.xreal_W = {}

    def initialize_xreal(self, ref):
        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.name = "Dummy [xreal]"
        self.load_d_d(ref, dum, 1)
        # ref.display(filename="somefile1.txt")
        # dum.display(filename="somefile2.txt")
        for fe in range(1, self._window_keep):
            for i in self.states:
                pn = i + "_ic"
                p = getattr(dum, pn)
                vs = getattr(dum, i)
                for ks in p.iterkeys():
                    p[ks].value = value(vs[(1, self.ncp_t) + (ks,)])
            #: Solve
            self.solve_d(dum, o_tee=False)
            for i in self.states:
                self.xreal_W[(i, fe)] = []
                xs = getattr(dum, i)
                for k in xs.keys():
                    if k[1] == self.ncp_t:
                        print(i)
                        self.xreal_W[(i, fe)].append(value(xs[k]))

    def initialize_lsmhe(self, ref):
        """Initializes the lsmhe DEAL WITH INPUTS!!
        Args
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model"""
        self.journalizer("I", self._c_it, "initialize_olnmpc", "Attempting to initialize lsmhe")
        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.name = "Dummy I"
        #: Load current solution
        self.load_d_d(ref, dum, 1)
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
            #: Solve
            self.solve_d(dum, o_tee=False)
            #: Patch
            self.load_d_d(dum, self.olnmpc, finite_elem)

        self.journalizer("I", self._c_it, "initialize_olnmpc", "Attempting to initialize olnmpc Done")

    def extract_meas_(self, t, **kwargs):
        """Mechanism to assign a value of y0 to the current mhe from the dynamic model
        Args:
            t (int): int The current collocation point
        Returns:
            meas_dict (dict): A dictionary containing the measurements list by meas_var
        """
        src = kwargs.pop("src", self.d1)
        skip_update = kwargs.pop("skip_update", False)

        meas_dic = dict.fromkeys(self.y)
        l = []
        for i in self.y:
            lm = []
            var = getattr(src, i)
            for j in self.y_vars[i]:
                lm.append(value(var[(1, self.ncp_t,) + j]))
                l.append(value(var[(1, self.ncp_t,) + j]))
            meas_dic[i] = lm

        if not skip_update:  #: Update the mhe model
            y0dest = getattr(self.lsmhe, "yk0")

            for i in self.y:
                for j in self.y_vars[i]:
                    k = self.yk_key[(i, j)]
                    y0dest[t, k] = l[k]
        return meas_dic

    def adjust_nu0_mhe(self):
        """Adjust the initial guess for the nu variable"""
        for t in self.lsmhe.fe_t:
            k = 0
            for i in self.y:
                for j in self.y_vars[i]:
                    target = value(self.lsmhe.yk0[t, k]) - value(self.yk_l[t][k])
                    self.lsmhe.nuk[t, k].set_value(target)
                    k += 1

    def set_covariance_meas(self, cov_dict):
        """Sets covariance(inverse) for the measurements.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(meas_name, j), (meas_name, k), time]
        Returns:
            None
        """
        rtarget = getattr(self.lsmhe, "R_mhe")
        for key in cov_dict:
            vni = key[0]
            vnj = key[1]
            _t = key[2]

            v_i = self.yk_key[vni]
            v_j = self.yk_key[vnj]
            # try:
            if self.diag_Q_R:
                rtarget[_t, v_i] = 1 / cov_dict[vni, vnj, _t]
            else:
                rtarget[_t, v_i, v_j] = cov_dict[vni, vnj, _t]
            # except KeyError:
            #     print("Key error, {:} {:} {:}".format(vni, vnj, _t))

    def set_covariance_disturb(self, cov_dict):
        """Sets covariance(inverse) for the measurements.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(meas_name, j), (meas_name, k), time]
        Returns:
            None
        """
        qtarget = getattr(self.lsmhe, "Q_mhe")
        for key in cov_dict:
            vni = key[0]
            vnj = key[1]
            _t = key[2]
            v_i = self.xkN_key[vni]
            v_j = self.xkN_key[vnj]
            if self.diag_Q_R:
                qtarget[_t, v_i] = 1 / cov_dict[vni, vnj, _t]
            else:
                qtarget[_t, v_i, v_j] = cov_dict[vni, vnj, _t]

    def shift_mhe(self):
        """Shifts current initial guesses of variables for the mhe problem"""
        for v in self.lsmhe.component_objects(Var, active=True):
            if type(v.index_set()) == SimpleSet:  #: Don't want simple sets
                break
            else:
                kl = v.keys()
                if len(kl[0]) < 2:
                    break
                for k in kl:
                    if k[0] < self.nfe_t:
                        try:
                            v[k].set_value(v[(k[0] + 1,) + k[1:]])
                        except ValueError:
                            continue

