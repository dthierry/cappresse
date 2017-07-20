#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix, ConstraintList
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
        for x in self.x_noisy:
            n_s = getattr(self.lsmhe, x)  #: Noisy-state
            for jth in self.x_vars[x]:  #: the jth variable
                self.xkN_l.append(n_s[(1, 0) + jth])
                self.xkN_exl.append(0)  #: exclusion list for active bounds

        self.lsmhe.xkNk = Set(initialize=[i for i in range(0, len(self.xkN_l))])  #: Create set of noisy_states

        # self.lsmhe.x_pi = Var(self.lsmhe.xkNk, initialize=0.0)  #: Create variable x-x0 for objective
        # prior-state?
        self.lsmhe.x_0 = Param(self.lsmhe.xkNk, initialize=0.0, mutable=True)  #: Prior-state
        self.lsmhe.wk = Var(self.lsmhe.fe_t, self.lsmhe.xkNk, initialize=0.0)  #: Model disturbance
        self.lsmhe.PikN = Param(self.lsmhe.xkNk, self.lsmhe.xkNk,
                                initialize=lambda m, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Prior-Covariance
        self.lsmhe.Q = Param(range(1, self.nfe_t), self.lsmhe.xkNk, self.lsmhe.xkNk,
                             initialize=lambda m, t, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Disturbance-weight

        #: Create list of measurements vars
        self.yk_l = {}
        for t in range(1, self.nfe_t + 1):
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
        self.lsmhe.R = Param(self.lsmhe.fe_t, self.lsmhe.ykk, self.lsmhe.ykk,
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
            expr=sum(sum(self.lsmhe.wk[i, j] *
                         sum(self.lsmhe.Q[i, j, k] * self.lsmhe.wk[i, k] for k in self.lsmhe.xkNk)
                         for j in self.lsmhe.xkNk) for i in range(1, self.nfe_t)))

        self.lsmhe.R_expr = Expression(
            expr=sum(sum(self.lsmhe.nuk[i, j] *
                         sum(self.lsmhe.R[i, j, k] * self.lsmhe.nuk[i, k] for k in self.lsmhe.xkNk)
                         for j in self.lsmhe.xkNk) for i in self.lsmhe.fe_t))

        self.lsmhe.Arrival_expr = Expression(
            expr=sum((self.xkN_l[j] - self.lsmhe.x_0[j]) *
                     sum(self.lsmhe.PikN[j, k] * (self.xkN_l[k] - self.lsmhe.x_0[k]) for k in self.lsmhe.xkNk)
                     for j in self.lsmhe.xkNk))

        self.lsmhe.mhe_obfun = Objective(sense=minimize,
                                         expr=self.lsmhe.Arrival_expr + self.lsmhe.Q_expr + self.lsmhe.R_expr)

        self._window_keep = self.nfe_t + 2
        l = []
        for i in product(self.states, [j for j in range(0, self._window_keep)]):
            l.append(i)

        self.xreal_W = dict.fromkeys(l, [])

    def initialize_xreal(self):
        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.name = "Dummy [xreal]"
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
                xs = getattr(dum, i)
                for k in xs.keys():
                    if k[1] == self.ncp_t:
                        self.xreal_W[(i, fe)].append(xs[k])



    def initialize_lsmhe(self, ref):
        """Initializes the lsmhe
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