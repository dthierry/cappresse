#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix, ConstraintList
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from nmpc_mhe.dync.DynGen import DynGen
import numpy as np
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

        #: Create list of noisy states
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
        self.x_0 = Param(self.lsmhe.xkNk, initialize=0.0, mutable=True)  #: Prior-state
        self.wk = Var(self.nfe_t, self.lsmhe.xkNk, initialize=0.0)  #: Model disturbance
        self.PikN = Param(self.lsmhe.xkNk, self.lsmhe.xkNk,
                          initialize=lambda m, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Prior-Covariance
        self.Q = Param(range(1, self.nfe_t), self.lsmhe.xkNk, self.lsmhe.xkNk,
                       initialize=lambda m, t, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Disturbance-weight

        #: Create list of measurements vars
        self.yk_l = {}
        for t in range(1, self.nfe_t + 1):
            self.yk_l[t] = []
            for y in self.y:
                m_v = getattr(self.lsmhe, y)  #: Measured "state"
                for jth in self.y_vars[y]:  #: the jth variable
                    self.yk_l[t].append(m_v[(1, 0) + jth])

        self.lsmhe.ykk = Set(initialize=[i for i in range(0, len(self.yk_l))])  #: Create set of measured_vars
        self.lsmhe.nuk = Var(self.lsmhe.fe_t, self.lsmhe.ykk, initialize=0.0)   #: Measurement noise
        self.lsmhe.yk0 = Param(self.lsmhe.fe_t, self.lsmhe.ykk, initialize=1.0)
        self.lsmhe.hyk_c = Constraint(self.lsmhe.fe_t, self.lsmhe.ykk,
                                      rule=lambda mod, t, i: mod.yk0[t, i] - self.yk_l[t][i] - mod.nuk[t, i] == 0.0)
        self.R = Param(self.lsmhe.fe_t, self.lsmhe.ykk, self.lsmhe.ykk,
                       initialize=lambda mod, t, i, ii: 1.0 if i == ii else 0.0)

        if self.deact_ics:
            for i in self.states:
                self.lsmhe.del_component(i + "_icc")
        else:
            for i in self.states:
                if i in self.x_noisy:
                    ic_con = getattr(self.lsmhe, i + "_icc")
                    for k in ic_con.keys():
                        if k[2:] in self.x_vars[i]:
                            ic_con[k].deactivate()
        j = 0
        for i in self.x_noisy:
            cp_con = getattr(self.lsmhe, "cp_" + i)
            cp_exp = getattr(self.lsmhe, "noisy_" + i)
            for k in self.x_vars[i]:
                for t in range(1, self.nfe_t):
                    cp_con[t, k].set_value(cp_exp[t, k] == self.lsmhe.wk[t, j])
                    j += 1



