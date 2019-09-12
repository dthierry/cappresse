#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from pyomo.environ import *
from pyomo.core.base import ConcreteModel
from pyomo.dae import *
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.core.expr.numvalue import value
from pyomo.core.base import Constraint, Set, Param, Var, Suffix
from pyomo.core.kernel import exp

__author__ = "David Thierry @dthierry"  #: March 2018


def assert_num_vars_eq(model):
    # type: (pyomo.core.base.ConcreteModel) -> AnyWithNone
    pass


class servo_motor_dae(ConcreteModel):
    def __init__(self, nfe_t, ncp_t, **kwargs):
        #: type: (int, int, dict)
        """
            CSTR from Rodrigo's thesis
        Returns:
            cstr_rodrigo_dae: The model itself. Without discretization.
        """
        #: if steady == True fallback to steady-state computation
        self.nfe_t = nfe_t
        self.ncp_t = ncp_t
        self.discretized = False
        self.scheme = kwargs.pop('scheme', 'LAGRANGE-RADAU')
        self.steady = kwargs.pop('steady', False)
        self._t = kwargs.pop('_t', 1.0)
        ConcreteModel.__init__(self)

        if self.steady:
            self.t = Set(initialize=[1])
        else:
            self.t = ContinuousSet(bounds=(0, self._t))

        self.i = Set(initialize=[i for i in range(0, 4)])

        #: Our control var
        self.u = Var(self.t, initialize=0)
        self.u1 = Param(self.t, default=0, mutable=True)  #: We are making a sort-of port

        def u1_rule(m, t):
           # if t > 0:
            return m.u[t] == m.u1[t]
            #else:
             #   return Constraint.Skip

        # self.u1_cdummy = Constraint(self.t, rule=lambda m, i: m.Tjinb[i] == self.u1[i])
        self.u1_cdummy = Constraint(self.t, rule=u1_rule)
        #: u1 will contain the information from the NMPC problem. This is what drives the plant.
        #: how about smth like nmpc_u1 or u1_nmpc
        A = {}
        A[0, 0] = 0
        A[0, 1] = 1
        A[0, 2] = 0
        A[0, 3] = 1

        A[1, 0] = -128
        A[1, 1] = -2.5
        A[1, 2] = 6.4
        A[1, 3] = 0

        A[2, 0] = 0
        A[2, 1] = 0
        A[2, 2] = 0
        A[2, 3] = 1

        A[3, 0] = 128
        A[3, 1] = 0
        A[3, 2] = -6.4
        A[3, 3] = -10.2

        self.A = Param(self.i, self.i, initialize=A)
        h = {}
        h[0, 0] = 1
        h[0, 1] = 0
        h[0, 2] = 0
        h[0, 3] = 0

        h[1, 0] = 1282
        h[1, 1] = 0
        h[1, 2] = -64
        h[1, 3] = 0
        self.j = Set(initialize=[0, 1])
        self.hy = Param(self.j, self.i, initialize=h)
        # States
        self.x = Var(self.t, self.i, initialize=1)
        self.y0 = Var(self.t, initialize=1)
        self.y1 = Var(self.t, initialize=1)

        def yh0_rule(mod, t):
            if t > 0:
                return mod.y0[t] == sum(mod.hy[0, i] * mod.x[t, i] for i in mod.i)
            else:
                return Constraint.Skip

        def yh1_rule(mod, t):
            if t > 0:
                return mod.y1[t] == sum(mod.hy[1, i] * mod.x[t, i] for i in mod.i)
            else:
                return Constraint.Skip

        self.yh0_con = Constraint(self.t, rule=yh0_rule)
        self.yh1_con = Constraint(self.t, rule=yh1_rule)

        #: These guys have to be zero at the steady-state (steady).
        zero0 = dict.fromkeys(self.t * self.i)
        for key in zero0.keys():
            zero0[key] = 0.0
        if self.steady:
            self.xdot = zero0
        else:
            self.xdot = DerivativeVar(self.x, initialize=1)

        #: These guys as well (steady).
        self.x_ic = Param(self.i, default=0.0, mutable=True)

        def ode_x(mod, t, i):
            if t > 0:
                if i != 3:
                    return mod.xdot[t, i] == sum(mod.A[i, j] * mod.x[t, j] for j in mod.i)
                else:
                    return mod.xdot[t, i] == sum(mod.A[i, j] * mod.x[t, j] for j in mod.i) + mod.u[t]
            else:
                return Constraint.Skip

        self.de_x = Constraint(self.t, self.i, rule=ode_x)

        def _rule_x0(mod, i):
            return mod.x[0, i] == mod.x_ic[i]

        #: No need of these guys at steady.
        if self.steady:
            self.x_icc = None
        else:
            self.x_icc = Constraint(self.i, rule=_rule_x0)



        # self.kdef.reconstruct()
        # self.de_ca.reconstruct()
        # self.de_T.reconstruct()
        # self.de_Tj.reconstruct()

        # Declare at framework level
        # self.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        # self.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        # self.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        # self.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        # self.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

    def write_nl(self):
        pass

    def clear_bounds(self):
        for var in self.component_data_objects(Var):
            var.setlb(None)
            var.setub(None)
