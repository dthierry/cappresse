#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from pyomo.environ import *
from pyomo.core.base import ConcreteModel
from pyomo.dae import *
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.core.kernel.numvalue import value
from pyomo.core.base import Constraint, Set, Param, Var, Suffix
from pyomo.core.kernel import exp


__author__ = "David Thierry @dthierry" #: March 2018

def assert_num_vars_eq(model):
    # type: (pyomo.core.base.ConcreteModel) -> AnyWithNone
    pass
    

class cstr_rodrigo_dae(ConcreteModel):
    def __init__(self, nfe_t, ncp_t, **kwargs):
        #: type: (int, int, dict)
        """
            CSTR from Rodrigo's thesis
        Returns:
            cstr_rodrigo_dae: The model itself. Without discretization.
        """
        #: if steady == True fallback to steady-state computation
        self.nfe_t = nfe_t  #:
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

        self.Cainb = Param(default=1.0)
        self.Tinb = Param(default=275.0)
        # self.Tjinb = Param(default=250.0)

        #: Our control var
        self.Tjinb = Var(self.t, initialize=250)
        self.u1 = Param(self.t, default=250, mutable=True)  #: We are making a sort-of port
        def u1_rule(m, i):
            return m.Tjinb[i] == m.u1[i]

        # self.u1_cdummy = Constraint(self.t, rule=lambda m, i: m.Tjinb[i] == self.u1[i])
        self.u1_cdummy = Constraint(self.t, rule=u1_rule)
        #: u1 will contain the information from the NMPC problem. This is what drives the plant.
        #: how about smth like nmpc_u1 or u1_nmpc

        self.V = Param(initialize=100)
        self.UA = Param(initialize=20000 * 60)
        self.rho = Param(initialize=1000)
        self.Cp = Param(initialize=4.2)
        self.Vw = Param(initialize=10)
        self.rhow = Param(initialize=1000)
        self.Cpw = Param(initialize=4.2)
        self.k0 = Param(initialize=4.11e13)
        self.E = Param(initialize=76534.704)
        self.R = Param(initialize=8.314472)
        self.Er = Param(initialize=lambda m: (value(self.E) / value(self.R)))
        self.dH = Param(initialize=596619.)

        self.F = Param(self.t, mutable=True, default=1.2000000000000000E+02)
        self.Fw = Param(self.t, mutable=True, default=3.0000000000000000E+01)

        # States
        self.Ca = Var(self.t,  initialize=1.60659680385930765667001907104350E-02)
        self.T = Var(self.t,  initialize=3.92336059452774350120307644829154E+02)
        self.Tj = Var(self.t,  initialize=3.77995395658401662331016268581152E+02)

        self.k = Var(self.t,  initialize=4.70706140E+02)
        self.kdef = Constraint(self.t)

        #: These guys have to be zero at the steady-state (steady).
        zero0 = dict.fromkeys(self.t)
        for key in zero0.keys():
            zero0[key] = 0.0
        if self.steady:
            self.Cadot = zero0
            self.Tdot = zero0
            self.Tjdot = zero0
        else:
            self.Cadot = DerivativeVar(self.Ca, initialize=-3.58709135E+01)
            self.Tdot = DerivativeVar(self.T, initialize=5.19191848E+03)
            self.Tjdot = DerivativeVar(self.Tj, initialize=-9.70467399E+02)
        #: These guys as well (steady).
        self.Ca_ic = Param( default=1.9193793974995963E-02, mutable=True)
        self.T_ic = Param( default=3.8400724261199036E+02, mutable=True)
        self.Tj_ic = Param( default=3.7127352272578315E+02, mutable=True)

        # m.Ca_ic = Param(m.ncstr, default=1.9193793974995963E-02)
        # m.T_ic = Param(m.ncstr, default=3.8400724261199036E+02)
        # m.Tj_ic = Param(m.ncstr, default=3.7127352272578315E+02)

        self.de_ca = Constraint(self.t)
        self.de_T = Constraint(self.t)
        self.de_Tj = Constraint(self.t)

        #: No need of these guys at steady.
        if self.steady:
            self.Ca_icc = None
            self.T_icc = None
            self.Tj_icc = None
        # else:
            # self.Ca_icc = Constraint()
            # self.T_icc = Constraint()
            # self.Tj_icc = Constraint()

        def _rule_k(m, i):
            if i == 0:
                return Constraint.Skip
            else:
                return m.k[i] == m.k0 * exp(-m.Er / m.T[i])

        def _rule_ca(m, i):
            if i == 0:
                return Constraint.Skip
            else:
                rule = m.Cadot[i] == (m.F[i] / m.V) * (m.Cainb - m.Ca[i]) - 2 * m.k[i] * m.Ca[i] ** 2
                return rule

        def _rule_t(m, i):
            if i == 0:
                return Constraint.Skip
            else:
                return m.Tdot[i] == (m.F[i] / m.V) * (m.Tinb - m.T[i]) + \
                2.0 * m.dH / (m.rho * m.Cp) * m.k[i] * m.Ca[i] ** 2 -\
                m.UA / (m.V * m.rho * m.Cp) * (m.T[i] - m.Tj[i])

        def _rule_tj(m, i):
            if i == 0:
                return Constraint.Skip
            else:
                return m.Tjdot[i] == \
                       (m.Fw[i] / m.Vw) * (m.Tjinb[i] - m.Tj[i]) + m.UA / (m.Vw * m.rhow * m.Cpw) * (m.T[i] - m.Tj[i])

        def _rule_ca0(m):
            return m.Ca[0] == m.Ca_ic

        def _rule_t0(m):
            return m.T[0] == m.T_ic

        def _rule_tj0(m):
            return m.Tj[0] == m.Tj_ic

        # let Ca0 := 1.9193793974995963E-02 ;
        # let T0  := 3.8400724261199036E+02 ;
        # let Tj0 := 3.7127352272578315E+02 ;

        self.kdef.rule = lambda m, i: _rule_k(m, i)
        self.de_ca.rule = lambda m, i: _rule_ca(m, i)
        self.de_T.rule = lambda m, i: _rule_t(m, i)
        self.de_Tj.rule = lambda m, i: _rule_tj(m, i)

        if self.steady:
            pass
        else:
            self.Ca_icc = Constraint(rule=_rule_ca0)
            # self.Ca_icc.rule = _rule_ca0
            self.T_icc = Constraint(rule=_rule_t0)
            self.Tj_icc = Constraint(rule=_rule_tj0)
            # self.Tj_icc = Constraint(
            # self.T_icc.rule = lambda m: _rule_t0(m)
            # self.Tj_icc.rule = lambda m: _rule_tj0(m)
            # self.Ca_icc.reconstruct()
            # self.T_icc.reconstruct()
            # self.Tj_icc.reconstruct()

        self.kdef.reconstruct()
        self.de_ca.reconstruct()
        self.de_T.reconstruct()
        self.de_Tj.reconstruct()

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
