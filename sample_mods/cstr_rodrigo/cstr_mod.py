#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from pyomo.environ import *
from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory

m = ConcreteModel()
ncstr = 1
m.ncstr = Set(initialize=[i for i in range(0, ncstr)])
m.t = ContinuousSet(bounds=(0, 1))


m.Cainb = Param(default=1.0)
m.Tinb = Param(default=1.0)
m.Tjinb = Param(default=1.0)

m.V = Param(initialize=100)
m.UA = Param(initialize=20000*60)
m.rho = Param(initialize=1000)
m.Cp = Param(initialize=4.2)
m.Vw = Param(initialize=10)
m.rhow = Param(initialize=1000)
m.Cpw = Param(initialize=4.2)
m.k0 = Param(initialize=4.11e13)
m.E = Param(initialize=76534.704)
m.R = Param(initialize=8.314472)
m.Er = Param(initialize=lambda m: (m.E/m.R))
m.dH = Param(initialize=596619.)

m.F = Param(m.t, mutable=True, default=1.2000000000000000E+02)
m.Fw = Param(m.t, mutable=True, default=3.0000000000000000E+01)

m.Ca_ic = Param(m.ncstr, default=1.9193793974995963E-02)
m.T_ic = Param(m.ncstr, default=3.8400724261199036E+02)
m.Tj_ic = Param(m.ncstr, default=3.7127352272578315E+02)

# print(value(m.Er))
# m.Ca_0 = Var(m.t, m.ncstr)
# m.T_0 = Var(m.t, m.ncstr)
# m.Tj_0 = Var(m.t, m.ncstr)
# States
m.Ca = Var(m.t, m.ncstr, initialize=1.0)
m.T = Var(m.t, m.ncstr, initialize=1.0)
m.Tj = Var(m.t, m.ncstr, initialize=1.0)

m.k = Var(m.t, m.ncstr)

m.Cadot = DerivativeVar(m.Ca)
m.Tdot = DerivativeVar(m.T)
m.Tjdot = DerivativeVar(m.Tj)


m.kdef = Constraint(m.t, m.ncstr)

m.ODE_ca = Constraint(m.t, m.ncstr)
m.ODE_T = Constraint(m.t, m.ncstr)
m.ODE_Tj = Constraint(m.t, m.ncstr)

m.Ca_icc = Constraint(m.ncstr)
m.T_icc = Constraint(m.ncstr)
m.Tj_icc = Constraint(m.ncstr)


def _rule_k(m, i, n):
    if i == 0:
        return Constraint.Skip
    else:
        return m.k[i, n] == m.k0 * exp(-m.Er/m.T[i, n])

def _rule_ca(m, i, n):
    if i == 0:
        return Constraint.Skip
    else:
        return m.Cadot[i, n] == (m.F[i]/m.V) * ((m.Cainb - m.Ca[i, n]) - 2 * m.k[i, n] * m.Ca[i, n] ** 2)


def _rule_t(m, i, n):
    if i == 0:
        return Constraint.Skip
    else:
        return m.Tdot[i, n] == (m.F[i]/m.V) * ((m.Tinb - m.T[i, n]) +
                                               2.0 * m.dH / (m.rho * m.Cp) * m.k[i, n] * m.Ca[i, n] ** 2 -
                                               m.UA / (m.V * m.rho * m.Cp) * (m.T[i, n] - m.Tj[i, n]))

def _rule_tj(m, i, n):
    if i == 0:
        return Constraint.Skip
    else:
        return m.Tjdot[i, n] == \
               (m.Fw[i] / m.Vw) * ((m.Tjinb - m.Tj[i, n]) + m.UA / (m.Vw * m.rhow * m.Cpw) * (m.T[i, n] - m.Tj[i, n]))


def _rule_ca0(m, n):
    return m.Ca[0, n] == m.Ca_ic[n]


def _rule_t0(m, n):
    return m.T[0, n] == m.T_ic[n]


def _rule_tj0(m, n):
    return m.Tj[0, n] == m.Tj_ic[n]


# let Ca0 := 1.9193793974995963E-02 ;
# let T0  := 3.8400724261199036E+02 ;
# let Tj0 := 3.7127352272578315E+02 ;


m.kdef.rule = lambda m, i, n: _rule_k(m, i, n)
m.ODE_ca.rule = lambda m, i, n: _rule_ca(m, i, n)
m.ODE_T.rule = lambda m, i, n: _rule_t(m, i, n)
m.ODE_Tj.rule = lambda m, i, n: _rule_tj(m, i, n)
m.Ca_icc.rule = lambda m, n: _rule_ca0(m, n)
m.T_icc.rule = lambda m, n: _rule_t0(m, n)
m.Tj_icc.rule = lambda m, n: _rule_tj0(m, n)

m.Ca.pprint()
def activate_bounds():
      for k in m.Ca.keys():
            m.Ca[k].setlb(0.0)
      for k in m.T.keys():
            m.T[k].setlb(3.6E+02)
      for k in m.Tj.keys():
            m.Tj[k].setlb(3.5E+02)
      return
def deactivate_bounds():
      for k in m.Ca.keys():
            m.Ca[k].setlb(None)
      for k in m.T.keys():
            m.T[k].setlb(None)
      for k in m.Tj.keys():
            m.Tj[k].setlb(None)
      return


activate_bounds()
m.Ca.pprint()
d = TransformationFactory('dae.collocation')
d.apply_to(m, nfe=3, ncp=2, scheme='LAGRANGE-RADAU')



#ip = SolverFactory('ipopt')
#ip.solve(m, tee=True)


deactivate_bounds()
m.Ca.pprint()

