#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from pyomo.core.base import ConcreteModel
from pyomo.environ import *
from pyomo.dae import *
from pyomo.opt import SolverFactory
from pyomo.core.kernel.numvalue import value
from pyomo.core.base import Constraint, sqrt, exp, log


def assert_num_vars_eq(model):
    # type: (pyomo.core.base.ConcreteModel) -> AnyWithNone
    pass
    

class cstr_rodrigo_dae(ConcreteModel):
    def __init__(self, nfe_t, ncp_t, **kwargs):
        self.nfe_t = nfe_t
        self.ncp_t = ncp_t
        self.scheme = kwargs.pop('scheme', 'LAGRANGE-RADAU')
        ConcreteModel.__init__(self)
        self.steady = kwargs.pop('steady', False)
        _t = kwargs.pop('_t', 1.0)

        ncstr = kwargs.pop('n_cstr', 1)
        self.ncstr = Set(initialize=[i for i in range(0, ncstr)])

        if self.steady:
            self.t = Set(initialize=[1])
        else:
            self.t = ContinuousSet(bounds=(0, _t))

        self.Cainb = Param(default=1.0)
        self.Tinb = Param(default=1.0)
        self.Tjinb = Param(default=1.0)

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
        self.Ca = Var(self.t, self.ncstr, initialize=1.0)
        self.T = Var(self.t, self.ncstr, initialize=1.0)
        self.Tj = Var(self.t, self.ncstr, initialize=1.0)

        self.k = Var(self.t, self.ncstr)
        self.kdef = Constraint(self.t, self.ncstr)

        #: These guys have to be zero at the steady-state (steady).
        zero0 = dict.fromkeys(self.t * self.ncstr)
        for key in zero0.keys():
            zero0[key] = 0.0
        if self.steady:
            self.Cadot = zero0
            self.Tdot = zero0
            self.Tjdot = zero0
        else:
            self.Cadot = DerivativeVar(self.Ca)
            self.Tdot = DerivativeVar(self.T)
            self.Tjdot = DerivativeVar(self.Tj)

        #: These guys as well (steady).
        self.Ca_ic = Param(self.ncstr, default=1.9193793974995963E-02)
        self.T_ic = Param(self.ncstr, default=3.8400724261199036E+02)
        self.Tj_ic = Param(self.ncstr, default=3.7127352272578315E+02)

        self.ODE_ca = Constraint(self.t, self.ncstr)
        self.ODE_T = Constraint(self.t, self.ncstr)
        self.ODE_Tj = Constraint(self.t, self.ncstr)

        #: No need of these guys at steady.
        if self.steady:
            self.Ca_icc = None
            self.T_icc = None
            self.Tj_icc = None
        else:
            self.Ca_icc = Constraint(self.ncstr)
            self.T_icc = Constraint(self.ncstr)
            self.Tj_icc = Constraint(self.ncstr)

        def _rule_k(m, i, n):
            if i == 0:
                return Constraint.Skip
            else:
                return m.k[i, n] == m.k0 * exp(-m.Er / m.T[i, n])

        def _rule_ca(m, i, n):
            if i == 0:
                return Constraint.Skip
            else:
                rule = m.Cadot[i, n] == (m.F[i] / m.V) * ((m.Cainb - m.Ca[i, n]) - 2 * m.k[i, n] * m.Ca[i, n] ** 2)
                return rule

        def _rule_t(m, i, n):
            if i == 0:
                return Constraint.Skip
            else:
                return m.Tdot[i, n] == (m.F[i] / m.V) * ((m.Tinb - m.T[i, n]) +
                                                         2.0 * m.dH / (m.rho * m.Cp) * m.k[i, n] * m.Ca[i, n] ** 2 -
                                                         m.UA / (m.V * m.rho * m.Cp) * (m.T[i, n] - m.Tj[i, n]))

        def _rule_tj(m, i, n):
            if i == 0:
                return Constraint.Skip
            else:
                return m.Tjdot[i, n] == \
                       (m.Fw[i] / m.Vw) * (
                                   (m.Tjinb - m.Tj[i, n]) + m.UA / (m.Vw * m.rhow * m.Cpw) * (m.T[i, n] - m.Tj[i, n]))

        def _rule_ca0(m, n):
            return m.Ca[0, n] == m.Ca_ic[n]

        def _rule_t0(m, n):
            return m.T[0, n] == m.T_ic[n]

        def _rule_tj0(m, n):
            return m.Tj[0, n] == m.Tj_ic[n]

        # let Ca0 := 1.9193793974995963E-02 ;
        # let T0  := 3.8400724261199036E+02 ;
        # let Tj0 := 3.7127352272578315E+02 ;

        self.kdef.rule = lambda m, i, n: _rule_k(m, i, n)
        self.ODE_ca.rule = lambda m, i, n: _rule_ca(m, i, n)
        self.ODE_T.rule = lambda m, i, n: _rule_t(m, i, n)
        self.ODE_Tj.rule = lambda m, i, n: _rule_tj(m, i, n)

        if self.steady:
            pass
        else:
            self.Ca_icc.rule = lambda m, n: _rule_ca0(m, n)
            self.T_icc.rule = lambda m, n: _rule_t0(m, n)
            self.Tj_icc.rule = lambda m, n: _rule_tj0(m, n)
            self.Ca_icc.reconstruct()
            self.T_icc.reconstruct()
            self.Tj_icc.reconstruct()

        self.kdef.reconstruct()
        self.ODE_ca.reconstruct()
        self.ODE_T.reconstruct()
        self.ODE_Tj.reconstruct()


        # Declare at framework level
        self.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

        self.discretizer = TransformationFactory('dae.collocation')

    def create_bounds(self):
        pass

    def write_nl(self):
        pass

    def clear_bounds(self):
        pass

    def discretize(self):
        if self.steady:
            print("Already discretized")
        else:
            self.discretizer.apply_to(self, nfe=self.nfe_t, ncp=self.ncp_t, scheme=self.scheme)


#
# def activate_bounds():
#   for k in self.Ca.keys():
#         self.Ca[k].setlb(0.0)
#   for k in self.T.keys():
#         self.T[k].setlb(3.6E+02)
#   for k in self.Tj.keys():
#         self.Tj[k].setlb(3.5E+02)
#   return
#
# def deactivate_bounds():
#   for k in self.Ca.keys():
#         self.Ca[k].setlb(None)
#   for k in self.T.keys():
#         self.T[k].setlb(None)
#   for k in self.Tj.keys():
#         self.Tj[k].setlb(None)
#   return


def t_ij(time_set, i, j):
    """Return the corresponding time"""
    # h = time_set.last()/time_set.get_discretization_info()['nfe']
    h = time_set.get_finite_elements()[1] - time_set.get_finite_elements()[0]
    tau = time_set.get_discretization_info()['tau_points']
    fe = time_set.get_finite_elements()[i]
    time = fe + tau[j] * h
    return round(time, 6)


def fe_cp(time_set, t):
    """Return the corresponding fe and cp for a given time"""
    fe_l = time_set.get_lower_element_boundary(t)
    print("fe_l", fe_l)
    fe = None
    j = 0
    for i in time_set.get_finite_elements():
        if fe_l == i:
            fe = j
            break
        j += 1
    h = time_set.get_finite_elements()[1] - time_set.get_finite_elements()[0]
    tauh = [i * h for i in time_set.get_discretization_info()['tau_points']]
    j = 0  #: Watch out for LEGENDRE
    cp = None
    for i in tauh:
        if round(i + fe_l, 6) == t:
            cp = j
            break
        j += 1
    return (fe, cp)


def main():
    mod = cstr_rodrigo_dae(2, 3)
    mod.discretize()
    mod.pprint()
    return mod


if __name__ == '__main__':
    m = main()
    ip = SolverFactory('ipopt')
    ip.solve(m, tee=True)
