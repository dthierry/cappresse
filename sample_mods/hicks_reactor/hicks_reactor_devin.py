#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from pyomo.environ import *
from pyomo.core.base import ConcreteModel
from pyomo.dae import *
from pyomo.opt import SolverFactory, TerminationCondition
from pyomo.core.base.numvalue import value
from pyomo.core.base import Constraint, Set, Param, Var, Suffix
from pyomo.core.kernel import exp

__author__ = "Kuan-Han Lin" #: Jul 2020


class hicks_reactor_devin_dae_w_AE(ConcreteModel):
    def __init__(self, nfe_t, ncp_t, **kwargs):
        #: type: (int, int, dict)
        """
            Hicks reactor from Devin's thesis
        Returns:
            hicks_reactor_devin_dae: The model itself. Without discretization.
        """
        #: if steady == True fallback to steady-state computation
        self.nfe_t = nfe_t  
        self.ncp_t = ncp_t
        self.discretized = False
        self.scheme = kwargs.pop('scheme', 'LAGRANGE-RADAU')
        self.steady = kwargs.pop('steady', False)
        self._t = kwargs.pop('_t', 1.0)
        ncstr = kwargs.pop('n_cstr', 1)
        ConcreteModel.__init__(self)
        self.ncstr = Set(initialize=[i for i in range(0, ncstr)])
        
        if self.steady:
            self.t = Set(initialize=[1])
        else:
            self.t = ContinuousSet(bounds=(0, self._t))
            
        #control inputs
        self.u1 = Var(self.t, initialize = 0.5833)
        self.u2 = Var(self.t, initialize = 0.5)
        
        self.d_u1 = Param(self.t, default = 0.5833, mutable = True)
        self.d_u2 = Param(self.t, default = 0.5, mutable = True)
        
        #dummy_rule
        def u1_rule(m,i):
            return m.u1[i] == m.d_u1[i]
        def u2_rule(m,i):
            return m.u2[i] == m.d_u2[i]
        
        #dummy constraints
        self.d_u1_cdummy = Constraint(self.t, rule = u1_rule)
        self.d_u2_cdummy = Constraint(self.t, rule = u2_rule)
        
        #parameters
        self.zcw = Param(initialize = 0.38)
        self.zf = Param(initialize = 0.395)
        self.Ea = Param(initialize = 5.)
        self.nu = Param(initialize = 1.94E-4)
        self.k0 = Param(initialize = 300.)
        self.U1 = Param(initialize = 600.)
        self.U2 = Param(initialize = 40.)
        
        #states
        self.zc = Var(self.t, self.ncstr, initialize = 0.6416)
        self.zT = Var(self.t, self.ncstr, initialize = 0.5387)
        
        self.AV = Var(self.t, self.ncstr, initialize = 345.)
        self.AE = Constraint(self.t, self.ncstr)

        #: These guys have to be zero at the steady-state (steady).
        zero0 = dict.fromkeys(self.t * self.ncstr)
        for key in zero0.keys():
            zero0[key] = 0.0
        if self.steady:
            self.zcdot = zero0
            self.zTdot = zero0
        else:
            self.zcdot = DerivativeVar(self.zc, wrt = self.t, initialize = 0.0)
            self.zTdot = DerivativeVar(self.zT, wrt = self.t, initialize = 0.0)
            
        self.zc_ic = Param(self.ncstr, default=0.6416, mutable=True)
        self.zT_ic = Param(self.ncstr, default=0.5387, mutable=True)
        
        self.de_zc = Constraint(self.t, self.ncstr)
        self.de_zT = Constraint(self.t, self.ncstr)
        
        #: No need of these guys at steady.
        if self.steady:
            self.zc_icc = None
            self.zT_icc = None
        else:
            self.zc_icc = Constraint(self.ncstr)
            self.zT_icc = Constraint(self.ncstr)
            
        def _rule_zc(m,i,n):
            if i == 0:
                return Constraint.Skip
            else:
                return m.zcdot[i,n] == (1-m.zc[i,n])/(m.U2*m.u2[i]) - m.AV[i, n]
        
        def _rule_zT(m,i,n):
            if i == 0:
                return Constraint.Skip
            else:
                return m.zTdot[i,n] == (m.zf-m.zT[i,n])/(m.U2*m.u2[i]) + m.AV[i, n] - m.nu*m.U1*m.u1[i]*(m.zT[i,n]-m.zcw)
        
        def _rule_AE(m,i,n):
            if i == 0:
                return Constraint.Skip
            else:
                return m.AV[i, n] == m.k0*m.zc[i,n]*exp(-m.Ea/m.zT[i,n])
        
        def _rule_zc0(m,n):
            return m.zc[0,n] - m.zc_ic[n] == 0.0
        
        def _rule_zT0(m,n):
            return m.zT[0,n] - m.zT_ic[n] == 0.0
        
        self.AE.rule = lambda m, i, n: _rule_AE(m, i, n)
        self.de_zc.rule = lambda m, i, n: _rule_zc(m, i, n)
        self.de_zT.rule = lambda m, i, n: _rule_zT(m, i, n)
        self.AE.reconstruct()
        self.de_zc.reconstruct()
        self.de_zT.reconstruct()
        
        if self.steady:
            pass
        else:
            self.zc_icc.rule = lambda m, n: _rule_zc0(m, n)
            self.zT_icc.rule = lambda m, n: _rule_zT0(m, n)
            self.zc_icc.reconstruct()
            self.zT_icc.reconstruct()

