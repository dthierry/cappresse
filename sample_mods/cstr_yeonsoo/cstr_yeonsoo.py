#!/usr/bin/env python3
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


class cstr_yeonsoo_dae(ConcreteModel):
    def __init__(self, nfe_t, ncp_t, **kwargs):
        #: type: (int, int, dict)
        """
            CSTR from Yeonsoo's paper,  first example of amsNMPC
        Returns:
            cstr_yeonsoo_dae: The model itself. Without discretization.
        """
        self.nfe_t = nfe_t  
        self.ncp_t = ncp_t
        self.discretized = False
        self.scheme = kwargs.pop('scheme', 'LAGRANGE-RADAU')
        self.steady = kwargs.pop('steady', False)
        self._t = kwargs.pop('_t', 1.0)
        ncstr = kwargs.pop('n_cstr', 1) #number of cstr KH.L
        ConcreteModel.__init__(self) #similar to self = ConcreteModel() KH.L
        self.ncstr = Set(initialize=[i for i in range(0, ncstr)]) # self.ncstr = Set(initialize = 0)

        if self.steady:
            self.t = Set(initialize=[1])
        else:
            self.t = ContinuousSet(bounds=(0, self._t)) # 0~1
            
        # Control inputs:
        self.u1 = Var(self.t, initialize = 800.)
        self.u2 = Var(self.t, initialize = 10.)
        
        self.d_u1 = Param(self.t, default = 800., mutable = True)
        self.d_u2 = Param(self.t, default = 10., mutable = True)
        
        #dummy_rule
        def u1_rule(m,i):
            return m.u1[i] == m.d_u1[i]
        def u2_rule(m,i):
            return m.u2[i] == m.d_u2[i]
        
        #dummy constraints
        self.d_u1_cdummy = Constraint(self.t, rule = u1_rule)
        self.d_u2_cdummy = Constraint(self.t, rule = u2_rule)
        
        #Parameters
        self.zf = Param(initialize = 0.3947368421)
        self.zc = Param(initialize = 0.3815789474) 
        beta = 3.
        self.k = Param(initialize = 300 * 7.6 ** (beta - 1))
        self.E = Param(initialize = 5.)
        self.alpha = Param(initialize = 1.95 * 10 ** (-4))
        
        #states
        self.z1 = Var(self.t, self.ncstr, initialize = 0.175)
        self.z2 = Var(self.t, self.ncstr, initialize = 0.7)
        
        #: These guys have to be zero at the steady-state (steady).
        zero0 = dict.fromkeys(self.t * self.ncstr)
        for key in zero0.keys():
            zero0[key] = 0.0
        if self.steady:
            self.z1dot = zero0
            self.z2dot = zero0
        else:
            self.z1dot = DerivativeVar(self.z1, wrt = self.t, initialize = 0.0)
            self.z2dot = DerivativeVar(self.z2, wrt = self.t, initialize = 0.0)
            
        self.z1_ic = Param(self.ncstr, default=0.175, mutable=True)
        self.z2_ic = Param(self.ncstr, default=0.7, mutable=True)
        
        self.de_z1 = Constraint(self.t, self.ncstr)
        self.de_z2 = Constraint(self.t, self.ncstr)
        
        #: No need of these guys at steady.
        if self.steady:
            self.z1_icc = None
            self.z2_icc = None
        else:
            self.z1_icc = Constraint(self.ncstr)
            self.z2_icc = Constraint(self.ncstr)
            
        def _rule_z1(m,i,n):
            if i == 0:
                return Constraint.Skip
            else:
                return m.z1dot[i,n] == 1./m.u2[i] * (1.-m.z1[i,n]) - m.k * exp(-m.E/m.z2[i,n]) * m.z1[i,n]**3
                
        def _rule_z2(m,i,n):
            if i == 0:
                return Constraint.Skip
            else:
                return m.z2dot[i,n] == 1./m.u2[i] * (m.zf - m.z2[i,n]) + m.k * exp(-m.E/m.z2[i,n]) * m.z1[i,n]**3 - m.alpha*m.u1[i]*(m.z2[i,n]-m.zc)
            
        def _rule_z10(m,n):
            return m.z1[0,n] - m.z1_ic[n] == 0.
        
        def _rule_z20(m,n):
            return m.z2[0,n] - m.z2_ic[n] == 0.
        
        self.de_z1.rule = lambda m, i, n: _rule_z1(m, i, n)
        self.de_z2.rule = lambda m, i, n: _rule_z2(m, i, n)
        self.de_z1.reconstruct()
        self.de_z2.reconstruct()
        
        if self.steady:
            pass
        else:
            self.z1_icc.rule = lambda m, n: _rule_z10(m, n)
            self.z2_icc.rule = lambda m, n: _rule_z20(m, n)
            self.z1_icc.reconstruct()
            self.z2_icc.reconstruct()
            
            

            
        
        

