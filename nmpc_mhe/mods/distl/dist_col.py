#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from pyomo.core.base import ConcreteModel, Set, Constraint, Var,\
    Param, Objective, minimize, sqrt, exp, Suffix, Expression, value
from nmpc_mhe.aux.cpoinsc import collptsgen
from nmpc_mhe.aux.lagrange_f import lgr, lgry, lgrdot, lgrydot
from dist_col_mod import *
from six import itervalues, iterkeys, iteritems
from pyomo.opt import ProblemFormat, SolverFactory
import re, os

"""
Version 03.
Need a reference model that can initialize the reference steady-state model.
"""

__author__ = 'David M Thierry @dthierry'


class DistDiehlNegrete(ConcreteModel):
    def __init__(self, nfe_t, ncp_t, **kwargs):
        ConcreteModel.__init__(self)

        steady = kwargs.pop('steady', False)
        _t = kwargs.pop('_t', 1.0)
        Ntray = kwargs.pop('Ntray', 42)
        # --------------------------------------------------------------------------------------------------------------
        # Orthogonal Collocation Parameters section

        # Radau
        self._alp_gauB_t = 1
        self._bet_gauB_t = 0

        if steady:
            print("[I] " + str(self.__class__.__name__) + " NFE and NCP Overriden - Steady state mode")
            self.nfe_t = 1
            self.ncp_t = 1
        else:
            self.nfe_t = nfe_t
            self.ncp_t = ncp_t

        self.tau_t = collptsgen(self.ncp_t, self._alp_gauB_t, self._bet_gauB_t)

        # start at zero
        self.tau_i_t = {0: 0.}
        # create a list

        for ii in range(1, self.ncp_t + 1):
            self.tau_i_t[ii] = self.tau_t[ii - 1]

        # ======= SETS ======= #
        # For finite element = 1 .. NFE
        # This has to be > 0

        self.fe_t = Set(initialize=[ii for ii in range(1, self.nfe_t + 1)])

        # collocation points
        # collocation points for differential variables
        self.cp_t = Set(initialize=[ii for ii in range(0, self.ncp_t + 1)])
        # collocation points for algebraic variables
        self.cp_ta = Set(within=self.cp_t, initialize=[ii for ii in range(1, self.ncp_t + 1)])

        # create collocation param
        self.taucp_t = Param(self.cp_t, initialize=self.tau_i_t)

        self.ldot_t = Param(self.cp_t, self.cp_t, initialize=
        (lambda m, j, k: lgrdot(k, m.taucp_t[j], self.ncp_t, self._alp_gauB_t, self._bet_gauB_t)))  #: watch out for this!

        self.l1_t = Param(self.cp_t, initialize=
        (lambda m, j: lgr(j, 1, self.ncp_t, self._alp_gauB_t, self._bet_gauB_t)))

        # --------------------------------------------------------------------------------------------------------------
        # Model parameters
        self.Ntray = Ntray
        self.tray = Set(initialize=[i for i in range(1, Ntray + 1)])
        self.feed = Param(self.tray,
                          initialize=lambda m, t: 57.5294 if t == 21 else 0.0,
                          mutable=True)

        self.xf = Param(initialize=0.32, mutable=True)  # feed mole fraction
        self.hf = Param(initialize=9081.3)  # feed enthalpy

        self.hlm0 = Param(initialize=2.6786e-04)
        self.hlma = Param(initialize=-0.14779)
        self.hlmb = Param(initialize=97.4289)
        self.hlmc = Param(initialize=-2.1045e04)

        self.hln0 = Param(initialize=4.0449e-04)
        self.hlna = Param(initialize=-0.1435)
        self.hlnb = Param(initialize=121.7981)
        self.hlnc = Param(initialize=-3.0718e04)

        self.r = Param(initialize=8.3147)
        self.a = Param(initialize=6.09648)
        self.b = Param(initialize=1.28862)
        self.c1 = Param(initialize=1.016)
        self.d = Param(initialize=15.6875)
        self.l = Param(initialize=13.4721)
        self.f = Param(initialize=2.615)

        self.gm = Param(initialize=0.557)
        self.Tkm = Param(initialize=512.6)
        self.Pkm = Param(initialize=8.096e06)

        self.gn = Param(initialize=0.612)
        self.Tkn = Param(initialize=536.7)
        self.Pkn = Param(initialize=5.166e06)

        self.CapAm = Param(initialize=23.48)
        self.CapBm = Param(initialize=3626.6)
        self.CapCm = Param(initialize=-34.29)

        self.CapAn = Param(initialize=22.437)
        self.CapBn = Param(initialize=3166.64)
        self.CapCn = Param(initialize=-80.15)

        self.pstrip = Param(initialize=250)
        self.prect = Param(initialize=190)

        def _p_init(m, t):
            ptray = 9.39e04
            if t <= 20:
                return _p_init(m, 21) + m.pstrip * (21 - t)
            elif 20 < t < m.Ntray:
                return ptray + m.prect * (m.Ntray - t)
            elif t == m.Ntray:
                return 9.39e04

        self.p = Param(self.tray, initialize=_p_init)

        self.T29_des = Param(initialize=343.15)
        self.T15_des = Param(initialize=361.15)
        self.Dset = Param(initialize=1.83728)
        self.Qcset = Param(initialize=1.618890)
        self.Qrset = Param(initialize=1.786050)
        # self.Recset = Param()

        self.alpha_T29 = Param(initialize=1)
        self.alpha_T15 = Param(initialize=1)
        self.alpha_D = Param(initialize=1)
        self.alpha_Qc = Param(initialize=1)
        self.alpha_Qr = Param(initialize=1)
        self.alpha_Rec = Param(initialize=1)

        def _alpha_init(m, i):
            if i <= 21:
                return 0.62
            else:
                return 0.35

        self.alpha = Param(self.tray,
                           initialize=lambda m, t: 0.62 if t <= 21 else 0.35)

        # --------------------------------------------------------------------------------------------------------------
        # States (differential) section
        zero_tray = dict.fromkeys(self.tray)
        zero3 = dict.fromkeys(self.fe_t * self.cp_t * self.tray)

        for key in zero3.iterkeys():
            zero3[key] = 0.0

        def __m_init(m, i, j, t):
            if t < m.Ntray:
                return 4000.
            elif t == 1:
                return 104340.
            elif t == m.Ntray:
                return 5000.

        # Liquid hold-up
        self.M = Var(self.fe_t, self.cp_t, self.tray,
                     initialize=__m_init)
        # Mole-fraction
        self.x = Var(self.fe_t, self.cp_t, self.tray, initialize=lambda m, i, j, t: 0.999 * t / m.Ntray)

        self.M_ic = zero_tray if steady else Param(self.tray, initialize=0.0, mutable=True)
        self.x_ic = zero_tray if steady else Param(self.tray, initialize=0.0, mutable=True)

        self.dM_dt = zero3 if steady else Var(self.fe_t, self.cp_t, self.tray, initialize=0.0)
        self.dx_dt = zero3 if steady else Var(self.fe_t, self.cp_t, self.tray, initialize=0.0)

        self.w_M = object
        self.w_x = object

        # --------------------------------------------------------------------------------------------------------------
        # States (algebraic) section
        # Tray temperature
        self.T = Var(self.fe_t, self.cp_ta, self.tray,
                     initialize=lambda m, i, j, t: ((370.781 - 335.753) / m.Ntray) * t + 370.781)
        self.Tdot = Var(self.fe_t, self.cp_ta, self.tray, initialize=1e-05)  #: Not really a der_var

        # saturation pressures
        self.pm = Var(self.fe_t, self.cp_ta, self.tray, initialize=1e4)
        self.pn = Var(self.fe_t, self.cp_ta, self.tray, initialize=1e4)

        # Vapor mole flowrate
        self.V = Var(self.fe_t, self.cp_ta, self.tray, initialize=44.0)

        def _l_init(m, i, j, t):
            if 2 <= t <= 21:
                return 83.
            elif 22 <= t <= 42:
                return 23
            elif t == 1:
                return 40

        # Liquid mole flowrate
        self.L = Var(self.fe_t, self.cp_ta, self.tray, initialize=_l_init)

        # Vapor mole frac & diff var
        self.y = Var(self.fe_t, self.cp_ta, self.tray,
                     initialize=lambda m, i, j, t: ((0.99 - 0.005) / m.Ntray) * t + 0.005)

        # Liquid enthalpy    # enthalpy
        self.hl = Var(self.fe_t, self.cp_ta, self.tray, initialize=10000.)

        # Liquid enthalpy    # enthalpy
        self.hv = Var(self.fe_t, self.cp_ta, self.tray, initialize=5e+04)
        # Re-boiler & condenser heat
        self.Qc = Var(self.fe_t, self.cp_ta, initialize=1.6e06)
        self.D = Var(self.fe_t, self.cp_ta, initialize=18.33)
        # vol holdups
        self.Vm = Var(self.fe_t, self.cp_ta, self.tray, initialize=6e-05)

        self.Mv = Var(self.fe_t, self.cp_ta, self.tray,
                      initialize=lambda m, i, j, t: 0.23 if 1 < t < m.Ntray else 0.0)
        self.Mv1 = Var(self.fe_t, self.cp_ta, initialize=8.57)
        self.Mvn = Var(self.fe_t, self.cp_ta, initialize=0.203)

        hi_t = dict.fromkeys(self.fe_t)
        for key in hi_t.keys():
            hi_t[key] = 1.0 if steady else _t/self.nfe_t

        self.hi_t = hi_t if steady else Param(self.fe_t, initialize=hi_t)

        # --------------------------------------------------------------------------------------------------------------
        #: Controls
        self.u1 = Param(self.fe_t, initialize=7.72700925775773761472464684629813E-01, mutable=True)  #: Dummy
        self.u2 = Param(self.fe_t, initialize=1.78604740940007800236344337463379E+06, mutable=True)  #: Dummy

        self.Rec = Var(self.fe_t, initialize=7.72700925775773761472464684629813E-01)
        self.Qr = Var(self.fe_t, initialize=1.78604740940007800236344337463379E+06)
        # --------------------------------------------------------------------------------------------------------------
        # Constraint section (differential equations)
        #: Control constraint
        self.u1_e = Expression(self.fe_t, rule=lambda m, i: self.Rec[i])
        self.u2_e = Expression(self.fe_t, rule=lambda m, i: self.Qr[i])

        self.u1_c = Constraint(self.fe_t, rule=lambda m, i: self.u1[i] == self.u1_e[i])
        self.u2_c = Constraint(self.fe_t, rule=lambda m, i: self.u2[i] == self.u2_e[i])

        #: Differential equations
        self.de_M = Constraint(self.fe_t, self.cp_ta, self.tray, rule=m_ode)
        self.de_x = Constraint(self.fe_t, self.cp_ta, self.tray, rule=x_ode)

        #: Collocation equations
        self.dvar_t_M = None if steady else Constraint(self.fe_t, self.cp_ta, self.tray, rule=M_COLL)
        self.dvar_t_x = None if steady else Constraint(self.fe_t, self.cp_ta, self.tray, rule=x_coll)

        #: Continuation equations (redundancy here)
        if self.nfe_t > 1:
            self.noisy_M = None if steady else Expression(self.fe_t, self.tray, rule=M_CONT)
            self.noisy_x = None if steady else Expression(self.fe_t, self.tray, rule=x_cont)

            self.cp_M = None if steady else \
                Constraint(self.fe_t, self.tray,
                           rule=lambda m, i, t: self.noisy_M[i, t] == 0.0 if i < self.nfe_t else Constraint.Skip)
            self.cp_x = None if steady else \
                Constraint(self.fe_t, self.tray,
                           rule=lambda m, i, t: self.noisy_x[i, t] == 0.0 if i < self.nfe_t else Constraint.Skip)

        self.M_icc = None if steady else Constraint(self.tray, rule=acm)
        self.x_icc = None if steady else Constraint(self.tray, rule=acx)

        # --------------------------------------------------------------------------------------------------------------
        # Constraint section (algebraic equations)

        self.hrc = Constraint(self.fe_t, self.cp_ta, rule=hrc)
        self.gh = Constraint(self.fe_t, self.cp_ta, self.tray, rule=gh)
        self.ghb = Constraint(self.fe_t, self.cp_ta, rule=ghb)
        self.ghc = Constraint(self.fe_t, self.cp_ta, rule=ghc)
        self.hkl = Constraint(self.fe_t, self.cp_ta, self.tray, rule=hkl)
        self.hkv = Constraint(self.fe_t, self.cp_ta, self.tray, rule=hkv)
        self.lpself = Constraint(self.fe_t, self.cp_ta, self.tray, rule=lpm)
        self.lpn = Constraint(self.fe_t, self.cp_ta, self.tray, rule=lpn)
        self.dp = Constraint(self.fe_t, self.cp_ta, self.tray, rule=dp)
        self.lTdot = Constraint(self.fe_t, self.cp_ta, self.tray, rule=lTdot)
        self.gy0 = Constraint(self.fe_t, self.cp_ta, rule=gy0)
        self.gy = Constraint(self.fe_t, self.cp_ta, self.tray, rule=gy)
        self.dMV = Constraint(self.fe_t, self.cp_ta, self.tray, rule=dMV)
        self.dMv1 = Constraint(self.fe_t, self.cp_ta, rule=dMv1)
        self.dMvn = Constraint(self.fe_t, self.cp_ta, rule=dMvn)
        self.hyd = Constraint(self.fe_t, self.cp_ta, self.tray, rule=hyd)
        self.hyd1 = Constraint(self.fe_t, self.cp_ta, rule=hyd1)
        self.hydN = Constraint(self.fe_t, self.cp_ta, rule=hydN)
        self.dvself = Constraint(self.fe_t, self.cp_ta, self.tray, rule=dvm)

        # Suffixes
        self.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

    def write_nl(self):
        """Writes the nl file and the respective row & col"""
        name = str(self.__class__.__name__) + ".nl"
        self.write(filename=name,
                   format=ProblemFormat.nl,
                   io_options={"symbolic_solver_labels": True})

    def create_bounds(self):
        """Creates bounds for the variables"""
        for value in itervalues(self.M):
            value.setlb(1.0)
        for value in itervalues(self.T):
            value.setlb(200)
        for value in itervalues(self.pm):
            value.setlb(1.0)
        for value in itervalues(self.pn):
            value.setlb(1.0)
        for value in itervalues(self.L):
            value.setlb(0.0)
        for value in itervalues(self.V):
            value.setlb(0.0)

        for value in itervalues(self.x):
            value.setlb(0.0)
        for value in itervalues(self.y):
            value.setlb(0.0)

        for value in itervalues(self.hl):
            value.setlb(1.0)
        for value in itervalues(self.hv):
            value.setlb(1.0)

        for value in itervalues(self.Qc):
            value.setlb(0.0)
        for value in itervalues(self.D):
            value.setlb(0.0)

        for value in itervalues(self.Vm):
            value.setlb(0.0)
        for value in itervalues(self.Mv):
            value.setlb(0.155 + 1e-06)
        for value in itervalues(self.Mv1):
            value.setlb(8.5 + 1e-06)
        for value in itervalues(self.Mvn):
            value.setlb(0.17 + 1e-06)

        for value in itervalues(self.M):
            value.setub(1e+07)
        for value in itervalues(self.T):
            value.setub(500)
        for value in itervalues(self.pm):
            value.setub(5e+07)
        for value in itervalues(self.pn):
            value.setub(5e+07)
        for value in itervalues(self.L):
            value.setub(1e+03)
        for value in itervalues(self.V):
            value.setub(1e+03)

        for value in itervalues(self.x):
            value.setub(1.0)
        for value in itervalues(self.y):
            value.setub(1.0)

        for value in itervalues(self.hl):
            value.setub(1e+07)
        for value in itervalues(self.hv):
            value.setub(1e+07)

        for value in itervalues(self.Qc):
            value.setub(1e+08)
        for value in itervalues(self.D):
            value.setub(1e+04)

        for value in itervalues(self.Vm):
            value.setub(1e+04)
        for value in itervalues(self.Mv):
            value.setub(1e+04)
        for value in itervalues(self.Mv1):
            value.setub(1e+04)
        for value in itervalues(self.Mvn):
            value.setub(1e+04)

    @staticmethod
    def parse_ig_ampl(file_i):
        lines = file_i.readlines()
        dict = {}
        for line in lines:
            kk = re.split('(?:let)|[:=\s\[\]]', line)
            try:
                var = kk[2]
                print(var)
                key = kk[3]
                key = re.split(',', key)
                actual_key = []
                for k in key:
                    actual_key.append(int(k))
                actual_key.append(actual_key.pop(0))
                actual_key = tuple(actual_key)

                value = kk[8]
                value = float(value)
                dict[var, actual_key] = value
            except IndexError:
                continue
        file_i.close()
        return dict

    def init_steady_ref(self):
        """If the model is steady, we try to initialize it with an initial guess from ampl"""
        cur_dir = os.path.dirname(__file__)
        ampl_ig = os.path.join(cur_dir, "iv_ss.txt")
        file_tst = open(ampl_ig, "r")
        if self.nfe_t == 1 and self.ncp_t == 1:
            somedict = self.parse_ig_ampl(file_tst)
            for var in self.component_objects(Var, active=True):
                vx = getattr(self, str(var))
                for v, k in var.iteritems():
                    try:
                        vx[v] = somedict[str(var), v]
                    except KeyError:
                        continue
            solver = SolverFactory('ipopt')
            someresults = solver.solve(self, tee=True)

    def equalize_u(self, direction="u_to_r"):
        """set current controls to the values of their respective dummies"""
        if direction == "u_to_r":
            for i in iterkeys(self.Rec):
                self.Rec[i].set_value(value(self.u1[i]))
            for i in iterkeys(self.Rec):
                self.Qr[i].set_value(value(self.u2[i]))
        elif direction == "r_to_u":
            for i in iterkeys(self.u1):
                self.u1[i].value = value(self.Rec[i])
            for i in iterkeys(self.u2):
                self.u2[i].value = value(self.Qr[i])