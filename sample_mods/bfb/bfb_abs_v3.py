#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.core.base import Param, ConcreteModel, Var, Constraint, Set, exp, sqrt, Suffix, value
from pyomo.opt import ProblemFormat
from nmpc_mhe.aux.cpoinsc import collptsgen
from nmpc_mhe.aux.lagrange_f import lgr, lgry, lgrdot, lgrydot
from bfb_abs_cons_v3 import *
from initial_s import ss
import os

"""
BFB
"""
__all__ = ["bfb_dae"]

__author__ = 'David M Thierry'

class bfb_dae(ConcreteModel):
    def __init__(self, nfe_t, ncp_t, **kwargs):
        ConcreteModel.__init__(self)
        steady = kwargs.pop('steady', False)
        _t = kwargs.pop('_t', 1.0)

        nfe_x = kwargs.pop('nfe_x', 5)
        ncp_x = kwargs.pop('ncp_x', 3)

        self._L = 5.
        self._alp_gauB_x = 0
        self._bet_gauB_x = 0

        self._alp_gauB_t = 1
        self._bet_gauB_t = 0
        self.nfe_x = nfe_x
        self.ncp_x = ncp_x

        if steady:
            print("[I] " + str(self.__class__.__name__) + " NFE and NCP Overriden - Steady state mode")
            self.nfe_t = 1
            self.ncp_t = 1
        else:
            self.nfe_t = nfe_t
            self.ncp_t = ncp_t

        self.tau_x = collptsgen(self.ncp_x, self._alp_gauB_x, self._bet_gauB_x)
        self.tau_t = collptsgen(self.ncp_t, self._alp_gauB_t, self._bet_gauB_t)

        # start at zero
        self.tau_i_x = {0: 0.}
        self.tau_i_t = {0: 0.}
        # create a list

        for ii in range(1, ncp_x + 1):
            self.tau_i_x[ii] = self.tau_x[ii - 1]

        for ii in range(1, self.ncp_t + 1):
            self.tau_i_t[ii] = self.tau_t[ii - 1]

        # ======= SETS ======= #
        # For finite element = 1 .. K
        # This has to be > 0

        self.fe_t = Set(initialize=[ii for ii in range(1, self.nfe_t + 1)])
        self.fe_x = Set(initialize=[ii for ii in range(1, self.nfe_x + 1)])

        # collocation points
        # collocation points for diferential variables
        self.cp_x = Set(initialize=[ii for ii in range(0, self.ncp_x + 1)])
        self.cp_t = Set(initialize=[ii for ii in range(0, self.ncp_t + 1)])

        self.cp_xa = Set(within=self.cp_x, initialize=[ii for ii in range(1, self.ncp_x + 1)])
        self.cp_ta = Set(within=self.cp_t, initialize=[ii for ii in range(1, self.ncp_t + 1)])

        # components
        self.sp = Set(initialize=['c', 'h', 'n'])

        # create collocation param
        self.taucp_x = Param(self.cp_x, initialize=self.tau_i_x)
        self.taucp_t = Param(self.cp_t, initialize=self.tau_i_t)

        self.ldot_x = Param(self.cp_x, self.cp_x, initialize=
        (lambda m, j, k: lgrdot(j, m.taucp_x[k], ncp_x, self._alp_gauB_x, self._bet_gauB_x)))
        # (lambda m, i, j: fldoti_x(m, i, j, ncp_x, self._alp_gauB_x, self._bet_gauB_x)))
        self.ldot_t = Param(self.cp_t, self.cp_t, initialize=
        (lambda m, i, j: fldoti_t(m, i, j, self.ncp_t, self._alp_gauB_t, self._bet_gauB_t)))
        self.lydot = Param(self.cp_x, self.cp_x, initialize=
        (lambda m, i, j: fldotyi(m, i, j, ncp_x, self._alp_gauB_x, self._bet_gauB_x)))
        self.l1_x = Param(self.cp_x, initialize=
        (lambda m, i: flj1_x(m, i, ncp_x, self._alp_gauB_x, self._bet_gauB_x)))
        self.l1_t = Param(self.cp_t, initialize=
        (lambda m, i: flj1_t(m, i, self.ncp_t, self._alp_gauB_t, self._bet_gauB_t)))
        self.l1y = Param(self.cp_x, initialize=
        (lambda m, i: fljy1(m, i, ncp_x, self._alp_gauB_x, self._bet_gauB_x)))
        self.lgr = Param(self.cp_x, self.cp_x, initialize=
        (lambda m, i, j: f_lj_x(m, i, j, ncp_x, self._alp_gauB_x, self._bet_gauB_x)))

        self.A1 = Param(initialize=56200.0)
        self.A2 = Param(initialize=2.62)
        self.A3 = Param(initialize=98.9)
        self.ah = Param(initialize=0.8)
        self.Ao = Param(initialize=4e-4)
        self.ap = Param(initialize=90.49773755656109)
        self.Ax = Param(initialize=63.32160110335595)
        self.cpg_mol = Param(initialize=30.0912)

        _cpgcsb = {'c': 38.4916, 'h': 33.6967}
        _cpgcst = {'c': 38.4913, 'h': 33.6968}
        _cpgcsc = {'c': 39.4617, 'h': 29.1616}
        _cpgcgc = {'c': 39.2805, 'h': 33.7988, 'n': 29.1578}
        _cpgcge = {'c': 39.3473, 'h': 33.8074, 'n': 29.1592}
        _cpgcse = {'c': 39.3486, 'h': 33.8076}

        self.cpgcsb = Param(self.sp, initialize=_cpgcsb)
        self.cpgcst = Param(self.sp, initialize=_cpgcst)
        self.cpgcgc = Param(self.sp, initialize=_cpgcgc)
        self.cpgcge = Param(self.sp, initialize=_cpgcge)
        self.cpgcsc = Param(self.sp, initialize=_cpgcsc)
        self.cpgcse = Param(self.sp, initialize=_cpgcse)
        self.cps = Param(initialize=1.13)
        self.Cr = Param(initialize=1.0)
        self.dH1 = Param(initialize=-52100)
        self.dH2 = Param(initialize=-36300)
        self.dH3 = Param(initialize=-64700, mutable=True)
        self.dp = Param(initialize=1.5e-4)
        self.dPhx = Param(initialize=0.01)
        self.dS1 = Param(initialize=-78.5)
        self.dS2 = Param(initialize=-88.1)
        self.dS3 = Param(initialize=-175)
        self.Dt = Param(initialize=9.0)
        self.Dte = Param(initialize=2.897869210295575)
        self.dx = Param(initialize=0.02)
        self.E1 = Param(initialize=28200)
        self.E2 = Param(initialize=58200)
        self.E3 = Param(initialize=57700)
        self.emf = Param(initialize=0.5)
        self.fw = Param(initialize=0.2)
        self.hw = Param(initialize=1.5)
        self.K_d = Param(initialize=100)
        self.kg = Param(initialize=0.0280596)
        self.kp = Param(initialize=1.36)
        self.Lb = Param(initialize=5.0)
        self.m1 = Param(initialize=1.17)
        self.mug = Param(initialize=1.92403e-5)
        self.nv = Param(initialize=2350.0)
        self.Nx = Param(initialize=941.0835981537471)
        self.phis = Param(initialize=1.0)
        self.Pr = Param(initialize=0.7385161078322956)
        self.rhohx = Param(initialize=985.9393497324021)
        self.rhos = Param(initialize=442.0)

        self.pi = Param(initialize=3.14)
        self.R = Param(initialize=8.314472)
        self.gc = Param(initialize=9.81)

        # heat exchanger input condition
        self.HXIn_P = Param(initialize=1.12, mutable=True)
        self.HXIn_T = Param(initialize=33, mutable=True)
        self.HXIn_F = Param(initialize=60000, mutable=True)

        # Gas input condition
        self.GasIn_T = Param(self.fe_t, initialize=40, mutable=True)  # needs i_value
        self._GasIn_z = {'c': 0.13, 'h': 0.06, 'n': 0.81}
        self._GasIn_z = {}
        for i in range(1, self.nfe_t + 1):
            self._GasIn_z[i, 'c'] = 0.13
            self._GasIn_z[i, 'h'] = 0.06
            self._GasIn_z[i, 'n'] = 0.81
        # self.GasIn_z = Param(self.sp, initialize=_GasIn_z)
        self.GasIn_z = Param(self.fe_t, self.sp, initialize=self._GasIn_z, mutable=True)

        self.flue_gas_P = Param(initialize=1.8)

        # Solid input condition
        _nin = {'c': 0.01, 'h': 0.7, 'n': 0.7}
        self.nin = Param(self.sp, initialize=_nin, mutable=True)
        # mol/kg
        self.SolidIn_T = Param(initialize=50, mutable=True)
        self.sorbent_P = Param(initialize=1.5)

        # atmosphere pressure
        self.Out2_P = Param(initialize=1.0)
        # input gas valve
        self.CV_1 = Param(initialize=5.696665718420114)
        # output gas valve
        self.CV_2 = Param(initialize=12.97878700936543)
        # input solid valve
        self.CV_3 = Param(initialize=17483.58063173724)
        # output solid valve
        self.CV_4 = Param(initialize=11187.66019532553)

        # self.per_opening3 = Param(self.fe_t, initialize=50., mutable=True)
        self.per_opening4 = Param(initialize=50)

        self.eavg = Param(initialize=0.591951)


        self.llast = Param(initialize=5.)
        self.lenleft = Param(initialize=5.)
        self.hi_x = Param(self.fe_x, initialize=fir_hi)
        hi_t = dict.fromkeys(self.fe_t)
        for key in hi_t.keys():
            hi_t[key] = 1.0 if steady else _t/self.nfe_t

        self.hi_t = hi_t if steady else Param(self.fe_t, initialize=hi_t)
        self.l = Param(self.fe_x, self.cp_x, initialize=fl_irule)

        # --------------------------------------------------------------------------------------------------------------
        #: First define differential state variables (state: x, ic-Param: x_ic, derivative-Var:dx_dt
        #: States (differential) section

        zero2_x = dict.fromkeys(self.fe_x * self.cp_x)
        zero3_x = dict.fromkeys(self.fe_x * self.cp_x * self.sp)
        zero4 = dict.fromkeys(self.fe_t * self.cp_ta * self.fe_x * self.cp_x)
        zero5 = dict.fromkeys(self.fe_t * self.cp_ta * self.fe_x * self.cp_x * self.sp)

        for key in zero2_x.keys():
            zero2_x[key] = 0.0
        for key in zero3_x.keys():
            zero3_x[key] = 0.0
        for key in zero4.keys():
            zero4[key] = 0.0
        for key in zero5.keys():
            zero5[key] = 0.0

        #:  State-variables
        self.Ngb = Var(self.fe_t, self.cp_t, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.Hgb = Var(self.fe_t, self.cp_t, self.fe_x, self.cp_x, initialize=1.)
        self.Ngc = Var(self.fe_t, self.cp_t, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.Hgc = Var(self.fe_t, self.cp_t, self.fe_x, self.cp_x, initialize=1.)
        self.Nsc = Var(self.fe_t, self.cp_t, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.Hsc = Var(self.fe_t, self.cp_t, self.fe_x, self.cp_x, initialize=1.)
        self.Nge = Var(self.fe_t, self.cp_t, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.Hge = Var(self.fe_t, self.cp_t, self.fe_x, self.cp_x, initialize=1.)
        self.Nse = Var(self.fe_t, self.cp_t, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.Hse = Var(self.fe_t, self.cp_t, self.fe_x, self.cp_x, initialize=1.)

        #:  Initial state-Param
        self.Ngb_ic = zero3_x if steady else Param(self.fe_x, self.cp_x, self.sp, initialize=1., mutable=True)
        self.Hgb_ic = zero2_x if steady else Param(self.fe_x, self.cp_x, initialize=1., mutable=True)
        self.Ngc_ic = zero3_x if steady else Param(self.fe_x, self.cp_x, self.sp, initialize=1., mutable=True)
        self.Hgc_ic = zero2_x if steady else Param(self.fe_x, self.cp_x, initialize=1., mutable=True)
        self.Nsc_ic = zero3_x if steady else Param(self.fe_x, self.cp_x, self.sp, initialize=1., mutable=True)
        self.Hsc_ic = zero2_x if steady else Param(self.fe_x, self.cp_x, initialize=1., mutable=True)
        self.Nge_ic = zero3_x if steady else Param(self.fe_x, self.cp_x, self.sp, initialize=1., mutable=True)
        self.Hge_ic = zero2_x if steady else Param(self.fe_x, self.cp_x, initialize=1., mutable=True)
        self.Nse_ic = zero3_x if steady else Param(self.fe_x, self.cp_x, self.sp, initialize=1., mutable=True)
        self.Hse_ic = zero2_x if steady else Param(self.fe_x, self.cp_x, initialize=1., mutable=True)

        #:  Derivative-var
        self.dNgb_dt = zero5 if steady else Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=0.0001)
        self.dHgb_dt = zero4 if steady else Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=0.0001)
        self.dNgc_dt = zero5 if steady else Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=0.0001)
        self.dHgc_dt = zero4 if steady else Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=0.0001)
        self.dNsc_dt = zero5 if steady else Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=0.0001)
        self.dHsc_dt = zero4 if steady else Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=0.0001)
        self.dNge_dt = zero5 if steady else Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=0.0001)
        self.dHge_dt = zero4 if steady else Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=0.0001)
        self.dNse_dt = zero5 if steady else Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=0.0001)
        self.dHse_dt = zero4 if steady else Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=0.0001)

        # --------------------------------------------------------------------------------------------------------------

        # --------------------------------------------------------------------------------------------------------------
        #: Algebraic variables
        self.HXIn_h = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.GasIn_P = Var(self.fe_t, self.cp_ta, initialize=1.)
        # self.GasIn_F = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.GasOut_P = Var(self.fe_t, self.cp_ta, initialize=1.)
        # self.GasOut_F = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.GasOut_T = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.GasOut_z = Var(self.fe_t, self.cp_ta, self.sp, initialize=1.)
        self.SolidIn_Fm = Param(self.fe_t, initialize=583860.584859)
        # self.SolidOut_Fm = Var(initialize=ic.SolidOut_Fm)
        self.SolidIn_P = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.SolidOut_P = Var(self.fe_t, self.cp_ta, initialize=1.)
        # self.SolidOut_T = Var(initialize=ic.SolidOut_T)
        # self.SorbOut_F = Var(initialize=ic.SorbOut_F)
        self.rhog_in = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.rhog_out = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.DownOut_P = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.h_downcomer = Var(self.fe_t, self.cp_ta, initialize=1.)
        # self.hsinb = Var(initialize=ic.hsinb)
        self.hsint = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.vmf = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.db0 = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.Sit = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.Sot = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.g1 = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.Ar = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.cbt = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.cct = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.cet = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.cb = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.cbin = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.cc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.ccwin = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.ce = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.cein = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.D = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.db = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.dbe = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.dbm = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.dbu = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.delta = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.dThx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.e = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.ebin = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.ecwin = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.ed = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.eein = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.fb = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.fc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.fcw = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.fn = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.g2 = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.g3 = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Gb = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Hbc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Hce = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.hd = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Hgbulk = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.hl = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.hp = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Hsbulk = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.hsc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.hse = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.ht = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.hxh = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Jc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Je = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.k1c = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.k1e = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.k2c = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.k2e = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.k3c = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.k3e = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Kbc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.Kce = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.Kcebs = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Ke1c = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Ke1e = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Ke2c = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Ke2e = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Ke3c = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Ke3e = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Kgbulk = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.kpa = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Ksbulk = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.nc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.ne = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.Nup = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.P = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Phx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.r1c = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.r1e = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.r2c = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.r2e = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.r3c = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.r3e = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Red = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.rgc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.rge = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.rhog = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.rsc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.rse = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.tau = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Tgb = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Tgc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Tge = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Thx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Tsc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Tse = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.Ttube = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.vb = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.vbr = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.ve = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.vg = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.yb = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.yc = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.ye = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)

        self.dhxh_dx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.dcbin_dx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.dcein_dx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.debin_dx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.decwin_dx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.deein_dx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.dP_dx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.dPhx_dx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)
        self.dccwin_dx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, initialize=1.)
        self.dz_dx = Var(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, initialize=1.)

        self.cbin_l = Var(self.fe_t, self.cp_ta, self.sp, initialize=1.)
        self.cein_l = Var(self.fe_t, self.cp_ta, self.sp, initialize=1.)
        self.ebin_l = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.ecwin_l = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.eein_l = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.hxh_l = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.P_l = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.Phx_l = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.ccwin_l = Var(self.fe_t, self.cp_ta, self.sp, initialize=1.)
        self.hse_l = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.ne_l = Var(self.fe_t, self.cp_ta, self.sp, initialize=1.)
        self.Gb_l = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.Tgb_l = Var(self.fe_t, self.cp_ta, initialize=1.)
        self.yb_l = Var(self.fe_t, self.cp_ta, self.sp, initialize=1.)
        self.c_capture = Var(self.fe_t, self.cp_ta, initialize=1.)
        # --------------------------------------------------------------------------------------------------------------
        #: Controls
        self.u1 = Param(self.fe_t, initialize=9937.98446662, mutable=True)
        self.u2 = Param(self.fe_t, initialize=9286.03346463, mutable=True)

        self.GasIn_F = Var(self.fe_t, initialize=9937.98446662)
        self.GasOut_F = Var(self.fe_t, initialize=9286.03346463)
        # self.per_opening1 = Var(self.fe_t, initialize=85.)
        # self.per_opening2 = Var(self.fe_t, initialize=50.)

        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        #: Constraints for the differential states
        #: Then the ode-Con:de_x, collocation-Con:dvar_t_x, noisy-Expr: noisy_x, cp-Constraint: cp_x, initial-Con: x_icc
        #: Differential equations

        #: Differential equations
        self.de_ngb = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=de_ngb_rule)
        self.de_hgb = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=de_hgb_rule)
        self.de_ngc = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=de_ngc_rule)
        self.de_hgc = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=de_hgc_rule)
        self.de_nsc = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=de_nsc_rule)
        self.de_hsc = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=de_hsc_rule)
        self.de_nge = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=de_nge_rule)
        self.de_hge = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=de_hge_rule)
        self.de_nse = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=de_nse_rule)
        self.de_hse = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=de_hse_rule)


        #: Collocation equations
        self.dvar_t_Ngb = None if steady else Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp,
                                                         rule=fdvar_t_ngb)
        self.dvar_t_Hgb = None if steady else Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_t_hgb)
        self.dvar_t_Ngc = None if steady else Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp,
                                                         rule=fdvar_t_ngc)
        self.dvar_t_Hgc = None if steady else Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_t_hgc)
        self.dvar_t_Nsc = None if steady else Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp,
                                                         rule=fdvar_t_nsc)
        self.dvar_t_Hsc = None if steady else Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_t_hsc)
        self.dvar_t_Nge = None if steady else Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp,
                                                         rule=fdvar_t_nge)
        self.dvar_t_Hge = None if steady else Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_t_hge)
        self.dvar_t_Nse = None if steady else Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp,
                                                         rule=fdvar_t_nse)
        self.dvar_t_Hse = None if steady else Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_t_hse)

        #: Continuation equations (redundancy here)
        if self.nfe_t > 1:
            #: Noisy expressions
            self.noisy_Ngb = None if steady else Expression(self.fe_t, self.fe_x, self.cp_x, self.sp, rule=fcp_t_ngb)
            self.noisy_Hgb = None if steady else Expression(self.fe_t, self.fe_x, self.cp_x, rule=fcp_t_hgb)
            self.noisy_Ngc = None if steady else Expression(self.fe_t, self.fe_x, self.cp_x, self.sp, rule=fcp_t_ngc)
            self.noisy_Hgc = None if steady else Expression(self.fe_t, self.fe_x, self.cp_x, rule=fcp_t_hgc)
            self.noisy_Nsc = None if steady else Expression(self.fe_t, self.fe_x, self.cp_x, self.sp, rule=fcp_t_nsc)
            self.noisy_Hsc = None if steady else Expression(self.fe_t, self.fe_x, self.cp_x, rule=fcp_t_hsc)
            self.noisy_Nge = None if steady else Expression(self.fe_t, self.fe_x, self.cp_x, self.sp, rule=fcp_t_nge)
            self.noisy_Hge = None if steady else Expression(self.fe_t, self.fe_x, self.cp_x, rule=fcp_t_hge)
            self.noisy_Nse = None if steady else Expression(self.fe_t, self.fe_x, self.cp_x, self.sp, rule=fcp_t_nse)
            self.noisy_Hse = None if steady else Expression(self.fe_t, self.fe_x, self.cp_x, rule=fcp_t_hse)



            self.cp_Ngb = None if steady else Constraint(self.fe_t, self.fe_x, self.cp_x, self.sp,
                                                         rule=lambda m, i, ix, jx, c:
                                                         m.noisy_Ngb[i, ix, jx, c] == 0.0
                                                         if i < m.nfe_t and 0 < jx <= m.ncp_x else Constraint.Skip)
            self.cp_Hgb = None if steady else Constraint(self.fe_t, self.fe_x, self.cp_x,
                                                         rule=lambda m, i, ix, jx:
                                                         m.noisy_Hgb[i, ix, jx] == 0.0
                                                         if i < m.nfe_t and 0 < jx <= m.ncp_x else Constraint.Skip)
            self.cp_Ngc = None if steady else Constraint(self.fe_t, self.fe_x, self.cp_x, self.sp,
                                                         rule=lambda m, i, ix, jx, c:
                                                         m.noisy_Ngc[i, ix, jx, c] == 0.0
                                                         if i < m.nfe_t and 0 < jx <= m.ncp_x else Constraint.Skip)
            self.cp_Hgc = None if steady else Constraint(self.fe_t, self.fe_x, self.cp_x,
                                                         rule=lambda m, i, ix, jx:
                                                         m.noisy_Hgc[i, ix, jx] == 0.0
                                                         if i < m.nfe_t and 0 < jx <= m.ncp_x else Constraint.Skip)
            self.cp_Nsc = None if steady else Constraint(self.fe_t, self.fe_x, self.cp_x, self.sp,
                                                         rule=lambda m, i, ix, jx, c:
                                                         m.noisy_Nsc[i, ix, jx, c] == 0.0
                                                         if i < m.nfe_t and 0 < jx <= m.ncp_x else Constraint.Skip)
            self.cp_Hsc = None if steady else Constraint(self.fe_t, self.fe_x, self.cp_x,
                                                         rule=lambda m, i, ix, jx:
                                                         m.noisy_Hsc[i, ix, jx] == 0.0
                                                         if i < m.nfe_t and 0 < jx <= m.ncp_x else Constraint.Skip)
            self.cp_Nge = None if steady else Constraint(self.fe_t, self.fe_x, self.cp_x, self.sp,
                                                         rule=lambda m, i, ix, jx, c:
                                                         m.noisy_Nge[i, ix, jx, c] == 0.0
                                                         if i < m.nfe_t and 0 < jx <= m.ncp_x else Constraint.Skip)
            self.cp_Hge = None if steady else Constraint(self.fe_t, self.fe_x, self.cp_x,
                                                         rule=lambda m, i, ix, jx:
                                                         m.noisy_Hge[i, ix, jx] == 0.0
                                                         if i < m.nfe_t and 0 < jx <= m.ncp_x else Constraint.Skip)
            self.cp_Nse = None if steady else Constraint(self.fe_t, self.fe_x, self.cp_x, self.sp,
                                                         rule=lambda m, i, ix, jx, c:
                                                         m.noisy_Nse[i, ix, jx, c] == 0.0
                                                         if i < m.nfe_t and 0 < jx <= m.ncp_x else Constraint.Skip)
            self.cp_Hse = None if steady else Constraint(self.fe_t, self.fe_x, self.cp_x,
                                                         rule=lambda m, i, ix, jx:
                                                         m.noisy_Hse[i, ix, jx] == 0.0
                                                         if i < m.nfe_t and 0 < jx <= m.ncp_x else Constraint.Skip)


        #: Initial condition-Constraints
        self.Ngb_icc = None if steady else Constraint(self.fe_x, self.cp_x, self.sp, rule=ic_ngb_rule)
        self.Hgb_icc = None if steady else Constraint(self.fe_x, self.cp_x, rule=ic_hgb_rule)
        self.Ngc_icc = None if steady else Constraint(self.fe_x, self.cp_x, self.sp, rule=ic_ngc_rule)
        self.Hgc_icc = None if steady else Constraint(self.fe_x, self.cp_x, rule=ic_hgc_rule)
        self.Nsc_icc = None if steady else Constraint(self.fe_x, self.cp_x, self.sp, rule=ic_nsc_rule)
        self.Hsc_icc = None if steady else Constraint(self.fe_x, self.cp_x, rule=ic_hsc_rule)
        self.Nge_icc = None if steady else Constraint(self.fe_x, self.cp_x, self.sp, rule=ic_nge_rule)
        self.Hge_icc = None if steady else Constraint(self.fe_x, self.cp_x, rule=ic_hge_rule)
        self.Nse_icc = None if steady else Constraint(self.fe_x, self.cp_x, self.sp, rule=ic_nse_rule)
        self.Hse_icc = None if steady else Constraint(self.fe_x, self.cp_x, rule=ic_hse_rule)


        #: Algebraic definitions
        self.ae_ngb = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=ngb_rule)
        self.ae_hgb = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=hgb_rule)
        self.ae_ngc = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=ngc_rule)
        self.ae_hgc = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=hgc_rule)
        self.ae_nsc = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=nsc_rule)
        self.ae_hsc = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=hsc_rule)
        self.ae_nge = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=nge_rule)
        self.ae_hge = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=hge_rule)
        self.ae_nse = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=nse_rule)
        self.ae_hse = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=hse_rule)


        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        #: Algebraic constraints
        self.a1 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a1_rule)
        self.a3 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a3_rule)
        self.a4 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a4_rule)
        self.a5 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a5_rule)
        self.a7 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=a7_rule)
        self.a8 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=a8_rule)
        self.a9 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=a9_rule)
        self.a11_2 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a11_rule_2)
        self.a12 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a12_rule)
        self.a13 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a13_rule)
        self.a14 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a14_rule)
        self.a15 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=a15_rule)
        self.a16 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=a16_rule)
        self.a17 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=a17_rule)
        self.a18 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a18_rule)
        self.a19 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a19_rule)
        self.a20 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a20_rule)
        self.a21 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a21_rule)
        self.a22 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a22_rule)
        self.a23 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a23_rule)
        self.a24 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a24_rule)
        self.a25 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a25_rule)
        self.a26 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a26_rule)
        self.a27 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a27_rule)
        self.a28 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a28_rule)
        self.a29 = Constraint(self.fe_t, self.cp_ta, rule=a29_rule)
        self.a30 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a30_rule)
        self.a31 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a31_rule)
        self.a32 = Constraint(self.fe_t, self.cp_ta, rule=a32_rule)
        self.a33 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a33_rule)
        self.a34 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a34_rule)
        self.a35 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a35_rule)
        self.a36 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a36_rule)
        self.a37 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a37_rule)
        self.a38 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=a38_rule)
        self.a39 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=a39_rule)
        self.a40 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a40_rule)
        self.a41 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a41_rule)
        self.a42 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a42_rule)
        self.a43 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a43_rule)
        self.a44 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a44_rule)
        self.a45 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a45_rule)
        self.a46 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a46_rule)
        self.a47 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a47_rule)
        self.a48 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a48_rule)
        self.a49 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a49_rule)
        self.a50 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a50_rule)
        self.a51 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a51_rule)
        self.a52 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a52_rule)
        self.a53 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a53_rule)
        self.a54 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a54_rule)
        self.a55 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a55_rule)
        self.a56 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a56_rule)
        self.a57 = Constraint(self.fe_t, self.cp_ta, rule=a57_rule)
        self.a58 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a58_rule)
        self.a59 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a59_rule)
        self.a60 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a60_rule)
        self.a61 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a61_rule)
        self.a62 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a62_rule)
        self.a63 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a63_rule)
        self.a64 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a64_rule)
        self.a65 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a65_rule)
        self.a66 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a66_rule)
        self.a67 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a67_rule)
        self.a68 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a68_rule)
        self.a69 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a69_rule)
        self.a70 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a70_rule)
        self.a71 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a71_rule)
        self.a72 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a72_rule)
        self.a73 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a73_rule)
        self.a74 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a74_rule)
        self.a75 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a75_rule)
        self.a76 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a76_rule)
        self.a77 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a77_rule)
        self.a78 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a78_rule)
        self.a79 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a79_rule)
        self.a80 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a80_rule)
        self.a81 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a81_rule)
        self.a82 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a82_rule)
        self.a83 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a83_rule)
        self.a84 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a84_rule)
        self.a85 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a85_rule)
        self.a86 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a86_rule)
        self.a87 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a87_rule)
        self.a88 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a88_rule)
        self.a89 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=a89_rule)

        self.i1 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=i1_rule)
        self.i2 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=i2_rule)
        self.i3 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=i3_rule)
        self.i4 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=i4_rule)
        self.i5 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=i5_rule)
        self.i6 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=i6_rule)
        self.i7 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=i7_rule)
        self.i8 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=i8_rule)
        self.e1 = Constraint(self.fe_t, self.cp_ta, rule=e1_rule)
        self.e2 = Constraint(self.fe_t, self.cp_ta, rule=e2_rule)
        self.e3 = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=e3_rule)
        self.e4 = Constraint(self.fe_t, self.cp_ta, rule=e4_rule)
        self.e5 = Constraint(self.fe_t, self.cp_ta, rule=e5_rule)
        self.e7 = Constraint(self.fe_t, self.cp_ta, rule=e7_rule)
        self.e8 = Constraint(self.fe_t, self.cp_ta, rule=e8_rule)
        self.e9 = Constraint(self.fe_t, self.cp_ta, rule=e9_rule)
        self.e10 = Constraint(self.fe_t, self.cp_ta, self.sp, rule=e10_rule)
        self.x_3 = Constraint(self.fe_t, self.cp_ta, rule=x_3_rule)
        self.e12 = Constraint(self.fe_t, self.cp_ta, rule=e12_rule)
        self.e13 = Constraint(self.fe_t, self.cp_ta, rule=e13_rule)
        self.e14 = Constraint(self.fe_t, self.cp_ta, self.sp, rule=e14_rule)
        self.e15 = Constraint(self.fe_t, self.cp_ta, rule=e15_rule)
        self.e16 = Constraint(self.fe_t, self.cp_ta, rule=e16_rule)
        self.e20 = Constraint(self.fe_t, self.cp_ta, rule=e20_rule)  #: BC

        self.e25 = Constraint(self.fe_t, self.cp_ta, self.sp, rule=e25_rule)  #: BC
        self.e26 = Constraint(self.fe_t, self.cp_ta, rule=e26_rule)
        # self.v1 = Constraint(self.fe_t, self.cp_ta, rule=v1_rule)
        self.v2 = Constraint(self.fe_t, self.cp_ta, rule=v2_rule)
        # self.v4 = Constraint(self.fe_t, self.cp_ta, rule=v4_rule)
        self.v5 = Constraint(self.fe_t, self.cp_ta, rule=v5_rule)
        # self.v3 = Constraint(self.fe_t, self.cp_ta, rule=v3_rule)

        #: BVP equations
        self.de_x_cbin = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=fdvar_x_cbin_)
        self.de_x_cein = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=fdvar_x_cein_)
        self.de_x_ebin = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_x_ebin_)
        self.de_x_ecwin = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_x_ecwin_)
        self.de_x_eein = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_x_eein_)
        self.de_x_hxh = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_x_hxh_)
        self.de_x_p = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_x_p_)
        self.de_x_phx = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, rule=fdvar_x_phx_)
        self.de_x_ccwin = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.cp_x, self.sp, rule=fdvar_x_ccwin_)


        self.cp1_c = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.sp, rule=fcp_x_cbin)
        self.cp2_c = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.sp, rule=fcp_x_cein)
        self.cp3_c = Constraint(self.fe_t, self.cp_ta, self.fe_x, rule=fcp_x_ebin)
        self.cp4_c = Constraint(self.fe_t, self.cp_ta, self.fe_x, rule=fcp_x_ecwin)
        self.cp5_c = Constraint(self.fe_t, self.cp_ta, self.fe_x, rule=fcp_x_eein)
        self.cp6_c = Constraint(self.fe_t, self.cp_ta, self.fe_x, rule=fcp_x_hxh)
        # self.cp7_c = Constraint(self.fe_x, rule=fcp1_7)
        self.cp8_c = Constraint(self.fe_t, self.cp_ta, self.fe_x, rule=fcp_x_p)
        self.cp9_c = Constraint(self.fe_t, self.cp_ta, self.fe_x, rule=fcp_x_phx)
        self.cp10_c = Constraint(self.fe_t, self.cp_ta, self.fe_x, self.sp, rule=fcp_x_ccwin)


        #: Last cp point equation (differential-bvp)
        self.zl_1 = Constraint(self.fe_t, self.cp_ta, self.sp, rule=fzl_1)
        self.zl_2 = Constraint(self.fe_t, self.cp_ta, self.sp, rule=fzl_2)
        self.zl_3 = Constraint(self.fe_t, self.cp_ta, rule=fzl_3)
        self.zl_4 = Constraint(self.fe_t, self.cp_ta, rule=fzl_4)
        self.zl_5 = Constraint(self.fe_t, self.cp_ta, rule=fzl_5)
        self.zl_6 = Constraint(self.fe_t, self.cp_ta, rule=fzl_6)  #
        self.zl_8 = Constraint(self.fe_t, self.cp_ta, rule=fzl_8)
        self.zl_9 = Constraint(self.fe_t, self.cp_ta, rule=fzl_9)  #
        self.zl_10 = Constraint(self.fe_t, self.cp_ta, self.sp, rule=fzl_10)


        #: Last cp point equation (albegraic-bvp)
        self.yl_hse = Constraint(self.fe_t, self.cp_ta, rule=fyl_hse)
        self.yl_ne = Constraint(self.fe_t, self.cp_ta, self.sp, rule=fyl_ne)
        # self.yl_7 = Constraint(rule=fyl_7)
        # self.yl_11 = Constraint(rule=fyl_11)
        self.yl_gb = Constraint(self.fe_t, self.cp_ta, rule=fyl_gb)
        self.yl_tgb = Constraint(self.fe_t, self.cp_ta, rule=fyl_tgb)
        self.yl_yb = Constraint(self.fe_t, self.cp_ta, self.sp, rule=fyl_yb)

        #: BC at the bottom
        self.jn_bc1 = Constraint(self.fe_t, self.cp_ta, self.sp, rule=ic_jn)
        self.jh_bc1 = Constraint(self.fe_t, self.cp_ta, rule=ic_jh)


        self.cc_def = Constraint(self.fe_t, self.cp_ta, rule=cc_rule)

        # --------------------------------------------------------------------------------------------------------------
        #: Control constraint
        self.u1_e = Expression(self.fe_t, rule=lambda m, i: self.GasIn_F[i])
        self.u2_e = Expression(self.fe_t, rule=lambda m, i: self.GasOut_F[i])

        self.u1_c = Constraint(self.fe_t, rule=lambda m, i: self.u1[i] == self.u1_e[i])
        self.u2_c = Constraint(self.fe_t, rule=lambda m, i: self.u2[i] == self.u2_e[i])
        # --------------------------------------------------------------------------------------------------------------
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
        print('problem with bounds activated')
        for it in range(1, self.nfe_t + 1):
            for jt in range(1, self.ncp_t + 1):
                self.GasOut_P[it, jt].setlb(0)
                self.GasIn_F[it].setlb(0)
                self.GasOut_F[it].setlb(0)
                self.GasOut_T[it, jt].setlb(0)
                # self.SolidIn_Fm[it, jt].setlb(0)
                self.SolidIn_P[it, jt].setlb(0)
                self.SolidOut_P[it, jt].setlb(0)
                self.rhog_in[it, jt].setlb(0)
                self.rhog_out[it, jt].setlb(0)
                self.DownOut_P[it, jt].setlb(0)
                self.h_downcomer[it, jt].setlb(0)
                self.vmf[it, jt].setlb(0)
                self.db0[it, jt].setlb(0)
                self.Sit[it, jt].setlb(0)
                self.Sot[it, jt].setlb(0)
                self.g1[it, jt].setlb(0)
                self.P_l[it, jt].setlb(0)
                self.Phx_l[it, jt].setlb(0)
                self.ebin_l[it, jt].setlb(0)
                self.Gb_l[it, jt].setlb(0)
                self.Tgb_l[it, jt].setlb(0)
                self.c_capture[it, jt].setlb(0)
        for it in range(1, self.nfe_t + 1):
            for jt in range(1, self.ncp_t + 1):
                for cx in ['c', 'h', 'n']:
                    self.GasOut_z[it, jt, cx].setlb(0)
                    self.cbin_l[it, jt, cx].setlb(0)
                    self.cein_l[it, jt, cx].setlb(0)
                    self.ccwin_l[it, jt, cx].setlb(0)
                    self.ne_l[it, jt, cx].setlb(0)
                    self.yb_l[it, jt, cx].setlb(0)

        for it in range(1, self.nfe_t + 1):
            for jt in range(1, self.ncp_t + 1):
                for ix in range(1, self.nfe_x + 1):
                    for jx in range(0, self.ncp_x + 1):
                        self.Ar[it, jt, ix, jx].setlb(0)
                        self.cbt[it, jt, ix, jx].setlb(0)
                        self.cct[it, jt, ix, jx].setlb(0)
                        self.cet[it, jt, ix, jx].setlb(0)
                        self.db[it, jt, ix, jx].setlb(0)
                        self.dbe[it, jt, ix, jx].setlb(0)
                        self.dbm[it, jt, ix, jx].setlb(0)
                        self.dbu[it, jt, ix, jx].setlb(0)
                        self.delta[it, jt, ix, jx].setlb(0)
                        self.e[it, jt, ix, jx].setlb(0)
                        self.ebin[it, jt, ix, jx].setlb(0)
                        self.ed[it, jt, ix, jx].setlb(0)
                        self.fb[it, jt, ix, jx].setlb(0)
                        self.fc[it, jt, ix, jx].setlb(0)
                        self.fcw[it, jt, ix, jx].setlb(0)
                        self.fn[it, jt, ix, jx].setlb(0)
                        self.g2[it, jt, ix, jx].setlb(0)
                        self.g3[it, jt, ix, jx].setlb(0)
                        self.Gb[it, jt, ix, jx].setlb(0)
                        self.Hbc[it, jt, ix, jx].setlb(0)
                        self.Hce[it, jt, ix, jx].setlb(0)
                        self.hd[it, jt, ix, jx].setlb(0)
                        self.hl[it, jt, ix, jx].setlb(0)
                        self.hp[it, jt, ix, jx].setlb(0)
                        self.ht[it, jt, ix, jx].setlb(0)
                        self.Jc[it, jt, ix, jx].setlb(0)
                        self.Je[it, jt, ix, jx].setlb(0)
                        self.k1c[it, jt, ix, jx].setlb(1e-07)
                        self.k1e[it, jt, ix, jx].setlb(0)
                        self.k2c[it, jt, ix, jx].setlb(0)
                        self.k2e[it, jt, ix, jx].setlb(0)
                        self.k3c[it, jt, ix, jx].setlb(0)
                        self.k3e[it, jt, ix, jx].setlb(0)
                        self.db[it, jt, ix, jx].setlb(0)
                        self.dbe[it, jt, ix, jx].setlb(0)
                        self.dbm[it, jt, ix, jx].setlb(0)
                        self.dbu[it, jt, ix, jx].setlb(0)
                        self.delta[it, jt, ix, jx].setlb(0)
                        self.e[it, jt, ix, jx].setlb(0)
                        self.ebin[it, jt, ix, jx].setlb(0)
                        self.ed[it, jt, ix, jx].setlb(0)
                        self.fb[it, jt, ix, jx].setlb(0)
                        self.fc[it, jt, ix, jx].setlb(0)
                        self.fcw[it, jt, ix, jx].setlb(0)
                        self.fn[it, jt, ix, jx].setlb(0)
                        self.g2[it, jt, ix, jx].setlb(0)
                        self.g3[it, jt, ix, jx].setlb(0)
                        self.Gb[it, jt, ix, jx].setlb(0)
                        self.Hbc[it, jt, ix, jx].setlb(0)
                        self.Hce[it, jt, ix, jx].setlb(0)
                        self.hd[it, jt, ix, jx].setlb(0)
                        self.hl[it, jt, ix, jx].setlb(0)
                        self.hp[it, jt, ix, jx].setlb(0)
                        self.ht[it, jt, ix, jx].setlb(0)
                        self.Jc[it, jt, ix, jx].setlb(0)
                        self.Je[it, jt, ix, jx].setlb(0)
                        self.k1c[it, jt, ix, jx].setlb(0)
                        self.k1e[it, jt, ix, jx].setlb(0)
                        self.k2c[it, jt, ix, jx].setlb(0)
                        self.k2e[it, jt, ix, jx].setlb(0)
                        self.k3c[it, jt, ix, jx].setlb(0)
                        self.k3e[it, jt, ix, jx].setlb(0)
                        self.Kcebs[it, jt, ix, jx].setlb(0)
                        self.Ke1c[it, jt, ix, jx].setlb(0)
                        self.Ke1e[it, jt, ix, jx].setlb(0)
                        self.Ke2c[it, jt, ix, jx].setlb(0)
                        self.Ke2e[it, jt, ix, jx].setlb(0)
                        self.Ke3c[it, jt, ix, jx].setlb(0)
                        self.Ke3e[it, jt, ix, jx].setlb(0)
                        self.kpa[it, jt, ix, jx].setlb(0)
                        self.Nup[it, jt, ix, jx].setlb(0)
                        self.P[it, jt, ix, jx].setlb(1e-07)
                        self.Phx[it, jt, ix, jx].setlb(0)
                        self.Red[it, jt, ix, jx].setlb(0)
                        self.r1c[it, jt, ix, jx].setlb(1e-07)
                        self.rhog[it, jt, ix, jx].setlb(0)
                        self.tau[it, jt, ix, jx].setlb(0)
                        self.Tgb[it, jt, ix, jx].setlb(0)
                        self.Tgc[it, jt, ix, jx].setlb(0)
                        self.Tge[it, jt, ix, jx].setlb(0)
                        self.Thx[it, jt, ix, jx].setlb(0)
                        self.Tsc[it, jt, ix, jx].setlb(0)
                        self.Tse[it, jt, ix, jx].setlb(0)
                        self.Ttube[it, jt, ix, jx].setlb(0)
                        self.vb[it, jt, ix, jx].setlb(0)
                        self.vbr[it, jt, ix, jx].setlb(0)
                        self.ve[it, jt, ix, jx].setlb(0)
                        self.vg[it, jt, ix, jx].setlb(0)
        for it in range(1, self.nfe_t + 1):
            for jt in range(1, self.ncp_t + 1):
                for ix in range(1, self.nfe_x + 1):
                    for jx in range(0, self.ncp_x + 1):
                        for cx in ['c', 'h', 'n']:
                            self.cb[it, jt, ix, jx, cx].setlb(0)
                            self.cbin[it, jt, ix, jx, cx].setlb(0)
                            self.cc[it, jt, ix, jx, cx].setlb(0)
                            self.ccwin[it, jt, ix, jx, cx].setlb(0)
                            self.ce[it, jt, ix, jx, cx].setlb(0)
                            self.cein[it, jt, ix, jx, cx].setlb(0)
                            self.D[it, jt, ix, jx, cx].setlb(0)
                            self.Kbc[it, jt, ix, jx, cx].setlb(0)
                            self.Kce[it, jt, ix, jx, cx].setlb(0)

                            self.nc[it, jt, ix, jx, cx].setlb(1e-07)
                            self.ne[it, jt, ix, jx, cx].setlb(0)

                            self.yb[it, jt, ix, jx, cx].setlb(0)
                            self.yc[it, jt, ix, jx, cx].setlb(1e-07)
                            self.ye[it, jt, ix, jx, cx].setlb(1e-07)

        # for it in range(1, self.nfe_t + 1):
        #     for jt in range(1, self.ncp_t + 1):
        #         for ix in range(1, self.nfe_x + 1):
        #             for jx in range(1, self.ncp_x + 1):
        #                 self.Hgb[it, jt, ix, jx].setlb(0)
        #                 self.Hgc[it, jt, ix, jx].setlb(0)
        #                 self.Hsc[it, jt, ix, jx].setlb(0)
        #                 self.Hge[it, jt, ix, jx].setlb(0)
        #                 self.Hse[it, jt, ix, jx].setlb(0)
        #                 self.Ws[it, jt, ix, jx].setlb(0)
        #                 for cx in ['c', 'h', 'n']:
        #                     self.Ngb[it, jt, ix, jx, cx].setlb(0)
        #                     self.Ngc[it, jt, ix, jx, cx].setlb(0)
        #                     self.Nge[it, jt, ix, jx, cx].setlb(0)
        #                     self.Nse[it, jt, ix, jx, cx].setlb(0)

    def clear_bounds(self):
        """Sets bounds of variables
        Args:
            None
        Returns:
            None"""
        for var in self.component_data_objects(Var):
            var.setlb(None)
            var.setub(None)

    def init_steady_ref(self):
        """If the model is steady, we try to initialize it with an initial guess from ampl"""
        self.create_bounds()
        for var in self.component_data_objects(Var):
            try:
                var.set_value(ss[var.parent_component().name, var.index()])
            except KeyError:
                pass
        if self.nfe_t == 1 and self.ncp_t == 1:
            solver = SolverFactory('asl:ipopt')
            solver.options["linear_solver"] = "ma57"
            # solver.options["halt_on_ampl_error"] = "yes"
            with open("ipopt.opt", "w") as f:
                f.write("max_iter 10000\n")
                # f.write("mu_init 1e-08\n")
                f.write("bound_push 1e-06\n")
                f.close()
            solver.options["print_user_options"] = "yes"
            self.display(filename="whatevs.txt")
            someresults = solver.solve(self, tee=True, symbolic_solver_labels=True, report_timing=True)
            self.solutions.load_from(someresults)

    def equalize_u(self, direction="u_to_r"):
        """set current controls to the values of their respective dummies"""
        if direction == "u_to_r":
            for i in self.GasIn_F.keys():
                self.GasIn_F[i].set_value(value(self.u1[i]))
            for i in self.GasOut_F.keys():
                self.GasOut_F[i].set_value(value(self.u2[i]))
        elif direction == "r_to_u":
            for i in self.u1.keys():
                self.u1[i].value = value(self.GasIn_F[i])
            for i in self.u2.keys():
                self.u2[i].value = value(self.GasOut_F[i])
