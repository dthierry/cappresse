#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from pyomo.core.base import Param, ConcreteModel, Var, Constraint, Set, Suffix
from pyomo.core.kernel.numvalue import value
from pyomo.opt import ProblemFormat
from nmpc_mhe.aux.cpoinsc import collptsgen
from nmpc_mhe.aux.lagrange_f import lgr, lgry, lgrdot, lgrydot
from nob5_con_hi_t import *
from initial_s_Gb import ss
import os, sys

"""
BFB reformulation with momemtum balance

"""


__author__ = 'David M Thierry'


    
nfe_t = 2
ncp_t = 2
nfe_x = 5
ncp_x = 3
_t = 1

m = ConcreteModel()

self.dref = dict()
_L = 5.
_alp_gauB_x = 0
_bet_gauB_x = 0

_alp_gauB_t = 1
_bet_gauB_t = 0
nfe_x = nfe_x
ncp_x = ncp_x

tau_x = collptsgen(ncp_x, _alp_gauB_x, _bet_gauB_x)
tau_t = collptsgen(ncp_t, _alp_gauB_t, _bet_gauB_t)

# start at zero
tau_i_x = {0: 0.}
tau_i_t = {0: 0.}
# create a list

for ii in range(1, ncp_x + 1):
    tau_i_x[ii] = tau_x[ii - 1]

for ii in range(1, ncp_t + 1):
    tau_i_t[ii] = tau_t[ii - 1]

# ======= SETS ======= #
# For finite element = 1 .. K
# This has to be > 0

m.fe_t = Set(initialize=[ii for ii in range(0, m.nfe_t)])
m.fe_t.pprint()
m.fe_x = Set(initialize=[ii for ii in range(0, nfe_x)])

# collocation points
# collocation points for diferential variables
m.cp_x = Set(initialize=[ii for ii in range(0, ncp_x + 1)])
m.cp_t = Set(initialize=[ii for ii in range(0, ncp_t + 1)])

m.cp_xa = Set(within=m.cp_x, initialize=[ii for ii in range(1, ncp_x + 1)])
m.cp_ta = Set(within=m.cp_t, initialize=[ii for ii in range(1, ncp_t + 1)])

# components
m.sp = Set(initialize=['c', 'h', 'n'])
m.sp2 = Set(initialize=['c', 'h'])

# create collocation param
m.taucp_x = Param(m.cp_x, initialize=tau_i_x)
m.taucp_x = Param(m.cp_t, initialize=tau_i_t)

m.ldot_x = Param(m.cp_x, m.cp_x, initialize=
(lambda m, j, k: lgrdot(j, m.taucp_x[k], ncp_x, _alp_gauB_x, _bet_gauB_x)))
# (lambda m, i, j: fldoti_x(m, i, j, ncp_x, _alp_gauB_x, _bet_gauB_x)))
m.ldot_t = Param(m.cp_t, m.cp_t, initialize=
(lambda m, i, j: fldoti_t(m, i, j, ncp_t, _alp_gauB_t, _bet_gauB_t)))
def lydot2(m, i, j):
    y = fldotyi(m, i, j, ncp_x, _alp_gauB_x, _bet_gauB_x)
    if abs(y) > 1e-08:
        return y
    else:
        return 0
# m.lydot = Param(m.cp_x, m.cp_x, initialize=
# (lambda m, i, j: fldotyi(m, i, j, ncp_x, _alp_gauB_x, _bet_gauB_x)))

m.lydot = Param(m.cp_x, m.cp_x, initialize=lydot2)

m.l1_x = Param(m.cp_x, initialize=
(lambda m, i: flj1_x(m, i, ncp_x, _alp_gauB_x, _bet_gauB_x)))
m.l1_t = Param(m.cp_t, initialize=
(lambda m, i: flj1_t(m, i, ncp_t, _alp_gauB_t, _bet_gauB_t)))
m.l1y = Param(m.cp_x, initialize=
(lambda m, i: fljy1(m, i, ncp_x, _alp_gauB_x, _bet_gauB_x)))
m.lgr = Param(m.cp_x, m.cp_x, initialize=
(lambda m, i, j: f_lj_x(m, i, j, ncp_x, _alp_gauB_x, _bet_gauB_x)))

m.A1 = Param(initialize=56200.0)
m.A2 = Param(initialize=2.62)
m.A3 = Param(initialize=98.9)
m.ah = Param(initialize=0.8)
m.Ao = Param(initialize=4e-4)
m.ap = Param(initialize=90.49773755656109)
self.Ax = Param(initialize=63.32160110335595)
self.cpg_mol = Param(initialize=30.0912)

_cpgcsb = {'c': 38.4916, 'h': 33.6967}
_cpgcst = {'c': 38.4913, 'h': 33.6968}
_cpgcsc = {'c': 39.4617, 'h': 29.1616}
_cpgcgc = {'c': 39.2805, 'h': 33.7988, 'n': 29.1578}
_cpgcge = {'c': 39.3473, 'h': 33.8074, 'n': 29.1592}
_cpgcse = {'c': 39.3486, 'h': 33.8076}

self.cpgcsb = Param(m.sp, initialize=_cpgcsb)
self.cpgcst = Param(m.sp, initialize=_cpgcst)
self.cpgcgc = Param(m.sp, initialize=_cpgcgc)
self.cpgcge = Param(m.sp, initialize=_cpgcge)
self.cpgcsc = Param(m.sp, initialize=_cpgcsc)
self.cpgcse = Param(m.sp, initialize=_cpgcse)
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
self.GasIn_T = Param(m.fe_t, initialize=40, mutable=True)  # needs i_value
self._GasIn_z = {'c': 0.13, 'h': 0.06, 'n': 0.81}
self._GasIn_z = {}
for i in range(0, m.nfe_t):
    self._GasIn_z[i, 'c'] = 0.13
    self._GasIn_z[i, 'h'] = 0.06
    self._GasIn_z[i, 'n'] = 0.81
# self.GasIn_z = Param(m.sp, initialize=_GasIn_z)
self.GasIn_z = Param(m.fe_t, m.sp, initialize=self._GasIn_z, mutable=True)

# self.flue_gas_P = Param(initialize=1.8)

# Solid input condition
_nin = {'c': 0.01, 'h': 0.7, 'n': 0.7}
self.nin = Param(m.sp, initialize=_nin, mutable=True)
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

self.per_opening3 = Param(m.fe_t, initialize=50., mutable=True)
self.per_opening4 = Param(initialize=50)

self.eavg = Param(initialize=0.591951)

self.llast = Param(initialize=5.)
self.lenleft = Param(initialize=10.)
self.hi_x = Param(m.fe_x, initialize=fir_hi)
hi_t = dict.fromkeys(m.fe_t)
for key in hi_t.keys():
    hi_t[key] = 1.0 if steady else _t/m.nfe_t

self.hi_t = hi_t if steady else Param(m.fe_t, initialize=hi_t, mutable=True)

print(m.nfe_t)
print(_t)
# print()
print(type(hi_t))
print(hi_t)
if type(self.hi_t) != dict:
    self.hi_t.pprint()

self.l = Param(m.fe_x, m.cp_x, initialize=fl_irule)

# --------------------------------------------------------------------------------------------------------------
#: First define differential state variables (state: x, ic-Param: x_ic, derivative-Var:dx_dt
#: States (differential) section

zero2_x = dict.fromkeys(m.fe_x * m.cp_x)
zero3_x = dict.fromkeys(m.fe_x * m.cp_x * m.sp)
zero4 = dict.fromkeys(m.fe_t * m.cp_ta * m.fe_x * m.cp_x)
zero5 = dict.fromkeys(m.fe_t * m.cp_ta * m.fe_x * m.cp_x * m.sp)

for key in zero2_x.keys():
    zero2_x[key] = 0.0
for key in zero3_x.keys():
    zero3_x[key] = 0.0
for key in zero4.keys():
    zero4[key] = 0.0
for key in zero5.keys():
    zero5[key] = 0.0

#:  State-variables
self.Ngb = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.Hgb = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, initialize=1.)
self.Ngc = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.Hgc = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, initialize=1.)
self.Nsc = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.Hsc = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, initialize=1.)
self.Nge = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.Hge = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, initialize=1.)
self.Nse = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.Hse = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, initialize=1.)
self.mom = Var(m.fe_t, m.cp_t, m.fe_x, m.cp_x, initialize=1.)

self.vg = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Gb = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)


#:  Initial state-Param
self.Hgc_ic = zero2_x if steady else Param(m.fe_x, m.cp_x, initialize=1., mutable=True)
self.Nsc_ic = zero3_x if steady else Param(m.fe_x, m.cp_x, m.sp, initialize=1., mutable=True)
self.Hsc_ic = zero2_x if steady else Param(m.fe_x, m.cp_x, initialize=1., mutable=True)
self.Hge_ic = zero2_x if steady else Param(m.fe_x, m.cp_x, initialize=1., mutable=True)
self.Nse_ic = zero3_x if steady else Param(m.fe_x, m.cp_x, m.sp, initialize=1., mutable=True)
self.Hse_ic = zero2_x if steady else Param(m.fe_x, m.cp_x, initialize=1., mutable=True)

#:  Derivative-var
self.dHgc_dt = zero4 if steady else Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=0.0001)
self.dNsc_dt = zero5 if steady else Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=0.0001)
self.dHsc_dt = zero4 if steady else Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=0.0001)
self.dHge_dt = zero4 if steady else Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=0.0001)
self.dNse_dt = zero5 if steady else Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=0.0001)
self.dHse_dt = zero4 if steady else Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=0.0001)

# --------------------------------------------------------------------------------------------------------------

# --------------------------------------------------------------------------------------------------------------
#: Algebraic variables
#: BVP diff variables
self.dvg_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
# self.dGb_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.dcb_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.dTgb_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.dP_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.drhog_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)

self.dhxh_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.dcein_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.decwin_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.deein_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.dPhx_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.dccwin_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
# self.dz_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)

#: Second-order derivatives
self.dvgx_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.dcbx_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.dTgbx_dx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)

#: Second-order dummy derivatives
self.vgx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.cbx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.Tgbx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)

self.HXIn_h = Var(m.fe_t, m.cp_ta, initialize=1.)
self.GasIn_P = Param(m.fe_t, initialize=1.315040276216880)
self.GasIn_F = Param(m.fe_t, initialize=9937.98446662)
self.GasOut_F = Var(m.fe_t, m.cp_ta, initialize=1.)
self.GasOut_T = Var(m.fe_t, m.cp_ta, initialize=1.)
self.GasOut_z = Var(m.fe_t, m.cp_ta, m.sp, initialize=1.)
self.hsint = Var(m.fe_t, m.cp_ta, initialize=1.)
# self.vmf = Var(m.fe_t, m.cp_ta, initialize=1.)
self.wvmf = Var(m.fe_t, m.cp_ta, initialize=-4.77, bounds=(-50, 5))

self.db0 = Var(m.fe_t, m.cp_ta, initialize=1.)
self.Sit = Var(m.fe_t, initialize=162.183495794)
self.Sot = Var(m.fe_t, m.cp_ta, initialize=1.)
self.g1 = Var(m.fe_t, m.cp_ta, initialize=1.)
self.Ar = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.cb = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
#self.cc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)

self.wcc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=2.7, bounds=(-50, 5))
self.ccwin = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.cein = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)

self.ce = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)

self.D = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.db = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.dbe = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.dbm = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.dbu = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.delta = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.dThx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.ecwin = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.ed = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.eein = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.fb = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.fc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.fcw = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.fn = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.g2 = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.g3 = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Hbc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Hce = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.hd = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Hgbulk = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
# self.hl = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.whl = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=-4.5384, bounds=(-50, 5))
# self.hp = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.whp = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=-2.6, bounds=(-50, 5))
self.Hsbulk = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.hsc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.hse = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.ht = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.hxh = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Jc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Je = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.k1c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.k1e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.k2c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.k2e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.k3c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.k3e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Kbc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.Kce = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.Kcebs = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Ke1c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Ke1e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)

self.Ke2c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Ke2e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Ke3c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Ke3e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)

self.wKe2c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1., bounds=(-100, 5))
self.wKe2e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1., bounds=(-100, 5))
self.wKe3c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1., bounds=(-100, 5))
self.wKe3e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1., bounds=(-100, 5))


self.Kgbulk = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.kpa = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Ksbulk = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.nc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.ne = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.Nup = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.P = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Phx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.r1c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.r1e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.r2c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.r2e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.r3c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.r3e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Red = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.rgc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp2, initialize=1.)
self.rge = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp2, initialize=1.)
self.rhog = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.rsc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.rse = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.tau = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Tgb = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Tgc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Tge = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Thx = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Tsc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Tse = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.Ttube = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.vb = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.vbr = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
#self.ve = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.wve = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=-3.5, bounds=(-50,5))



# self.yb = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
wyb = {}
wyb['h'] = -4.25444644995
wyb['c'] = -3.66076140272
wyb['n'] = -0.0407322869606
wyc = {}
wyc['h'] = -4.06652502791
wyc['c'] = -4.68790044102
wyc['n'] = -0.0266960183982
wye = {}
wye['h'] = -4.14202576203
wye['c'] = -4.96629735827
wye['n'] = -0.0231248623728

self.wyb = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=lambda m, i, j, k, l, s: wyb[s], bounds=(-50, 5))
# self.yc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.wyc = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=lambda m, i, j, k, l, s: wyc[s], bounds=(-50, 5))
#self.ye = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=1.)
self.wye = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, initialize=lambda m, i, j, k, l, s: wye[s], bounds=(-50, 5))

# self.z = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.)
self.cein_l = Var(m.fe_t, m.cp_ta, m.sp, initialize=1.)
self.ecwin_l = Var(m.fe_t, m.cp_ta, initialize=1.)
self.eein_l = Var(m.fe_t, m.cp_ta, initialize=1.)
self.hxh_l = Var(m.fe_t, m.cp_ta, initialize=1.)
# self.P_l = Var(m.fe_t, m.cp_ta, initialize=1.)
self.Phx_l = Var(m.fe_t, m.cp_ta, initialize=1.)
self.ccwin_l = Var(m.fe_t, m.cp_ta, m.sp, initialize=1.)
self.hse_l = Var(m.fe_t, m.cp_ta, initialize=1.)
self.ne_l = Var(m.fe_t, m.cp_ta, m.sp, initialize=1.)
self.cb_l = Var(m.fe_t, m.cp_ta, m.sp, initialize=1.)

self.c_capture = Var(m.fe_t, m.cp_ta, initialize=1.)
# --------------------------------------------------------------------------------------------------------------
#: Controls
self.u1 = Param(m.fe_t, initialize=162.183495794, mutable=True)
# self.u2 = Param(m.fe_t, initialize=50., mutable=True)

# self.per_opening1 = Var(m.fe_t, initialize=85.)
# self.per_opening2 = Var(m.fe_t, initialize=50.)

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
#: Constraints for the differential states
#: Then the ode-Con:de_x, collocation-Con:dvar_t_x, noisy-Expr: noisy_x, cp-Constraint: cp_x, initial-Con: x_icc
#: Differential equations

#: Differential equations
self.de_Ngb = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=de_ngb_rule)
self.de_Hgb = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=de_hgb_rule)
self.de_Ngc = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=de_ngc_rule)
self.de_Hgc = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=de_hgc_rule)
self.de_Nsc = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=de_nsc_rule)
self.de_Hsc = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=de_hsc_rule)
self.de_Nge = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=de_nge_rule)
self.de_Hge = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=de_hge_rule)
self.de_Nse = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=de_nse_rule)
self.de_Hse = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=de_hse_rule)
# self.de_Gb = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=de_Gb_rule)

self.de_mom = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=alt_de_Gb_rule)
# self.alt_de_Gb.deactivate()
# self.de_Gb.deactivate()

#: Collocation equations
self.dvar_t_Hgc = None if steady else Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_t_hgc)
self.dvar_t_Nsc = None if steady else Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp,
                                                 rule=fdvar_t_nsc)
self.dvar_t_Hsc = None if steady else Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_t_hsc)
self.dvar_t_Hge = None if steady else Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_t_hge)
self.dvar_t_Nse = None if steady else Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp,
                                                 rule=fdvar_t_nse)
self.dvar_t_Hse = None if steady else Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_t_hse)

#: Continuation equations (redundancy here)
if m.nfe_t > 1:
    #: Noisy expressions
    self.noisy_Hgc = None if steady else Expression(m.fe_t, m.fe_x, m.cp_x, rule=fcp_t_hgc)
    self.noisy_Nsc = None if steady else Expression(m.fe_t, m.fe_x, m.cp_x, m.sp, rule=fcp_t_nsc)
    self.noisy_Hsc = None if steady else Expression(m.fe_t, m.fe_x, m.cp_x, rule=fcp_t_hsc)
    self.noisy_Hge = None if steady else Expression(m.fe_t, m.fe_x, m.cp_x, rule=fcp_t_hge)
    self.noisy_Nse = None if steady else Expression(m.fe_t, m.fe_x, m.cp_x, m.sp, rule=fcp_t_nse)
    self.noisy_Hse = None if steady else Expression(m.fe_t, m.fe_x, m.cp_x, rule=fcp_t_hse)


    self.cp_Hgc = None if steady else Constraint(m.fe_t, m.fe_x, m.cp_x,
                                                 rule=lambda m, i, ix, jx:
                                                 m.noisy_Hgc[i, ix, jx] == 0.0
                                                 if i < (m.nfe_t - 1) and 0 < jx <= m.ncp_x else Constraint.Skip)
    self.cp_Nsc = None if steady else Constraint(m.fe_t, m.fe_x, m.cp_x, m.sp,
                                                 rule=lambda m, i, ix, jx, c:
                                                 m.noisy_Nsc[i, ix, jx, c] == 0.0
                                                 if i < (m.nfe_t - 1) and 0 < jx <= m.ncp_x else Constraint.Skip)
    self.cp_Hsc = None if steady else Constraint(m.fe_t, m.fe_x, m.cp_x,
                                                 rule=lambda m, i, ix, jx:
                                                 m.noisy_Hsc[i, ix, jx] == 0.0
                                                 if i < (m.nfe_t - 1) and 0 < jx <= m.ncp_x else Constraint.Skip)

    self.cp_Hge = None if steady else Constraint(m.fe_t, m.fe_x, m.cp_x,
                                                 rule=lambda m, i, ix, jx:
                                                 m.noisy_Hge[i, ix, jx] == 0.0
                                                 if i < (m.nfe_t - 1) and 0 < jx <= m.ncp_x else Constraint.Skip)
    self.cp_Nse = None if steady else Constraint(m.fe_t, m.fe_x, m.cp_x, m.sp,
                                                 rule=lambda m, i, ix, jx, c:
                                                 m.noisy_Nse[i, ix, jx, c] == 0.0
                                                 if i < (m.nfe_t - 1) and 0 < jx <= m.ncp_x else Constraint.Skip)
    self.cp_Hse = None if steady else Constraint(m.fe_t, m.fe_x, m.cp_x,
                                                 rule=lambda m, i, ix, jx:
                                                 m.noisy_Hse[i, ix, jx] == 0.0
                                                 if i < (m.nfe_t - 1) and 0 < jx <= m.ncp_x else Constraint.Skip)

#: Initial condition-Constraints
self.Hgc_icc = None if steady else Constraint(m.fe_x, m.cp_x, rule=ic_hgc_rule)
self.Nsc_icc = None if steady else Constraint(m.fe_x, m.cp_x, m.sp, rule=ic_nsc_rule)
self.Hsc_icc = None if steady else Constraint(m.fe_x, m.cp_x, rule=ic_hsc_rule)
self.Hge_icc = None if steady else Constraint(m.fe_x, m.cp_x, rule=ic_hge_rule)
self.Nse_icc = None if steady else Constraint(m.fe_x, m.cp_x, m.sp, rule=ic_nse_rule)
self.Hse_icc = None if steady else Constraint(m.fe_x, m.cp_x, rule=ic_hse_rule)

#: Algebraic definitions
self.ae_ngb = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=ngb_rule)
self.ae_hgb = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=hgb_rule)
self.ae_ngc = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=ngc_rule)
self.ae_hgc = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=hgc_rule)
self.ae_nsc = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=nsc_rule)
self.ae_hsc = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=hsc_rule)
self.ae_nge = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=nge_rule)
self.ae_hge = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=hge_rule)
self.ae_nse = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=nse_rule)
self.ae_hse = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=hse_rule)
self.mom_rule = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=mom_rule)

self.ae_Gb = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=Gb_rule)

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
#: Algebraic constraints
#: ddx collocation
self.xdvar_cein = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=fdvar_x_cein_)
self.xdvar_ecwin = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_x_ecwin_)
self.xdvar_eein = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_x_eein_)
self.xdvar_hxh = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_x_hxh_)
self.xdvar_phx = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_x_phx_)
self.xdvar_ccwin = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=fdvar_x_ccwin_)

# self.xdvar_z = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_z_)

self.xdvar_vg = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_x_vg_)
# self.xdvar_Gb = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_x_Gb_)
self.xdvar_cb = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=fdvar_x_cb_)
self.xdvar_Tgb = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_x_Tgb_)
# self.xdvar_P = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_x_P_)
self.xdvar_P = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a21_rule)
self.xdvar_rhog = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=drhogx_rule)

self.xcp_cein = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.sp, rule=fcp_x_cein)
self.xcp_ecwin = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_x_ecwin)
self.xcp_eein = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_x_eein)
self.xcp_hxh = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_x_hxh)
self.xcp_phx = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_x_phx)
self.xcp_ccwin = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.sp, rule=fcp_x_ccwin)
# self.xcp_z = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_z_)

self.xcp_vg = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_x_vb)
# self.xcp_Gb = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_x_Gb)
self.xcp_cb = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.sp, rule=fcp_x_cb)
self.xcp_Tgb = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_x_Tgb)
# self.xcp_P = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_x_P)
#: P is now piecewise-continuous

#: d2dx2 collocation
self.xdvar_vgx = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_x_dvg_dx_)
self.xdvar_cbx = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=fdvar_x_dcb_dx_)
self.xdvar_Tbgx = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=fdvar_x_dTgb_dx_)

self.xcp_vgx = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_x_dvb_dx)
self.xcp_cbx = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.sp, rule=fcp_x_dcb_dx)
self.xcp_Tbgx = Constraint(m.fe_t, m.cp_ta, m.fe_x, rule=fcp_x_dTgb_dx)


#: d2dx2 dummy de
self.xde_vgx = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=dum_dex_vg_rule)
self.xde_cbx = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=dum_dex_cb_rule)
self.xde_Tgbx = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=dum_dex_Tgb_rule)
self.xde_P = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=dpdx_rule)

self.xde_hxh = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=dhxh_rule)
self.xde_Phx = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=dphx_rule)  # dPhx_dx constraint

# self.xde_z = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=dex_z_rule)

#: Last cp point equation (differential-bvp)
self.xl_cein = Constraint(m.fe_t, m.cp_ta, m.sp, rule=fzl_cein)
self.xl_ecwin = Constraint(m.fe_t, m.cp_ta, rule=fzl_ecwin)
self.xl_eein = Constraint(m.fe_t, m.cp_ta, rule=fzl_eein)
self.xl_hxh = Constraint(m.fe_t, m.cp_ta, rule=fzl_hxh)  #
self.xl_Phx = Constraint(m.fe_t, m.cp_ta, rule=fzl_phx)  #
self.xl_ccin = Constraint(m.fe_t, m.cp_ta, m.sp, rule=fzl_ccwin)

self.xl_cb = Constraint(m.fe_t, m.cp_ta, m.sp, rule=fzl_cb)

#: Boundary conditions
# self.xbc_Gb0 = Constraint(m.fe_t, m.cp_ta, rule=bc_Gb0_rule)
self.xbc_Tgb0 = Constraint(m.fe_t, m.cp_ta, rule=bc_Tgb0_rule)
self.xbc_cb0 = Constraint(m.fe_t, m.cp_ta, m.sp, rule=bc_cb0_rule)
# self.xbc_P0 = Constraint(m.fe_t, m.cp_ta, rule=bc_P0_rule)
self.xbc_vgf0 = Constraint(m.fe_t, m.cp_ta, rule=bc_vg0_rule)
# self.xbc_vgf0.deactivate()


self.xbx_hxh = Constraint(m.fe_t, m.cp_ta, rule=bc_hxh_rule)
self.xbx_pxh = Constraint(m.fe_t, m.cp_ta, rule=bc_phx_rule)


#: Boundary conditions on ddx
self.xbc_vgx = Constraint(m.fe_t, m.cp_ta, rule=bc_vgx_rule)
self.xbc_Tgbx = Constraint(m.fe_t, m.cp_ta, rule=bc_Tgbx_rule)
self.xbc_cbx = Constraint(m.fe_t, m.cp_ta, m.sp, rule=bc_cbx_rule)

self.xbc_mol = Constraint(m.fe_t, m.cp_ta, m.sp, rule=bc_mol_rule)
self.xbc_ene = Constraint(m.fe_t, m.cp_ta, rule=bc_ene_rule)

self.xbc_mol0 = Constraint(m.fe_t, m.cp_ta, m.sp, rule=bc_mol0_rule)
self.xbc_ene0 = Constraint(m.fe_t, m.cp_ta, rule=bc_ene0_rule)

# self.xbc_z0 = Constraint(m.fe_t, m.cp_ta, rule=bc_z0_rule)

self.a4 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a4_rule)
self.a5 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a5_rule)
self.a8 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=a8_rule)
self.a9 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=a9_rule)

def a11_rule_alternative(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return 0.0 == m.Je[it, jt, ix, jx] - m.Jc[it, jt, ix, jx]
        # return m.dJc_dx[i, j] == m.dummyJ[i, j]
    else:
        return Constraint.Skip

self.a11_2 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a11_rule_alternative)
#self.a11_2 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a11_rule_2)#

# self.a12 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a12_rule)
self.a13 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a13_rule)
self.a14 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a14_rule)
self.a15 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=a15_rule)
self.a16 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=a16_rule)
self.a17 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=a17_rule)
# self.a18 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a18_rule)
self.a22 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a22_rule)
self.a23 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a23_rule)
self.a24 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a24_rule)
self.a25 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a25_rule)
self.a26 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a26_rule)
self.a27 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a27_rule)
self.a28 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a28_rule)
self.a29 = Constraint(m.fe_t, m.cp_ta, rule=a29_rule)
self.a30 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a30_rule)
self.a31 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a31_rule)
self.a32 = Constraint(m.fe_t, m.cp_ta, rule=a32_rule)
self.a33 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a33_rule)
self.a34 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a34_rule)
self.a35 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a35_rule)
self.a36 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a36_rule)
self.a37 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a37_rule)
self.a38 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=a38_rule)
self.a39 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=a39_rule)
self.a40 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a40_rule)
self.a41 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a41_rule)
self.a42 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a42_rule)
self.a43 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a43_rule)
self.a44 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a44_rule)
self.a45 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a45_rule)
self.a46 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a46_rule)
self.a47 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a47_rule)
self.a48 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a48_rule)
self.a49 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a49_rule)
self.a50 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a50_rule)
self.a51 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a51_rule)
self.a52 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a52_rule)

self.a54 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a54_rule)
self.a55 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a55_rule)
self.a56 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a56_rule)
self.a57 = Constraint(m.fe_t, m.cp_ta, rule=a57_rule)
self.a58 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a58_rule)

        def wKe2c_rule(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.Ke2c[it, jt, ix, jx]) == (exp(m.wKe2c[it, jt, ix, jx]))
            else:
                return Constraint.Skip

        def wKe2e_rule(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.Ke2e[it, jt, ix, jx]) == (exp(m.wKe2e[it, jt, ix, jx]))
            else:
                return Constraint.Skip

        def wKe3c_rule(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.Ke3c[it, jt, ix, jx]) == (exp(m.wKe3c[it, jt, ix, jx]))
            else:
                return Constraint.Skip

        def wKe3e_rule(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.Ke3e[it, jt, ix, jx]) == (exp(m.wKe3e[it, jt, ix, jx]))
            else:
                return Constraint.Skip
        # pls check this again
        self.wKe2c_ = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=wKe2c_rule)
        self.wKe2e_ = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=wKe2e_rule)
        self.wKe3c_ = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=wKe3c_rule)
        self.wKe3e_ = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=wKe3e_rule)

        ###
        self.wk2c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.0, bounds=(-100, 5))  #: dummy

        def wk2c_rule(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.k2c[it, jt, ix, jx]) == (exp(m.wk2c[it, jt, ix, jx]))
            else:
                return Constraint.Skip

        def a71_rulex(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.r2c[it, jt, ix, jx]) == (exp(m.wk2c[it, jt, ix, jx]) * (
                (1 - 2 * (m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) - (m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) *
                m.nc[it, jt, ix, jx, 'h'] * m.rhos * m.P[it, jt, ix, jx] * exp(m.wyc[it, jt, ix, jx, 'c']) * 1E5 - (
                ((m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) + (m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) * m.nc[
                    it, jt, ix, jx, 'c'] * m.rhos / exp(m.wKe2c[it, jt, ix, jx]))))
            else:
                return Constraint.Skip

        self.wk2c_ = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=wk2c_rule)
        self.a71 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a71_rulex)

        ###
        self.wk3c = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.0, bounds=(-100, 5))  #: dummy

        def wk3c_rule(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.k3c[it, jt, ix, jx]) == (exp(m.wk3c[it, jt, ix, jx]))
            else:
                return Constraint.Skip

        def a72_rulex(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.r3c[it, jt, ix, jx]) == \
                       (exp(m.wk3c[it, jt, ix, jx]) * (((1 - 2 * (m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) - (m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) ** 2) * ((m.P[it, jt, ix, jx] * exp(m.wyc[it, jt, ix, jx, 'c']) * 1E5) ** m.m1) - ((m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) * ((m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) + (m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) /exp(m.wKe3c[it, jt, ix, jx]))))
            else:
                return Constraint.Skip

        self.wk3c_ = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=wk3c_rule)
        self.a72 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a72_rulex)

        self.a73 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a73_rule)

        ###
        self.wk2e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.0, bounds=(-100, 5))  #: dummy

        def wk2e_rule(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.k2e[it, jt, ix, jx]) == (exp(m.wk2e[it, jt, ix, jx]))
            else:
                return Constraint.Skip

        def a74_rulex(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.r2e[it, jt, ix, jx]) == (exp(m.wk2e[it, jt, ix, jx]) * ((1. - 2. * (m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) - (
                       m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) * m.ne[it, jt, ix, jx, 'h'] * m.rhos * (
                        m.P[it, jt, ix, jx] * exp(m.wye[it, jt, ix, jx, 'c']) * 1E5) - (((m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) + (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) *
                        m.ne[it, jt, ix, jx, 'c'] * m.rhos / exp(m.wKe2e[it, jt, ix, jx]))))
            else:
                return Constraint.Skip

        self.wk2e_ = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=wk2e_rule)
        self.a74 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a74_rulex)
        ###

        self.wk3e = Var(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, initialize=1.0, bounds=(-100, 5))  #: dummy

        def wk3e_rule(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.k3e[it, jt, ix, jx]) == (exp(m.wk3e[it, jt, ix, jx]))
            else:
                return Constraint.Skip

        def a75_rulex(m, it, jt, ix, jx):
            if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
                return (m.r3e[it, jt, ix, jx]) == (exp(m.wk3e[it, jt, ix, jx]) * (((1. - 2. * (m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) - (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) ** 2) * (((m.P[it, jt, ix, jx] * exp(m.wye[it, jt, ix, jx, 'c'])) ** m.m1) * (1E5 ** m.m1)) - ((m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) * ((m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) + (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) /exp(m.wKe3e[it, jt, ix, jx]))))
            else:
                return Constraint.Skip

        self.wk3e_ = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=wk3e_rule)
        self.a75 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a75_rulex)

        self.a59 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a59_rule)
        self.a60 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a60_rule)
        self.a61 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a61_rule)
        self.a62 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a62_rule)
        self.a63 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a63_rule)
        self.a64 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a64_rule)
        self.a65 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a65_rule)
        self.a66 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a66_rule)
        self.a67 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a67_rule)
        self.a68 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a68_rule)
        self.a69 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a69_rule)
        self.a70 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a70_rule)


        self.a76 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a76_rule)
        self.a77 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a77_rule)
        self.a78 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a78_rule)
        self.a79 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a79_rule)
        self.a80 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a80_rule)
        self.a81 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a81_rule)
        self.a82 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a82_rule)
        self.a83 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a83_rule)
        # self.a84 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a84_rule)
        # self.a85 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a85_rule)
        self.a86 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a86_rule)
        self.a87 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a87_rule)
        self.a88 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a88_rule)
        self.a89 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=a89_rule)

        self.i1 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=i1_rule)
        self.i2 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=i2_rule)
        self.i3 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, m.sp, rule=i3_rule)
        self.i4 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=i4_rule)
        self.i5 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=i5_rule)
        self.i6 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=i6_rule)
        self.i7 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=i7_rule)
        self.i8 = Constraint(m.fe_t, m.cp_ta, m.fe_x, m.cp_x, rule=i8_rule)
        self.e1 = Constraint(m.fe_t, m.cp_ta, rule=e1_rule)


        self.e5 = Constraint(m.fe_t, m.cp_ta, rule=e5_rule)
        # self.e7 = Constraint(m.fe_t, m.cp_ta, rule=e7_rule)
        # self.e8 = Constraint(m.fe_t, m.cp_ta, rule=e8_rule)
        # self.e9 = Constraint(m.fe_t, m.cp_ta, rule=e9_rule)
        # self.e10 = Constraint(m.fe_t, m.cp_ta, m.sp, rule=e10_rule)
        # self.x_3 = Constraint(m.fe_t, m.cp_ta, rule=x_3_rule)
        self.e12s = Constraint(m.fe_t, m.cp_ta, rule=e12_rule)
        self.e13 = Constraint(m.fe_t, m.cp_ta, rule=e13_rule)
        self.e14 = Constraint(m.fe_t, m.cp_ta, m.sp, rule=e14_rule)
        # self.e15 = Constraint(m.fe_t, m.cp_ta, rule=e15_rule)
        # self.e16 = Constraint(m.fe_t, m.cp_ta, rule=e16_rule)

        def e20_rule_alternative(m, it, jt):
            if 0 < jt <= m.ncp_t:
                return (m.Sit[it] - m.Sot[it, jt]) == 0.0
            else:
                return Constraint.Skip

        self.e20 = Constraint(m.fe_t, m.cp_ta, rule=e20_rule_alternative)  #: BC

        #: Last cp point equation (albegraic-bvp)
        self.yl_hse = Constraint(m.fe_t, m.cp_ta, rule=fyl_hse)
        self.yl_ne = Constraint(m.fe_t, m.cp_ta, m.sp, rule=fyl_ne)

        #: BC at the bottom
        self.cc_def = Constraint(m.fe_t, m.cp_ta, rule=cc_rule)

        # --------------------------------------------------------------------------------------------------------------
        #: Control constraint
        self.u1_e = Expression(m.fe_t, rule=lambda m, i: m.Sit[i])
        # self.u2_e = Expression(m.fe_t, rule=lambda m, i: 0.0)

        self.u1_c = Constraint(m.fe_t, rule=lambda m, i: self.u1[i] == self.u1_e[i])
        # self.u2_c = Constraint(m.fe_t, rule=lambda m, i: self.u2[i] == self.u2_e[i])
        # --------------------------------------------------------------------------------------------------------------
        self.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

    def write_nl(self, **kwargs):
        """Writes the nl file and the respective row & col"""
        name = kwargs.pop("name", str(self.__class__.__name__) + ".nl")
        print("NL file written!" + name, file=sys.stderr)
        # name = str(self.__class__.__name__) + ".nl"
        self.write(filename=name,
                   format=ProblemFormat.nl,
                   # io_options={"symbolic_solver_labels": True, "file_determinism": 2})
                   io_options={"symbolic_solver_labels": True})


    def create_bounds(self):
        print('problem with bounds activated')
        for it in range(0, m.nfe_t):
            for jt in range(1, ncp_t + 1):
                # self.GasOut_P[it, jt].setlb(0)
                #self.GasOut_F[it, jt].setlb(0)
                #self.GasOut_T[it, jt].setlb(0)
                # self.SolidIn_Fm[it, jt].setlb(0)
                # self.SolidIn_P[it, jt].setlb(0)
                # self.SolidOut_P[it, jt].setlb(0)
                # self.rhog_in[it, jt].setlb(0)
                # self.rhog_out[it, jt].setlb(0)
                # self.DownOut_P[it, jt].setlb(0)
                # self.h_downcomer[it, jt].setlb(0)
                # self.vmf[it, jt].setlb(0)
                self.db0[it, jt].setlb(1e-05)
                # self.Sit[it].setlb(1e-05)
                self.Sot[it, jt].setlb(0)
                self.g1[it, jt].setlb(1e-05)
                # self.P_l[it, jt].setlb(0)
                self.Phx_l[it, jt].setlb(0)
                self.c_capture[it, jt].setlb(0)
        for it in range(0, m.nfe_t):
            for jt in range(1, ncp_t + 1):
                for cx in ['c', 'h', 'n']:
                    #self.GasOut_z[it, jt, cx].setlb(0)
                    self.cein_l[it, jt, cx].setlb(0)
                    self.ccwin_l[it, jt, cx].setlb(0)
                    self.ne_l[it, jt, cx].setlb(0)

        for it in range(0, m.nfe_t):
            for jt in range(1, ncp_t + 1):
                for ix in range(0, nfe_x):
                    for jx in range(0, ncp_x + 1):
                        self.Ar[it, jt, ix, jx].setlb(0)
                        self.db[it, jt, ix, jx].setlb(0)
                        self.dbe[it, jt, ix, jx].setlb(1e-05)
                        self.dbm[it, jt, ix, jx].setlb(1e-05)
                        self.dbu[it, jt, ix, jx].setlb(1e-05)
                        self.delta[it, jt, ix, jx].setlb(0)
                        self.e[it, jt, ix, jx].setlb(0)
                        self.ed[it, jt, ix, jx].setlb(1e-08)
                        self.fb[it, jt, ix, jx].setlb(0)
                        self.fc[it, jt, ix, jx].setlb(0)
                        self.fcw[it, jt, ix, jx].setlb(0)
                        self.fn[it, jt, ix, jx].setlb(0)
                        self.g2[it, jt, ix, jx].setlb(1e-05)
                        self.g3[it, jt, ix, jx].setlb(1e-05)
                        self.Gb[it, jt, ix, jx].setlb(0)
                        self.Hbc[it, jt, ix, jx].setlb(0)
                        self.Hce[it, jt, ix, jx].setlb(0)
                        self.hd[it, jt, ix, jx].setlb(0)
                        # self.hl[it, jt, ix, jx].setlb(0)
                        # self.hp[it, jt, ix, jx].setlb(0)
                        self.ht[it, jt, ix, jx].setlb(0)
                        # self.Jc[it, jt, ix, jx].setlb(0)
                        # self.Je[it, jt, ix, jx].setlb(0)
                        self.k1c[it, jt, ix, jx].setlb(1e-08)
                        self.k1e[it, jt, ix, jx].setlb(1e-08)

                        # self.k2c[it, jt, ix, jx].setlb(1e-08)
                        # self.k2e[it, jt, ix, jx].setlb(1e-08)

                        # self.k3c[it, jt, ix, jx].setlb(1e-08)
                        # self.k3e[it, jt, ix, jx].setlb(1e-08)
                        self.db[it, jt, ix, jx].setlb(0)
                        self.dbe[it, jt, ix, jx].setlb(0)
                        self.dbm[it, jt, ix, jx].setlb(0)
                        self.dbu[it, jt, ix, jx].setlb(0)
                        self.delta[it, jt, ix, jx].setlb(0)
                        self.e[it, jt, ix, jx].setlb(0)
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
                        # self.hl[it, jt, ix, jx].setlb(0)
                        # self.hp[it, jt, ix, jx].setlb(0)
                        self.ht[it, jt, ix, jx].setlb(0)
                        # self.k1c[it, jt, ix, jx].setlb(0)
                        # self.k1e[it, jt, ix, jx].setlb(0)
                        # self.k2c[it, jt, ix, jx].setlb(0)
                        # self.k2e[it, jt, ix, jx].setlb(0)
                        # self.k3c[it, jt, ix, jx].setlb(0)
                        # self.k3e[it, jt, ix, jx].setlb(0)
                        self.Kcebs[it, jt, ix, jx].setlb(0)
                        self.Ke1c[it, jt, ix, jx].setlb(0)
                        self.Ke1e[it, jt, ix, jx].setlb(0)
                        # self.Ke2c[it, jt, ix, jx].setlb(0)
                        # self.Ke2e[it, jt, ix, jx].setlb(0)
                        # self.Ke3c[it, jt, ix, jx].setlb(0)
                        # self.Ke3e[it, jt, ix, jx].setlb(0)
                        self.kpa[it, jt, ix, jx].setlb(0)
                        self.Nup[it, jt, ix, jx].setlb(-1)
                        self.P[it, jt, ix, jx].setlb(1e-02)
                        self.Phx[it, jt, ix, jx].setlb(0)
                        self.Red[it, jt, ix, jx].setlb(0)
                        self.rhog[it, jt, ix, jx].setlb(0)
                        self.tau[it, jt, ix, jx].setlb(0)
                        # self.Tgb[it, jt, ix, jx].setlb(0)
                        # self.Tgc[it, jt, ix, jx].setlb(0)
                        # self.Tge[it, jt, ix, jx].setlb(0)
                        # self.Thx[it, jt, ix, jx].setlb(0)
                        self.Tsc[it, jt, ix, jx].setlb(1e-08)
                        self.Tse[it, jt, ix, jx].setlb(1e-08)
                        # self.Ttube[it, jt, ix, jx].setlb(0)
                        self.vb[it, jt, ix, jx].setlb(1e-08)
                        self.vbr[it, jt, ix, jx].setlb(0)
                        #self.ve[it, jt, ix, jx].setlb(1e-08)
                        self.vg[it, jt, ix, jx].setlb(0)
        for it in range(0, m.nfe_t):
            for jt in range(1, ncp_t + 1):
                for ix in range(0, nfe_x):
                    for jx in range(0, ncp_x + 1):
                        for cx in ['c', 'h', 'n']:
                            # self.cb[it, jt, ix, jx, cx].setlb(0)
                            self.ccwin[it, jt, ix, jx, cx].setlb(0)
                            # self.ce[it, jt, ix, jx, cx].setlb(0)
                            self.cein[it, jt, ix, jx, cx].setlb(0)
                            self.D[it, jt, ix, jx, cx].setlb(1e-08)
                            self.Kbc[it, jt, ix, jx, cx].setlb(0)
                            self.Kce[it, jt, ix, jx, cx].setlb(0)

                            self.nc[it, jt, ix, jx, cx].setlb(0)
                            self.ne[it, jt, ix, jx, cx].setlb(0)

                            # self.yb[it, jt, ix, jx, cx].setlb(1e-08)
                            # self.yc[it, jt, ix, jx, cx].setlb(1e-08)
                            #self.ye[it, jt, ix, jx, cx].setlb(1e-07)

    def clear_bounds(self):
        """Sets bounds of variables
        Args:
            None
        Returns:
            None"""
        for var in self.component_data_objects(Var):
            var.setlb(None)
            var.setub(None)

    def init_steady_ref(self, snap_shot=True, verb=False):
        """If the model is steady, we try to initialize it with an initial guess from ampl"""


        for var in self.component_data_objects(Var):
            try:
                if snap_shot:
                    var.set_value(self.dref[var.parent_component().name, var.index()])
                else:
                    var.set_value(ss[var.parent_component().name, var.index()])
            except KeyError:
                if verb:
                    print(var.name,"\t",str(var.index()),"\tnot found")
                pass

        if nfe_x > 4:
            for var in self.component_objects(Var):
                for ks in var.keys():
                    if type(ks) != tuple:
                        break
                    if len(ks) >= 4:
                        if ks[2] > 4:
                            var[ks].set_value(value(var[(m.nfe_t, ncp_t, 1, ncp_x) + ks[4:]]))
                            # print(var[ks].value)

        if m.nfe_t == 1 and ncp_t == 1:
            solver = SolverFactory('asl:ipopt')
            solver.options["print_user_options"] = 'yes'
            with open("ipopt.opt", "w") as f:
                f.write("max_iter 300\n")
                f.write("start_with_resto yes\n")
                # f.write("bound_push 1e-08\n")
                f.write("print_info_string yes\n")
                f.close()
            # solver.options['halt_on_ampl_error'] = "yes"
            self.create_bounds()
            print("Number of variables\t", self.nvariables(),end="\t")
            print("Number of constraints\t", self.nconstraints())
            self.write(filename="01.nl",
                       format=ProblemFormat.nl, io_options={"symbolic_solver_labels": True})

            someresults = solver.solve(self, tee=True, symbolic_solver_labels=True)
            self.solutions.load_from(someresults)

            with open("ipopt.opt", "w") as f:
                f.write("max_iter 300\n")
                # f.write("mu_init 1e-08\n")
                f.write("max_cpu_time 120\n")
                f.write("expect_infeasible_problem yes\n")
                f.write("print_info_string yes\n")
                f.close()
            # self.clear_bounds()
            someresults = solver.solve(self, tee=True, symbolic_solver_labels=True)
            self.solutions.load_from(someresults)
            with open("tidyup.txt", "w") as f:
                for con in self.component_data_objects(Constraint, active=True):
                    vinf = value(con.body)
                    vs = str(vinf)
                    f.write(vs)
                    f.write("\n")
                f.close()

            # self.display()
            self.create_bounds()
            with open("ipopt.opt", "w") as f:
                f.write("linear_solver ma57\n")
                f.write("max_iter 20\n")
                f.close()
            someresults = solver.solve(self, tee=True, symbolic_solver_labels=True)
            self.solutions.load_from(someresults)
            # self.display(filename="here_son.txt")
            # sys.exit()
            with open("ipopt.opt", "w") as f:
                f.write("max_iter 300\n")
                f.write("max_cpu_time 60\n")
                f.write("start_with_resto yes\n")
                # f.write("bound_push 1e-08\n")
                f.write("print_info_string yes\n")
                f.close()
            someresults = solver.solve(self, tee=True, symbolic_solver_labels=True)
            self.solutions.load_from(someresults)

            with open("ipopt.opt", "w") as f:
                f.write("max_iter 300\n")
                f.write("max_cpu_time 200\n")
                f.write("linear_scaling_on_demand yes\n")

                f.write("print_info_string yes\n")
                f.close()
            someresults = solver.solve(self, tee=True, symbolic_solver_labels=True)
            self.solutions.load_from(someresults)

            # self.display(filename="here_son.txt")
            with open("ipopt.opt", "w") as f:
                # f.write("max_iter 300\n")
                f.write("max_cpu_time 6000\n")
                # f.write("linear_scaling_on_demand yes\n")
                # f.write("file_print_level 6\n")
                f.write("output_file \"testing.txt\"\n")
                f.write("print_info_string yes\n")
                f.write("linear_solver ma57\n")
                f.write("resto.dual_inf_tol 1e-06\n")
                f.write("resto.acceptable_dual_inf_tol 1e-06\n")
                # f.write("resto.constr_viol_tol 1e-06\n")
                # f.write("resto.acceptable_constr_viol_tol 1e-06\n")
                f.write("gamma_phi 1e-08\n")
                f.write("gamma_theta 1e-08\n")
                # f.write("resto.max_iter 1\n")
                f.close()

            self.create_bounds()



            someresults = solver.solve(self, tee=True, symbolic_solver_labels=True) # go into restoration minimize infes
            self.write_nl(name="active.nl")
            self.snap_shot(filename="fbound.py")
            with open("mult_bounds.txt", "w") as f:
                self.ipopt_zL_out.pprint(ostream=f)
                f.close()
            # self.ipopt_zL_out.pprint()
            # self.ipopt_zU_out.pprint()
            self.clear_bounds()
            someresults = solver.solve(self, tee=True, symbolic_solver_labels=True)
            self.display(filename="whatevs")
            # self.dumm = Var()
            self.write_nl(name="nonactive.nl")
            self.snap_shot(filename="fnbound.py")


            # sys.exit()


    def equalize_u(self, direction="u_to_r"):
        """set current controls to the values of their respective dummies"""
        if direction == "u_to_r":
            for i in self.Sit.keys():
                self.Sit[i].set_value(value(self.u1[i]))
            # for i in self.per_opening2.keys():
            #     self.per_opening2[i].set_value(value(self.u2[i]))
        elif direction == "r_to_u":
            for i in self.u1.keys():
                self.u1[i].value = value(self.Sit[i])
            # for i in self.u2.keys():
            #     self.u2[i].value = value(self.per_opening2[i])
    def snap_shot(self, filename="snap_shot.py"):
        k0 = ((),)
        with open(filename, "w") as f:
            f.write("snap = {}\n")
            for var in self.component_objects(Var, active=True):
                sv = var.name
                for key in var.iterkeys():
                    if key is None:
                        print(sv, key, type(key))
                        val = value(var)
                        f.write("snap[\'" + sv + "\'," + str(k0) + "] = " + str(val))
                    else:
                        val = value(var[key])
                        if type(key) == str:
                            key = tuple(key)
                        k = key
                        f.write("snap[\'" + sv + "\'," + str(k) + "] = " + str(val))
                    f.write("\n")
            f.close()

    def infes_test(self, reset_option=True):
        if reset_option:
            with open("ipopt.opt", "w") as f:
                # f.write("max_iter 300\n")
                # f.write("mu_init 1e-08\n")
                f.write("max_cpu_time 600\n")
                f.write("print_user_options yes\n")
                f.write("expect_infeasible_problem yes\n")
                f.write("linear_solver ma57\n")
                f.write("ma57_automatic_scaling #yes\n")
                f.write("print_info_string yes\n")
                f.close()
        solver = SolverFactory('asl:ipopt')
        someresults = solver.solve(self, tee=True, symbolic_solver_labels=True)
        self.solutions.load_from(someresults)

    def report_zL(self, filename="bounds_default.txt"):
        print("Bounds values written" + filename, file=sys.stderr)
        with open(filename, "w") as f:
            self.ipopt_zL_out.pprint(ostream=f)
            self.ipopt_zL_in.pprint(ostream=f)
            self.ipopt_zL_out.display(ostream=f)
            f.close()
