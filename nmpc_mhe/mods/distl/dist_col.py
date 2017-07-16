#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from pyomo.core.base import ConcreteModel, Set, Constraint, Var,\
    Param, Objective, minimize, sqrt, exp, Suffix
from bfbDAE.aux_bfb.ocpfe.cpoinsc import collptsgen
from bfbDAE.aux_bfb.ocpfe.lagrange_f import lgr, lgry, lgrdot, lgrydot
from six import itervalues, iterkeys

"""
Version 03.
I thought of doing this in a class, but I couldn't find a really motivating reason to do it. 
"""

__author__ = 'David M Thierry @dthierry'


class DistDiehlNegrete(object):
    def __init__(self, nfe_t, ncp_t, **kwargs):
        steady = kwargs.pop('steady', False)
        self._t = kwargs.pop('_t', 1.0)

        self._alp_gauB_t = 1
        self._bet_gauB_t = 0

        if steady:
            print("[I] " + str(self.__class__.__name__) + " NFE and NCP Overriden - Steady state mode")
            self.nfe_t = 1
            self.ncp_t = 1
        else:
            self.nfe_t = nfe_t
            self.ncp_t = ncp_t

        self.tau_t = collptsgen(ncp_t, self._alp_gauB_t, self._bet_gauB_t)

        # start at zero
        self.tau_i_t = {0: 0.}
        # create a list

        for ii in range(1, self.ncp_t + 1):
            self.tau_i_t[ii] = self.tau_t[ii - 1]

        # ======= SETS ======= #
        # For finite element = 1 .. K
        # This has to be > 0

        self.fe_t = Set(initialize=[ii for ii in range(1, self.nfe_t + 1)])

        # collocation points
        # collocation points for diferential variables
        self.cp_t = Set(initialize=[ii for ii in range(0, self.ncp_t + 1)])
        self.cp_ta = Set(within=self.cp_t, initialize=[ii for ii in range(1, self.ncp_t + 1)])

        # components
        self.sp = Set(initialize=['c', 'h', 'n'])

        # create collocation param

        self.taucp_t = Param(self.cp_t, initialize=self.tau_i_t)

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



def dist_col_rodrigo(nfe_t, ncp_t, hi_t, kind, init0=True, bnd_set=True,
                     states={}, controls=[], noisy_states={} ,measurements={}, objf_kind=None):
    # collocation polynomial parameters
    _alp_gauB_t = 1
    _bet_gauB_t = 0
    # polynomial roots (Radau)
    tau_t = collptsgen(ncp_t, _alp_gauB_t, _bet_gauB_t)
    tau_i_t = {0: 0.}
    # create a dictionary with the collocation points starting 0
    for ii in range(1, ncp_t + 1):
        tau_i_t[ii] = tau_t[ii - 1]
    # print(tau_i_t)
    m = ConcreteModel()

    m.i_flag = init0
    m.bnd_set = bnd_set

    # set of finite elements and collocation points
    if kind == 'steady' or kind == 'steady_control':
        print("STEADY_STATE PROBLEM")
        print("-"*120)
        m.cp = Set(initialize=[1])
        m.fe = Set(initialize=[1])
    else:
        print("DYNAMIC_PROBLEM", end="")
        print("\tMHE\n") if kind == 'mhe' else print("\n")
        print("-" * 120)
        m.fe = Set(initialize=[i for i in range(1, nfe_t + 1)])
        m.cp = Set(initialize=[i for i in range(0, ncp_t + 1)])

    if bnd_set:
        print("Bounds_set")

    Ntray = 42

    m.Ntray = Ntray
    m.nfe_t = nfe_t
    m.ncp_t = ncp_t

    m.tray = Set(initialize=[i for i in range(1, Ntray + 1)])

    def __init_feed(m, t):
        if t == 21:
            return 57.5294
        else:
            return 0

    m.feed = Param(m.tray, initialize=__init_feed, mutable=True)

    m.xf = Param(initialize=0.32, mutable=True) # feed mole fraction
    m.hf = Param(initialize=9081.3) # feed enthalpy

    m.hlm0 = Param(initialize=2.6786e-04)
    m.hlma = Param(initialize=-0.14779)
    m.hlmb = Param(initialize=97.4289)
    m.hlmc = Param(initialize=-2.1045e04)

    m.hln0 = Param(initialize=4.0449e-04)
    m.hlna = Param(initialize=-0.1435)
    m.hlnb = Param(initialize=121.7981)
    m.hlnc = Param(initialize=-3.0718e04)

    m.r = Param(initialize=8.3147)
    m.a = Param(initialize=6.09648)
    m.b = Param(initialize=1.28862)
    m.c1 = Param(initialize=1.016)
    m.d = Param(initialize=15.6875)
    m.l = Param(initialize=13.4721)
    m.f = Param(initialize=2.615)

    m.gm = Param(initialize=0.557)
    m.Tkm = Param(initialize=512.6)
    m.Pkm = Param(initialize=8.096e06)

    m.gn = Param(initialize=0.612)
    m.Tkn = Param(initialize=536.7)
    m.Pkn = Param(initialize=5.166e06)

    m.CapAm = Param(initialize=23.48)
    m.CapBm = Param(initialize=3626.6)
    m.CapCm = Param(initialize=-34.29)

    m.CapAn = Param(initialize=22.437)
    m.CapBn = Param(initialize=3166.64)
    m.CapCn = Param(initialize=-80.15)

    m.pstrip = Param(initialize=250)
    m.prect = Param(initialize=190)

    def _p_init(m, t):
        ptray = 9.39e04
        if t <= 20:
            return _p_init(m, 21) + m.pstrip * (21 - t)
        elif 20 < t < m.Ntray:
            return ptray + m.prect * (m.Ntray - t)
        elif t == m.Ntray:
            return 9.39e04

    m.p = Param(m.tray, initialize=_p_init)

    m.T29_des = Param(initialize=343.15)
    m.T15_des = Param(initialize=361.15)
    m.Dset = Param(initialize=1.83728)
    m.Qcset = Param(initialize=1.618890)
    m.Qrset = Param(initialize=1.786050)
    m.Recset = Param()

    m.alpha_T29 = Param(initialize=1)
    m.alpha_T15 = Param(initialize=1)
    m.alpha_D = Param(initialize=1)
    m.alpha_Qc = Param(initialize=1)
    m.alpha_Qr = Param(initialize=1)
    m.alpha_Rec = Param(initialize=1)

    def _alpha_init(m, i):
        if i <= 21:
            return 0.62
        else:
            return 0.35

    def _dotted_ss(m, ssflag):
        if ssflag == "steady" or kind == 'steady_control':
            return Param(m.fe, m.cp, m.tray, initialize=0.0)
        else:
            return Var(m.fe, m.cp, m.tray, initialize=0.0)

    m.alpha = Param(m.tray, initialize=_alpha_init)

    m.M_pred = Param(m.tray, initialize=0.0, mutable=True)
    m.x_pred = Param(m.tray, initialize=0.0, mutable=True)

    def __m_init(m, i, j, t):
        if m.i_flag:
            if t < m.Ntray:
                return 4000.
            elif t == 1:
                return 104340.
            elif t == m.Ntray:
                return 5000.
        else:
            return 0.

    # Liquid hold-up
    m.M = Var(m.fe, m.cp, m.tray, initialize=__m_init)
    m.Mdot = _dotted_ss(m, kind)
    # m.M_0 = Var(m.fe, m.tray, initialize=1e07)

    def __t_init(m, i, j, t):
        if m.i_flag:
            return ((370.781 - 335.753)/m.Ntray)*t + 370.781
        else:
            return 10.

    # Tray temperature
    m.T = Var(m.fe, m.cp, m.tray, initialize=__t_init)
    m.Tdot = Var(m.fe, m.cp, m.tray, initialize=0.0)

    # saturation pressures
    m.pm = Var(m.fe, m.cp, m.tray, initialize=1e4)
    m.pn = Var(m.fe, m.cp, m.tray, initialize=1e4)

    # define l-v flowrate

    def _v_init(m, i, j, t):
        if m.i_flag:
            return 44.
        else:
            return 0.

    # Vapor mole flowrate
    m.V = Var(m.fe, m.cp, m.tray, initialize=_v_init)

    def _l_init(m, i, j, t):
        if m.i_flag:
            if 2 <= t <= 21:
                return 83.
            elif 22 <= t <= 42:
                return 23
            elif t == 1:
                return 40
        else:
            return 0.

    # Liquid mole flowrate
    m.L = Var(m.fe, m.cp, m.tray, initialize=_l_init)

    # mol frac l-v

    def __x_init(m, i, j, t):
        if m.i_flag:
            return (0.999/m.Ntray)*t
        else:
            return 1

    # Liquid mole frac & diff var
    m.x = Var(m.fe, m.cp, m.tray, initialize=__x_init)

    m.xdot = _dotted_ss(m, kind)
    #m.x_0 = Var(m.fe, m.tray)

    # av

    def __y_init(m, i, j, t):
        if m.i_flag:
            return ((0.99-0.005)/m.Ntray)*t + 0.005
        else:
            return 1

    # Vapor mole frac & diff var
    m.y = Var(m.fe, m.cp, m.tray, initialize=__y_init)

    # Liquid enthalpy    # enthalpy
    m.hl = Var(m.fe, m.cp, m.tray, initialize=10000.)

    def __hv_init(m, i, j, t):
        if m.i_flag:
            if t < m.Ntray:
                return 5e4
        else:
            return 0.0

    # Liquid enthalpy    # enthalpy
    m.hv = Var(m.fe, m.cp, m.tray, initialize=__hv_init)
    # reboiler & condenser heat
    m.Qc = Var(m.fe, m.cp, initialize=1.6e06)
    m.D = Var(m.fe, m.cp, initialize=18.33)
    # vol holdups
    m.Vm = Var(m.fe, m.cp, m.tray, initialize=6e-05)

    def __mv_init(m, i, j, t):
        if m.i_flag:
            if 1 < t < m.Ntray:
                return 0.23
        else:
            return 0.0

    m.Mv = Var(m.fe, m.cp, m.tray, initialize=__mv_init)

    m.Mv1 = Var(m.fe, m.cp, initialize=8.57)
    m.Mvn = Var(m.fe, m.cp, initialize=0.203)

    # Process-noise is optional
    m.all_d_states = ["M", "x"]
    # case - dynamic sim and noisy -> disaster
    m.w_M = Var(m.fe, m.tray, initialize=0.0) if "M" in noisy_states.keys() else 0.0
    m.w_x = Var(m.fe, m.tray, initialize=0.0) if "x" in noisy_states.keys() else 0.0

    # Dictionary for process disturbance expression.
    m.W = {}
    d_state_list = []
    # d_state_list = [i for j in map(lambda vv: vv.iterkeys(), map(lambda v: getattr(m, v), m.all_d_states)) for i in j]
    for i in m.all_d_states:
        var = getattr(m, i)
        l = [j for j in iterkeys(var)]
        d_state_list += map(lambda v: (i,  v), l)

    m.W = m.W.fromkeys(d_state_list, 0.0)

    m.noisy_f = {}

    # Filter for w term in the continuation equations
    if noisy_states.__len__() > 0:
        # print("noisy > 1")
        for i in noisy_states.keys():
            idxs = noisy_states[i]
            var = getattr(m, "w_" + i)
            for j in idxs:
                m.noisy_f[(i, j)] = True
                for i_fe in range(1, nfe_t + 1):
                    idx = (i_fe, 0) + j
                    m.W[i, idx] = var[i_fe, j]

    def _bound_set_lb(m):
        if m.bnd_set:
            for value in itervalues(m.M):
                value.setlb(1.0)
            for value in itervalues(m.T):
                value.setlb(200)
            for value in itervalues(m.pm):
                value.setlb(1.0)
            for value in itervalues(m.pn):
                value.setlb(1.0)
            for value in itervalues(m.L):
                value.setlb(0.0)
            for value in itervalues(m.V):
                value.setlb(0.0)

            for value in itervalues(m.x):
                value.setlb(0.0)
            for value in itervalues(m.y):
                value.setlb(0.0)

            for value in itervalues(m.hl):
                value.setlb(1.0)
            for value in itervalues(m.hv):
                value.setlb(1.0)

            for value in itervalues(m.Qc):
                value.setlb(0.0)
            for value in itervalues(m.D):
                value.setlb(0.0)

            for value in itervalues(m.Vm):
                value.setlb(0.0)
            for value in itervalues(m.Mv):
                value.setlb(0.155 + 1e-06)
            for value in itervalues(m.Mv1):
                value.setlb(8.5 + 1e-06)
            for value in itervalues(m.Mvn):
                value.setlb(0.17 + 1e-06)
        else:
            pass

    def _bound_set_ub(m):
        if m.bnd_set:
            for value in itervalues(m.M):
                value.setub(1e+07)
            for value in itervalues(m.T):
                value.setub(500)
            for value in itervalues(m.pm):
                value.setub(5e+07)
            for value in itervalues(m.pn):
                value.setub(5e+07)
            for value in itervalues(m.L):
                value.setub(1e+03)
            for value in itervalues(m.V):
                value.setub(1e+03)

            for value in itervalues(m.x):
                value.setub(1.0)
            for value in itervalues(m.y):
                value.setub(1.0)

            for value in itervalues(m.hl):
                value.setub(1e+07)
            for value in itervalues(m.hv):
                value.setub(1e+07)

            for value in itervalues(m.Qc):
                value.setub(1e+08)
            for value in itervalues(m.D):
                value.setub(1e+04)

            for value in itervalues(m.Vm):
                value.setub(1e+04)
            for value in itervalues(m.Mv):
                value.setub(1e+04)
            for value in itervalues(m.Mv1):
                value.setub(1e+04)
            for value in itervalues(m.Mvn):
                value.setub(1e+04)

        else:
            pass

    _bound_set_lb(m)
    _bound_set_ub(m)

    # Controls
    def _controls(m, kind_prb, kind_var):
        if kind_var == "Rec":
            i_val = 7.72700925775773761472464684629813E-01
            bnd = (0.00001, 9.9999e-01)
        elif kind_var == "Qr":
            i_val = 1.78604740940007800236344337463379E+06
            bnd = (0.00, 1e+08)
        else:
            i_val = 0.0
        if kind_prb == 'nmpc' or kind_prb == 'steady_control':
            return Var(m.fe, initialize=i_val, bounds=bnd)
        else:
            return Param(m.fe, initialize=i_val, mutable=True)

    m.Rec = _controls(m, kind, "Rec")
    m.Qr = _controls(m, kind, "Qr")
    # Controls

    m.hi = [1., 1.] if kind == 'steady' or kind == 'steady_control' else Param(m.fe, initialize=hi_t)

    # mass balances
    def _modetr(m, i, j, k):
        if j > 0 and 1 < k < Ntray:
            return m.Mdot[i, j, k] == \
                   (m.V[i, j, k - 1] - m.V[i, j, k] + m.L[i, j, k + 1] - m.L[i, j, k] + m.feed[k]) * m.hi[i]
        else:
            return Constraint.Skip

    m.MODEtr = Constraint(m.fe, m.cp, m.tray, rule=_modetr)

    def _moder(m, i, j):
        if j > 0:
            return m.Mdot[i, j, 1] == \
                   (m.L[i, j, 2] - m.L[i, j, 1] - m.V[i, j, 1]) * m.hi[i]
        else:
            return Constraint.Skip

    m.MODEr = Constraint(m.fe, m.cp, rule=_moder)

    def _MODEc(m, i, j):
        if j > 0:
            return m.Mdot[i, j, Ntray] == \
                   (m.V[i, j, Ntray - 1] - m.L[i, j, Ntray] - m.D[i, j]) * m.hi[i]
        else:
            return Constraint.Skip

    m.MODEc = Constraint(m.fe, m.cp, rule=_MODEc)

    def _MCOLL(m, i, j, t):
        if j > 0:
            return m.Mdot[i, j, t] == \
                   sum(lgrdot(k, tau_i_t[j], ncp_t, _alp_gauB_t, _bet_gauB_t) * m.M[i, k, t] for k in m.cp)
        else:
            return Constraint.Skip

    m.MCOLL = None if kind == 'steady' or kind == 'steady_control' else Constraint(m.fe, m.cp, m.tray, rule=_MCOLL)

    ## !!!!!
    def _M_CONT(m, i, t):
        if i < nfe_t and nfe_t > 1:
            return m.M[i + 1, 0, t] == \
                   sum(lgr(j, 1, ncp_t, _alp_gauB_t, _bet_gauB_t) * m.M[i, j, t] for j in m.cp) + m.W["M", (i, 0, t)]
        else:
            return Constraint.Skip

    m.MCONT = None if kind == 'steady' or kind == 'steady_control' else Constraint(m.fe, m.tray, rule=_M_CONT)

    def _XODEtr(m, i, j, t):
        if j > 0 and 1 < t < Ntray:
            return m.M[i, j, t] * m.xdot[i, j, t] == \
                   (m.V[i, j, t - 1] * (m.y[i, j, t - 1] - m.x[i, j, t]) +
                    m.L[i, j, t + 1] * (m.x[i, j, t + 1] - m.x[i, j, t]) -
                    m.V[i, j, t] * (m.y[i, j, t] - m.x[i, j, t]) +
                    m.feed[t] * (m.xf - m.x[i, j, t])) * m.hi[i]
        else:
            return Constraint.Skip

    m.XODEtr = Constraint(m.fe, m.cp, m.tray, rule=_XODEtr)

    def _xoder(m, i, j):
        if j > 0:
            return m.M[i, j, 1] * m.xdot[i, j, 1] ==\
                   (m.L[i, j, 2] * (m.x[i, j, 2] - m.x[i, j, 1]) -
                    m.V[i, j, 1] * (m.y[i, j, 1] - m.x[i, j, 1])) * m.hi[i]
        else:
            return Constraint.Skip

    m.xoder = Constraint(m.fe, m.cp, rule=_xoder)

    def _xodec(m, i, j):
        if j > 0:
            return m.M[i, j, m.Ntray] * m.xdot[i, j, m.Ntray] == \
                   (m.V[i, j, Ntray - 1] * (m.y[i, j, m.Ntray - 1] - m.x[i, j, m.Ntray])) * m.hi[i]
        else:
            return Constraint.Skip

    m.xodec = Constraint(m.fe, m.cp, rule=_xodec)

    def _xcoll(m, i, j, t):
        if j > 0:
            return m.xdot[i, j, t] == \
                   sum(lgrdot(k, tau_i_t[j], ncp_t, _alp_gauB_t, _bet_gauB_t) * m.x[i, k, t] for k in m.cp)
        else:
            return Constraint.Skip

    m.xcoll = None if kind == 'steady' or kind == 'steady_control' else Constraint(m.fe, m.cp, m.tray, rule=_xcoll)

    def _xcont(m, i, t):
        if i < m.nfe_t and m.nfe_t > 1:
            return m.x[i + 1, 0, t] == \
                   sum(lgr(j, 1, ncp_t, _alp_gauB_t, _bet_gauB_t) * m.x[i, j, t] for j in m.cp) + m.W["x", (i, 0, t)]
        else:
            return Constraint.Skip

    m.xcont = None if kind == 'steady' or kind == 'steady_control' else Constraint(m.fe, m.tray, rule=_xcont)

    def _hrc(m, i, j):
        if j > 0:
            return m.D[i, j] - m.Rec[i]*m.L[i, j, m.Ntray] == 0
        else:
            return Constraint.Skip

    m.hrc = Constraint(m.fe, m.cp, rule=_hrc)

    # Energy balance
    def _gh(m, i, j, t):
        if j > 0 and 1 < t < Ntray:
            return m.M[i, j, t] * (
                m.xdot[i, j, t] * (
                    (m.hlm0 - m.hln0) * (m.T[i, j, t]**3) +
                    (m.hlma - m.hlna) * (m.T[i, j, t]**2) +
                    (m.hlmb - m.hlnb) * m.T[i, j, t] + m.hlmc - m.hlnc) +
                m.hi[i] * m.Tdot[i, j, t] * (
                    3*m.hln0*(m.T[i, j, t]**2) +
                    2*m.hlna * m.T[i, j, t] + m.hlnb +
                    m.x[i, j, t] *
                    (3*(m.hlm0 - m.hln0) * (m.T[i, j, t]**2) + 2 * (m.hlma - m.hlna) * m.T[i, j, t] + m.hlmb - m.hlnb))
            ) == (m.V[i, j, t-1] * (m.hv[i, j, t-1] - m.hl[i, j, t]) +
                  m.L[i, j, t+1] * (m.hl[i, j, t+1] - m.hl[i, j, t]) -
                  m.V[i, j, t] * (m.hv[i, j, t] - m.hl[i, j, t]) +
                  m.feed[t] * (m.hf - m.hl[i, j, t])) * m.hi[i]
        else:
            return Constraint.Skip

    m.gh = Constraint(m.fe, m.cp, m.tray, rule=_gh)

    def _ghb(m, i, j):
        if j > 0:
            return m.M[i, j, 1] * (m.xdot[i, j, 1] * ((m.hlm0 - m.hln0) * m.T[i, j, 1]**3 + (m.hlma - m.hlna)*m.T[i, j, 1]**2 + (m.hlmb - m.hlnb)*m.T[i, j, 1] + m.hlmc - m.hlnc) + m.hi[i] * m.Tdot[i, j, 1] * (3 * m.hln0 * m.T[i, j, 1]**2 + 2 * m.hlna * m.T[i, j, 1] + m.hlnb + m.x[i, j, 1] * (3 * (m.hlm0 - m.hln0) * m.T[i, j, 1]**2 + 2*(m.hlma - m.hlna) * m.T[i, j, 1] + m.hlmb - m.hlnb))) == \
                   (m.L[i, j, 2] * (m.hl[i, j, 2] - m.hl[i, j, 1]) - m.V[i, j, 1] * (m.hv[i, j, 1] - m.hl[i, j, 1]) + m.Qr[i]) * m.hi[i]
        else:
            return Constraint.Skip

    m.ghb = Constraint(m.fe, m.cp, rule=_ghb)

    def _ghc(m, i, j):
        if j > 0:
            return m.M[i, j, Ntray] * (m.xdot[i, j, Ntray] * ((m.hlm0 - m.hln0) * m.T[i, j, Ntray]**3 + (m.hlma - m.hlna) * m.T[i, j, Ntray]**2 + (m.hlmb - m.hlnb) * m.T[i, j, Ntray] + m.hlmc - m.hlnc) + m.hi[i] * m.Tdot[i, j, Ntray] * (3 * m.hln0 * m.T[i, j, Ntray]**2 + 2* m.hlna * m.T[i, j, Ntray] + m.hlnb + m.x[i, j, Ntray] * (3 * (m.hlm0 - m.hln0) * m.T[i, j, Ntray]**2 + 2 * (m.hlma - m.hlna) * m.T[i, j, Ntray] + m.hlmb - m.hlnb))) == \
                   (m.V[i, j, Ntray - 1] * (m.hv[i, j, Ntray - 1] - m.hl[i, j, Ntray]) - m.Qc[i, j]) * m.hi[i]
        else:
            return Constraint.Skip

    m.ghc = Constraint(m.fe, m.cp, rule=_ghc)

    def _hkl(m, i, j, t):
        if j > 0:
            return m.hl[i, j, t] == m.x[i, j, t]*(m.hlm0*m.T[i, j, t]**3 + m.hlma * m.T[i, j, t]**2 + m.hlmb * m.T[i, j, t] + m.hlmc) + (1 - m.x[i, j, t])*(m.hln0 * m.T[i, j, t]**3 + m.hlna*m.T[i, j, t]**2 + m.hlnb * m.T[i, j, t] + m.hlnc)
        else:
            return Constraint.Skip

    m.hkl = Constraint(m.fe, m.cp, m.tray, rule=_hkl)

    def _hkv(m, i, j, t):
        if j > 0 and t < Ntray:
            return m.hv[i, j, t] == m.y[i, j, t] * (m.hlm0 * m.T[i, j, t]**3 + m.hlma * m.T[i, j, t]**2 + m.hlmb * m.T[i, j, t] + m.hlmc + m.r * m.Tkm * sqrt(1 - (m.p[t]/m.Pkm) * (m.Tkm/m.T[i, j, t])**3)*(m.a - m.b * m.T[i, j, t]/m.Tkm + m.c1 * (m.T[i, j , t]/m.Tkm)**7 + m.gm * (m.d - m.l * m.T[i, j, t]/m.Tkm + m.f*(m.T[i, j, t]/m.Tkm)**7 ))) + (1 - m.y[i, j, t]) * (m.hln0 * m.T[i, j, t]**3 + m.hlna * m.T[i, j, t]**2 + m.hlnb * m.T[i, j, t] + m.hlnc + m.r * m.Tkn * sqrt(1 - (m.p[t]/m.Pkn)*(m.Tkn/m.T[i, j, t])**3)*(m.a - m.b * m.T[i, j, t]/m.Tkn + m.c1 * (m.T[i, j, t]/m.Tkn)**7 + m.gn*(m.d - m.l * m.T[i, j, t]/m.Tkn + m.f* (m.T[i, j, t]/m.Tkn)**7)))
        else:
            return Constraint.Skip

    m.hkv = Constraint(m.fe, m.cp, m.tray, rule=_hkv)

    def _lpm(m, i, j, t):
        if j > 0:
            return m.pm[i, j, t] == exp(m.CapAm - m.CapBm/(m.T[i, j, t] + m.CapCm))
        else:
            return Constraint.Skip

    m.lpm = Constraint(m.fe, m.cp, m.tray, rule=_lpm)

    def _lpn(m, i, j, t):
        if j > 0:
            return m.pn[i, j, t] == exp(m.CapAn - m.CapBn/(m.T[i, j, t] + m.CapCn))

        else:
            return Constraint.Skip

    m.lpn = Constraint(m.fe, m.cp, m.tray, rule=_lpn)

    def _dp(m, i, j, t):
        if j > 0:
            return m.p[t] == m.pm[i, j, t] * m.x[i, j, t] + (1 - m.x[i, j, t]) * m.pn[i, j, t]
        else:
            return Constraint.Skip

    m.dp = Constraint(m.fe, m.cp, m.tray, rule=_dp)

    def _lTdot(m, i, j, t):
        if j > 0:
            return m.Tdot[i, j, t] * m.hi[i] ==\
                   -(m.pm[i, j, t] - m.pn[i, j, t]) * m.xdot[i, j, t] / \
                   (m.x[i, j, t] *
                    exp(m.CapAm - m.CapBm/(m.T[i, j, t] + m.CapCm)) * m.CapBm/(m.T[i, j, t] + m.CapCm)**2 +
                    (1 - m.x[i, j, t]) *
                    exp(m.CapAn - m.CapBn/(m.T[i, j, t] + m.CapCn)) * m.CapBn/(m.T[i, j, t] + m.CapCn)**2)
        else:
            return Constraint.Skip

    m.lTdot = Constraint(m.fe, m.cp, m.tray, rule=_lTdot)

    def _gy0(m, i, j):
        if j > 0:
            return m.p[1] * m.y[i, j, 1] == m.x[i, j, 1] * m.pm[i, j, 1]
        else:
            return Constraint.Skip

    m.gy0 = Constraint(m.fe, m.cp, rule=_gy0)

    def _gy(m, i, j, t):
        if j > 0 and 1 < t < Ntray:
            return m.y[i, j, t] == \
                   m.alpha[t] * m.x[i, j, t] * m.pm[i, j, t] / m.p[t] + (1 - m.alpha[t]) * m.y[i, j, t - 1]
        else:
            return Constraint.Skip

    m.gy = Constraint(m.fe, m.cp, m.tray, rule=_gy)

    def _dMV(m, i, j, t):
        if j > 0 and 1 < t < Ntray:
            return m.Mv[i, j, t] == m.Vm[i, j, t] * m.M[i, j, t]
        else:
            return Constraint.Skip

    m.dMV = Constraint(m.fe, m.cp, m.tray, rule=_dMV)

    def _dMv1(m, i, j):
        if j > 0:
            return m.Mv1[i, j] == m.Vm[i, j, 1] * m.M[i, j, 1]
        else:
            return Constraint.Skip

    m.dMv1 = Constraint(m.fe, m.cp, rule=_dMv1)

    def _dMvn(m, i, j):
        if j > 0:
            return m.Mvn[i, j] == m.Vm[i, j, Ntray] * m.M[i, j, Ntray]
        else:
            return Constraint.Skip

    m.dMvn = Constraint(m.fe, m.cp, rule=_dMvn)

    def _hyd(m, i, j, t):
        if j > 0 and 1 < t < Ntray:
            return m.L[i, j, t] * m.Vm[i, j, t] == 0.166 * (m.Mv[i, j, t] - 0.155) ** 1.5
        else:
            return Constraint.Skip

    m.hyd = Constraint(m.fe, m.cp, m.tray, rule=_hyd)

    def _hyd1(m, i, j):
        if j > 0:
            return m.L[i, j, 1] * m.Vm[i, j, 1] == 0.166 * (m.Mv1[i, j] - 8.5) ** 1.5
        else:
            return Constraint.Skip

    m.hyd1 = Constraint(m.fe, m.cp, rule=_hyd1)

    def _hydN(m, i, j):
        if j > 0:
            return m.L[i, j, Ntray] * m.Vm[i, j, Ntray] == 0.166 * (m.Mvn[i, j] - 0.17) ** 1.5
        else:
            return Constraint.Skip

    m.hydN = Constraint(m.fe, m.cp, rule=_hydN)

    def _dvm(m, i, j, t):
        if j > 0:
            return m.Vm[i, j, t] == m.x[i, j, t] * ((1/2288) * 0.2685**(1 + (1 - m.T[i, j, t]/512.4)**0.2453)) + (1 - m.x[i, j, t]) * ((1/1235) * 0.27136**(1 + (1 - m.T[i, j, t]/536.4)**0.24))
        else:
            return Constraint.Skip

    m.dvm = Constraint(m.fe, m.cp, m.tray, rule=_dvm)

    # print(m.noisy_f, "noisy filter")

    # Initial conditions for the given noisy-filter
    def _acm(m, t, d_noisy):
        try:
            if d_noisy["M", (t,)]:
                return Constraint.Skip
        except KeyError:
            pass
        return m.M[1, 0, t] == m.M_pred[t]

    m.acm = None if kind == 'steady' or kind == 'steady_control' else \
        Constraint(m.tray, rule=(lambda mod, i1: _acm(mod, i1, m.noisy_f)))

    def _acx(m, t, d_noisy):
        try:
            if d_noisy["x", (t,)]:
                return Constraint.Skip
        except KeyError:
            pass
        return m.x[1, 0, t] == m.x_pred[t]

    m.acx = None if kind == 'steady' or kind == 'steady_control' else \
        Constraint(m.tray, rule=(lambda mod, i1: _acx(mod, i1, m.noisy_f)))

    # List of controls
    u_list = [i for j in list(map(lambda vv: vv.values(), map(lambda v: getattr(m, v), controls))) for i in j]
    # Set of controls (start from 0)
    m.r_set = Set(initialize=range(0, len(u_list)))
    # R matrix
    m.R = Param(m.r_set, m.r_set, initialize=lambda m, ii, jj: 1. if ii == jj else 0.0, mutable=True)
    # reference control input for controls
    m.u_r = Param(m.r_set, initialize=0.0, mutable=True)

    # List of states (might be different from dvs and avs)
    x_list = []
    for i in states.keys():
        var = getattr(m, i)
        for j in states[i]:
            x_list.append(var[j])
    # Set of states
    m.q_set = Set(initialize=range(0, len(x_list)))
    # Q matrix for states
    m.Q = Param(m.q_set, m.q_set, initialize=lambda m, ii, jj: 1. if ii == jj else 0.0, mutable=True)
    # reference states for control
    m.x_r = Param(m.q_set, initialize=0.0, mutable=True)

    # List of measurements (at the ncp point)
    y_l = {}
    if kind == "mhe":
        for i in range(1, nfe_t + 1):
            y_l[i] = []
            for meas in measurements.keys():
                var = getattr(m, meas)
                for j in measurements[meas]:
                    idx = (i, m.ncp_t) + j  #: Last collocation point
                    y_l[i].append(var[idx])
                    # print(var[idx])
    # Set of measurements
    m.rMHE_set = Set(initialize=range(0, len(y_l[1]))) if kind == "mhe" else None
    # (inverse)Covariance of measurements
    m.R_MHE = Param(m.fe, m.rMHE_set, m.rMHE_set,
                    initialize=lambda mod, t_l, m_j, m_k: 1. if m_j == m_k else 0.0, mutable=True) if kind == "mhe"\
        else None
    # Measurement noise
    m.v_ = Var(m.fe, m.rMHE_set, initialize=0.0) if kind == "mhe" else None
    # Measurement (default 1.0, pls update!)
    m.y0 = Param(m.fe, m.rMHE_set, initialize=1.0, mutable=True) if kind == "mhe" else None

    # Measurement noise constraints
    # case n0mpc or dynamic w measurements -> disaster
    def _h_cons(m, i, j, y):
        return m.y0[i, j] - y[i][j] - m.v_[i, j] == 0
    # print(measurements)
    m.h_con = Constraint(m.fe, m.rMHE_set, rule=lambda model, t_i, m_j: _h_cons(model, t_i, m_j, y_l)) \
        if measurements.__len__() > 0 else None

    # Plant noise
    w_l = {}
    if kind == "mhe":
        for i in range(1, nfe_t):
            w_l[i] = []  # flattened list of states at time i
            for state in noisy_states.keys():
                var = getattr(m, "w_" + state)
                for j in noisy_states[state]:
                        w_l[i].append(var[(i,) + j])
    # Plant noise set
    m.w_set = Set(initialize=range(0, len(w_l[1]))) if kind == "mhe" else None
    # (inverse)Covariance of plant noise at time i
    m.Q_MHE = Param(range(1, nfe_t), m.w_set, m.w_set,
                    initialize=lambda mod, t_i, st_j, st_k: 1. if st_j == st_k else 0.0, mutable=True) if kind == "mhe"\
        else None

    # Prior state estimate list
    zk_N_list = []
    for i in noisy_states.keys():
        var = getattr(m, i)
        for j in noisy_states[i]:
            # for k in range(1, nfe_t + 1):
            #     zk_N_list.append(var[(k, 0) + j])
            zk_N_list.append(var[(1, 0) + j])
    # Prior-state list
    m.zk_Nset = Set(initialize=range(0, len(zk_N_list)))
    # (inverse)Prior-covariance
    m.PI_k_N = Param(m.zk_Nset, m.zk_Nset, initialize=lambda m, ii, jj: 1. if ii == jj else 0.0, mutable=True)
    # Prior-state
    m.z_0 = Param(m.zk_Nset, initialize=0.0, mutable=True)

    # Set-point track (NMPC)
    def sptrk(m, x, u):
        expr_x = sum(sum(
            m.Q[i, j] * (x[i] - m.x_r[i]) * (x[j] - m.x_r[j]) for i in range(0, len(x))) for j in range(0, len(x)))
        expr_u = sum(sum(
            m.R[i, j] * (u[i] - m.u_r[i]) * (u[j] - m.u_r[j]) for i in range(0, len(u))) for j in range(0, len(u)))
        return expr_x + expr_u

    # Moving-Horizon Estimator: least-squares (MHE)
    def mhelsq(m, w, zk):
        expr_w = sum(
            sum(sum(m.Q_MHE[i, j, k] * w[i][j] * w[i][k] for j in m.w_set) for k in m.w_set)
            for i in range(1, nfe_t))  # one minus nfe_t
        expr_v = sum(
            sum(sum(m.R_MHE[i, j, k] * m.v_[i, j] * m.v_[i, k] for j in m.rMHE_set) for k in m.rMHE_set)
            for i in m.fe)
        expr_PHI = sum(sum(
            m.PI_k_N[i, j] * (zk[i] - m.z_0[i]) * (zk[j] - m.z_0[j]) for i in range(0, len(zk))) for j in range(0, len(zk)))
        return expr_w + expr_v + expr_PHI

    # Objective function
    def _obj_fun(m, obj_f, x, u, w, zk):
        if obj_f == "set_point_track":
            return sptrk(m, x, u)
        elif obj_f == "mhe_least_sq":
            return mhelsq(m, w, zk)
        else:
            return 1.0

    # m.O_F = Objective(rule=lambda x: _obj_fun(x, objf_kind, x_list, u_list, w_l, zk_N_list), sense=minimize)
    m.O_F = Objective(sense=minimize)
    m.dual = Suffix(direction=Suffix.IMPORT_EXPORT)

    return m