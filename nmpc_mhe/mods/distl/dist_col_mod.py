from __future__ import division
from __future__ import print_function
from pyomo.core.base import ConcreteModel, Set, Constraint, Var,\
    Param, Objective, minimize, sqrt, exp, Suffix

"""
Version 03. 
"""

__author__ = 'David M Thierry @dthierry'



# mass balances
def m_ode(m, i, j, k):
    if j > 0 and 1 < k < m.Ntray:
        return m.dM_dt[i, j, k] == \
               (m.V[i, j, k - 1] - m.V[i, j, k] + m.L[i, j, k + 1] - m.L[i, j, k] + m.feed[k]) * m.hi_t[i]
    elif j > 0 and k == 1:
        return m.dM_dt[i, j, 1] == \
               (m.L[i, j, 2] - m.L[i, j, 1] - m.V[i, j, 1]) * m.hi_t[i]
    elif j > 0 and k == m.Ntray:
        return m.dM_dt[i, j, m.Ntray] == \
               (m.V[i, j, m.Ntray - 1] - m.L[i, j, m.Ntray] - m.D[i, j]) * m.hi_t[i]

    else:
        return Constraint.Skip


# def moder(m, i, j):
#     if j > 0:
#         return m.dM_dt[i, j, 1] == \
#                (m.L[i, j, 2] - m.L[i, j, 1] - m.V[i, j, 1]) * m.hi_t[i]
#     else:
#         return Constraint.Skip


# def MODEc(m, i, j):
#     if j > 0:
#         return m.dM_dt[i, j, m.Ntray] == \
#                (m.V[i, j, m.Ntray - 1] - m.L[i, j, m.Ntray] - m.D[i, j]) * m.hi_t[i]
#     else:
#         return Constraint.Skip


def M_COLL(m, i, j, t):
    if j > 0:
        return m.dM_dt[i, j, t] == \
               sum(m.ldot_t[j, k] * m.M[i, k, t] for k in m.cp_t)
    else:
        return Constraint.Skip

## !!!!!
def M_CONT(m, i, t):
    if i < m.nfe_t and m.nfe_t > 1:
        return m.M[i + 1, 0, t] == \
               sum(m.l1_t[j] * m.M[i, j, t] for j in m.cp_t)
    else:
        return Constraint.Skip


def x_ode(m, i, j, t):
    if j > 0 and 1 < t < m.Ntray:
        return m.M[i, j, t] * m.dx_dt[i, j, t] == \
               (m.V[i, j, t - 1] * (m.y[i, j, t - 1] - m.x[i, j, t]) +
                m.L[i, j, t + 1] * (m.x[i, j, t + 1] - m.x[i, j, t]) -
                m.V[i, j, t] * (m.y[i, j, t] - m.x[i, j, t]) +
                m.feed[t] * (m.xf - m.x[i, j, t])) * m.hi_t[i]
    elif j > 0 and t == 1:
        return m.M[i, j, 1] * m.dx_dt[i, j, 1] == \
               (m.L[i, j, 2] * (m.x[i, j, 2] - m.x[i, j, 1]) -
                m.V[i, j, 1] * (m.y[i, j, 1] - m.x[i, j, 1])) * m.hi_t[i]
    elif j > 0 and t == m.Ntray:
        return m.M[i, j, m.Ntray] * m.dx_dt[i, j, m.Ntray] == \
               (m.V[i, j, m.Ntray - 1] * (m.y[i, j, m.Ntray - 1] - m.x[i, j, m.Ntray])) * m.hi_t[i]
    else:
        return Constraint.Skip

# def xoder(m, i, j):
#     if j > 0:
#         return m.M[i, j, 1] * m.dx_dt[i, j, 1] ==\
#                (m.L[i, j, 2] * (m.x[i, j, 2] - m.x[i, j, 1]) -
#                 m.V[i, j, 1] * (m.y[i, j, 1] - m.x[i, j, 1])) * m.hi_t[i]
#     else:
#         return Constraint.Skip

# def xodec(m, i, j):
#     if j > 0:
#         return m.M[i, j, m.Ntray] * m.dx_dt[i, j, m.Ntray] == \
#                (m.V[i, j, m.Ntray - 1] * (m.y[i, j, m.Ntray - 1] - m.x[i, j, m.Ntray])) * m.hi_t[i]
#     else:
#         return Constraint.Skip


def x_coll(m, i, j, t):
    if j > 0:
        return m.dx_dt[i, j, t] == \
               sum(m.ldot_t[j, k] * m.x[i, k, t] for k in m.cp_t)
    else:
        return Constraint.Skip


def x_cont(m, i, t):
    if i < m.nfe_t and m.nfe_t > 1:
        return m.x[i + 1, 0, t] == \
               sum(m.l1_t[j] * m.x[i, j, t] for j in m.cp_t)
    else:
        return Constraint.Skip


def hrc(m, i, j):
    if j > 0:
        return m.D[i, j] - m.Rec[i]*m.L[i, j, m.Ntray] == 0
    else:
        return Constraint.Skip


# Energy balance
def gh(m, i, j, t):
    if j > 0 and 1 < t < m.Ntray:
        return m.M[i, j, t] * (
            m.dx_dt[i, j, t] * (
                (m.hlm0 - m.hln0) * (m.T[i, j, t]**3) +
                (m.hlma - m.hlna) * (m.T[i, j, t]**2) +
                (m.hlmb - m.hlnb) * m.T[i, j, t] + m.hlmc - m.hlnc) +
            m.hi_t[i] * m.Tdot[i, j, t] * (
                3*m.hln0*(m.T[i, j, t]**2) +
                2*m.hlna * m.T[i, j, t] + m.hlnb +
                m.x[i, j, t] *
                (3*(m.hlm0 - m.hln0) * (m.T[i, j, t]**2) + 2 * (m.hlma - m.hlna) * m.T[i, j, t] + m.hlmb - m.hlnb))
        ) == (m.V[i, j, t-1] * (m.hv[i, j, t-1] - m.hl[i, j, t]) +
              m.L[i, j, t+1] * (m.hl[i, j, t+1] - m.hl[i, j, t]) -
              m.V[i, j, t] * (m.hv[i, j, t] - m.hl[i, j, t]) +
              m.feed[t] * (m.hf - m.hl[i, j, t])) * m.hi_t[i]
    else:
        return Constraint.Skip

def ghb(m, i, j):
    if j > 0:
        return m.M[i, j, 1] * (m.dx_dt[i, j, 1] * ((m.hlm0 - m.hln0) * m.T[i, j, 1]**3 + (m.hlma - m.hlna)*m.T[i, j, 1]**2 + (m.hlmb - m.hlnb)*m.T[i, j, 1] + m.hlmc - m.hlnc) + m.hi_t[i] * m.Tdot[i, j, 1] * (3 * m.hln0 * m.T[i, j, 1]**2 + 2 * m.hlna * m.T[i, j, 1] + m.hlnb + m.x[i, j, 1] * (3 * (m.hlm0 - m.hln0) * m.T[i, j, 1]**2 + 2*(m.hlma - m.hlna) * m.T[i, j, 1] + m.hlmb - m.hlnb))) == \
               (m.L[i, j, 2] * (m.hl[i, j, 2] - m.hl[i, j, 1]) - m.V[i, j, 1] * (m.hv[i, j, 1] - m.hl[i, j, 1]) + m.Qr[i]) * m.hi_t[i]
    else:
        return Constraint.Skip

def ghc(m, i, j):
    if j > 0:
        return m.M[i, j, m.Ntray] * (m.dx_dt[i, j, m.Ntray] * ((m.hlm0 - m.hln0) * m.T[i, j, m.Ntray]**3 + (m.hlma - m.hlna) * m.T[i, j, m.Ntray]**2 + (m.hlmb - m.hlnb) * m.T[i, j, m.Ntray] + m.hlmc - m.hlnc) + m.hi_t[i] * m.Tdot[i, j, m.Ntray] * (3 * m.hln0 * m.T[i, j, m.Ntray]**2 + 2* m.hlna * m.T[i, j, m.Ntray] + m.hlnb + m.x[i, j, m.Ntray] * (3 * (m.hlm0 - m.hln0) * m.T[i, j, m.Ntray]**2 + 2 * (m.hlma - m.hlna) * m.T[i, j, m.Ntray] + m.hlmb - m.hlnb))) == \
               (m.V[i, j, m.Ntray - 1] * (m.hv[i, j, m.Ntray - 1] - m.hl[i, j, m.Ntray]) - m.Qc[i, j]) * m.hi_t[i]
    else:
        return Constraint.Skip

def hkl(m, i, j, t):
    if j > 0:
        return m.hl[i, j, t] == m.x[i, j, t]*(m.hlm0*m.T[i, j, t]**3 + m.hlma * m.T[i, j, t]**2 + m.hlmb * m.T[i, j, t] + m.hlmc) + (1 - m.x[i, j, t])*(m.hln0 * m.T[i, j, t]**3 + m.hlna*m.T[i, j, t]**2 + m.hlnb * m.T[i, j, t] + m.hlnc)
    else:
        return Constraint.Skip

def hkv(m, i, j, t):
    if j > 0 and t < m.Ntray:
        return m.hv[i, j, t] == m.y[i, j, t] * (m.hlm0 * m.T[i, j, t]**3 + m.hlma * m.T[i, j, t]**2 + m.hlmb * m.T[i, j, t] + m.hlmc + m.r * m.Tkm * sqrt(1 - (m.p[t]/m.Pkm) * (m.Tkm/m.T[i, j, t])**3)*(m.a - m.b * m.T[i, j, t]/m.Tkm + m.c1 * (m.T[i, j , t]/m.Tkm)**7 + m.gm * (m.d - m.l * m.T[i, j, t]/m.Tkm + m.f*(m.T[i, j, t]/m.Tkm)**7 ))) + (1 - m.y[i, j, t]) * (m.hln0 * m.T[i, j, t]**3 + m.hlna * m.T[i, j, t]**2 + m.hlnb * m.T[i, j, t] + m.hlnc + m.r * m.Tkn * sqrt(1 - (m.p[t]/m.Pkn)*(m.Tkn/m.T[i, j, t])**3)*(m.a - m.b * m.T[i, j, t]/m.Tkn + m.c1 * (m.T[i, j, t]/m.Tkn)**7 + m.gn*(m.d - m.l * m.T[i, j, t]/m.Tkn + m.f* (m.T[i, j, t]/m.Tkn)**7)))
    else:
        return Constraint.Skip

def lpm(m, i, j, t):
    if j > 0:
        return m.pm[i, j, t] == exp(m.CapAm - m.CapBm/(m.T[i, j, t] + m.CapCm))
    else:
        return Constraint.Skip

def lpn(m, i, j, t):
    if j > 0:
        return m.pn[i, j, t] == exp(m.CapAn - m.CapBn/(m.T[i, j, t] + m.CapCn))

    else:
        return Constraint.Skip

def dp(m, i, j, t):
    if j > 0:
        return m.p[t] == m.pm[i, j, t] * m.x[i, j, t] + (1 - m.x[i, j, t]) * m.pn[i, j, t]
    else:
        return Constraint.Skip

def lTdot(m, i, j, t):
    if j > 0:
        return m.Tdot[i, j, t] * m.hi_t[i] ==\
               -(m.pm[i, j, t] - m.pn[i, j, t]) * m.dx_dt[i, j, t] / \
               (m.x[i, j, t] *
                exp(m.CapAm - m.CapBm/(m.T[i, j, t] + m.CapCm)) * m.CapBm/(m.T[i, j, t] + m.CapCm)**2 +
                (1 - m.x[i, j, t]) *
                exp(m.CapAn - m.CapBn/(m.T[i, j, t] + m.CapCn)) * m.CapBn/(m.T[i, j, t] + m.CapCn)**2)
    else:
        return Constraint.Skip

def gy0(m, i, j):
    if j > 0:
        return m.p[1] * m.y[i, j, 1] == m.x[i, j, 1] * m.pm[i, j, 1]
    else:
        return Constraint.Skip

def gy(m, i, j, t):
    if j > 0 and 1 < t < m.Ntray:
        return m.y[i, j, t] == \
               m.alpha[t] * m.x[i, j, t] * m.pm[i, j, t] / m.p[t] + (1 - m.alpha[t]) * m.y[i, j, t - 1]
    else:
        return Constraint.Skip

def dMV(m, i, j, t):
    if j > 0 and 1 < t < m.Ntray:
        return m.Mv[i, j, t] == m.Vm[i, j, t] * m.M[i, j, t]
    else:
        return Constraint.Skip

def dMv1(m, i, j):
    if j > 0:
        return m.Mv1[i, j] == m.Vm[i, j, 1] * m.M[i, j, 1]
    else:
        return Constraint.Skip

def dMvn(m, i, j):
    if j > 0:
        return m.Mvn[i, j] == m.Vm[i, j, m.Ntray] * m.M[i, j, m.Ntray]
    else:
        return Constraint.Skip

def hyd(m, i, j, t):
    if j > 0 and 1 < t < m.Ntray:
        return m.L[i, j, t] * m.Vm[i, j, t] == 0.166 * (m.Mv[i, j, t] - 0.155) ** 1.5
    else:
        return Constraint.Skip

def hyd1(m, i, j):
    if j > 0:
        return m.L[i, j, 1] * m.Vm[i, j, 1] == 0.166 * (m.Mv1[i, j] - 8.5) ** 1.5
    else:
        return Constraint.Skip

def hydN(m, i, j):
    if j > 0:
        return m.L[i, j, m.Ntray] * m.Vm[i, j, m.Ntray] == 0.166 * (m.Mvn[i, j] - 0.17) ** 1.5
    else:
        return Constraint.Skip

def dvm(m, i, j, t):
    if j > 0:
        return m.Vm[i, j, t] == m.x[i, j, t] * ((1/2288) * 0.2685**(1 + (1 - m.T[i, j, t]/512.4)**0.2453)) + (1 - m.x[i, j, t]) * ((1/1235) * 0.27136**(1 + (1 - m.T[i, j, t]/536.4)**0.24))
    else:
        return Constraint.Skip


# Initial conditions for the given noisy-filter
def acm(m, t):
    return m.M[1, 0, t] == m.M_ic[t]

def acx(m, t):
    return m.x[1, 0, t] == m.x_ic[t]
