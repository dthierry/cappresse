from __future__ import division

from pyomo.core.base import Constraint, sqrt, exp, Expression, log
from nmpc_mhe.aux.cpoinsc import collptsgen
from nmpc_mhe.aux.lagrange_f import lgr, lgry, lgrdot, lgrydot

"""
Version note implemented momentum balance and diffusive terms for the bubble region (gas)
have pressure given by ideal gas and dpdx by dummy
momentum per volume unit mom = vg * rhog   
"""

__author__ = 'David M Thierry @dthierry'


def fldoti_x(m, j, k, ncp, a, b):
    return lgrdot(j, m.taucp_x[k], ncp, a, b)


def fldoti_t(m, j, k, ncp, a, b):
    return lgrdot(j, m.taucp_t[k], ncp, a, b)


def fldotyi(m, j, k, ncp, a, b):
    if j > 0:
        return lgrydot(j, m.taucp_x[k], ncp, a, b)
    else:
        return 0.0


def flj1_x(m, j, ncp, a, b):
    return lgr(j, 1, ncp, a, b)


def flj1_t(m, j, ncp, a, b):
    return lgr(j, 1, ncp, a, b)


def fljy1(m, j, ncp, a, b):
    if j > 0:
        return lgry(j, 1, ncp, a, b)
    else:
        return 0.0


def f_lj_x(m, j, k, ncp, a, b):
    return lgr(j, m.taucp_x[k], ncp, a, b)


def fir_hi(m, i):
    reg = m.lenleft / m.nfe_x
    return reg


def fl_irule(m, j, k):
    h0 = sum(m.hi_x[i] for i in range(1, j))
    return float(m.hi_x[j] * m.tau_i_x[k] + h0)


def fini_cp(i, y, k, taucp):
    dy = y[i + 1] - y[i]
    if i == 1 and k == 1:
        yx = y[i]
        # yx = dy * taucp[k] + y[i]
    else:
        yx = dy * taucp[k] + y[i]
    return yx


def fini_cp_dv(i, y, k, taucp):
    dy = y[i + 1] - y[i]
    yx = dy * taucp[k] + y[i]
    return yx


def gasout_zi_rule(m, fet, cpt, i):
    return m.GasOut_z_ix[i]


# gas bubble
def ic_ngb_rule(m, ix, jx, c):
    if 0 < jx <= m.ncp_x:
        return m.Ngb[1, 0, ix, jx, c] == m.Ngb_ic[(ix, jx, c)]
    else:
        return Constraint.Skip

def ic_hgb_rule(m, ix, jx):
    if 0 < jx <= m.ncp_x:
        return m.Hgb[1, 0, ix, jx] == m.Hgb_ic[(ix, jx)]
    else:
        return Constraint.Skip

# gas cloud wake
def ic_ngc_rule(m, ix, jx, c):
    if 0 < jx <= m.ncp_x:
        return m.Ngc[1, 0, ix, jx, c] == m.Ngc_ic[(ix, jx, c)]
    else:
        return Constraint.Skip

def ic_hgc_rule(m, ix, jx):
    if 0 < jx <= m.ncp_x:
        return m.Hgc[1, 0, ix, jx] == m.Hgc_ic[(ix, jx)]
    else:
        return Constraint.Skip

# solid cloud wake
def ic_nsc_rule(m, ix, jx, c):
    if 0 < jx <= m.ncp_x:
        return m.Nsc[1, 0, ix, jx, c] == m.Nsc_ic[(ix, jx, c)]
    else:
        return Constraint.Skip

def ic_hsc_rule(m, ix, jx):
    if 0 < jx <= m.ncp_x:
        return m.Hsc[1, 0, ix, jx] == m.Hsc_ic[(ix, jx)]
    else:
        return Constraint.Skip

# gas emulsion
def ic_nge_rule(m, ix, jx, c):
    if 0 < jx <= m.ncp_x:
        return m.Nge[1, 0, ix, jx, c] == m.Nge_ic[(ix, jx, c)]
    else:
        return Constraint.Skip

def ic_hge_rule(m, ix, jx):
    if 0 < jx <= m.ncp_x:
        return m.Hge[1, 0, ix, jx] == m.Hge_ic[(ix, jx)]
    else:
        return Constraint.Skip

# solids emulsion
def ic_nse_rule(m, ix, jx, c):
    if 0 < jx <= m.ncp_x:
        return m.Nse[1, 0, ix, jx, c] == m.Nse_ic[(ix, jx, c)]
    else:
        return Constraint.Skip


def ic_hse_rule(m, ix, jx):
    if 0 < jx <= m.ncp_x:
        return m.Hse[1, 0, ix, jx] == m.Hse_ic[(ix, jx)]
    else:
        return Constraint.Skip



# expr ================================================================================================
# expr ================================================================================================


# gas cloud wake
# cc
def ngc_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ngc[it, jt, ix, jx, c] == \
               m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * m.ed[it, jt, ix, jx] * \
               m.cc[it, jt, ix, jx, c]
    else:
        return Constraint.Skip
# Tgc
def hgc_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hgc[it, jt, ix, jx] == \
               m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * m.ed[it, jt, ix, jx] * \
               sum(m.cc[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol * m.Tgc[it, jt, ix, jx]
    else:
        return Constraint.Skip

# solid cloud wake
# nc
def nsc_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Nsc[it, jt, ix, jx, c]/m.rhos == \
               m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) * \
               m.nc[it, jt, ix, jx, c]
    else:
        return Constraint.Skip

# Tsc
def hsc_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hsc[it, jt, ix, jx] == \
               m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) * m.rhos * \
               m.cps * m.Tsc[it, jt, ix, jx]
    else:
        return Constraint.Skip

# gas emulsion
# ce
def nge_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Nge[it, jt, ix, jx, c] == \
               m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * m.ed[
                   it, jt, ix, jx] * \
               m.ce[it, jt, ix, jx, c]
    else:
        return Constraint.Skip

# Tge
def hge_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hge[it, jt, ix, jx] == \
               m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * m.ed[
                   it, jt, ix, jx] * \
               sum(m.ce[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol * m.Tge[it, jt, ix, jx]
    else:
        return Constraint.Skip

# solids emulsion
# ne
def nse_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Nse[it, jt, ix, jx, c] == \
               m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * \
               (1. - m.ed[it, jt, ix, jx]) * m.rhos * \
               m.ne[it, jt, ix, jx, c]
    else:
        return Constraint.Skip


# Tse
def hse_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hse[it, jt, ix, jx]/m.rhos == \
               m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * \
               (1. - m.ed[it, jt, ix, jx]) * m.cps * m.Tse[it, jt, ix, jx]
    else:
        return Constraint.Skip

# solids in the bed
# expr ================================================================================================
#
# Ngb
def fdvar_t_ngb(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dNgb_dt[it, kt, ix, kx, c] == \
               sum(m.ldot_t[jt, kt] * m.Ngb[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

# Hgb
def fdvar_t_hgb(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dHgb_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Hgb[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

# Ngc
def fdvar_t_ngc(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dNgc_dt[it, kt, ix, kx, c] == \
               sum(m.ldot_t[jt, kt] * m.Ngc[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

# Hgc
def fdvar_t_hgc(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dHgc_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Hgc[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

# Nsc
def fdvar_t_nsc(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dNsc_dt[it, kt, ix, kx, c] == \
               sum(m.ldot_t[jt, kt] * m.Nsc[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

# Hsc
def fdvar_t_hsc(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dHsc_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Hsc[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

# Nge
def fdvar_t_nge(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dNge_dt[it, kt, ix, kx, c] == \
               sum(m.ldot_t[jt, kt] * m.Nge[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

# Hge
def fdvar_t_hge(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dHge_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Hge[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

# Nse
def fdvar_t_nse(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dNse_dt[it, kt, ix, kx, c] == \
               sum(m.ldot_t[jt, kt] * m.Nse[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

# Hse
def fdvar_t_hse(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dHse_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Hse[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

# expr ================================================================================================

# Ngbi0
def fcp_t_ngb(m, it, ix, kx, c):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Ngb[it + 1, 0, ix, kx, c] - \
               sum(m.l1_t[jt] * m.Ngb[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

# Hgbi0
def fcp_t_hgb(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Hgb[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Hgb[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

# Ngci0
def fcp_t_ngc(m, it, ix, kx, c):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Ngc[it + 1, 0, ix, kx, c] - \
               sum(m.l1_t[jt] * m.Ngc[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

# Hgci0
def fcp_t_hgc(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Hgc[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Hgc[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

# Nsei0
def fcp_t_nsc(m, it, ix, kx, c):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Nsc[it + 1, 0, ix, kx, c] - \
               sum(m.l1_t[jt] * m.Nsc[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

# Hsei0
def fcp_t_hsc(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Hsc[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Hsc[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

# Ngei0
def fcp_t_nge(m, it, ix, kx, c):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Nge[it + 1, 0, ix, kx, c] - \
               sum(m.l1_t[jt] * m.Nge[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

# Hgei0
def fcp_t_hge(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Hge[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Hge[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

# Nsei0
def fcp_t_nse(m, it, ix, kx, c):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Nse[it + 1, 0, ix, kx, c] - \
               sum(m.l1_t[jt] * m.Nse[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

# Hsei0
def fcp_t_hse(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Hse[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Hse[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

# vg
# def a1_rule(m, it, jt, ix, jx):
def Gb_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.vg[it, jt, ix, jx] * m.Ax * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) == m.Gb[it, jt, ix, jx]
    else:
        return Constraint.Skip


# hsc
def a4_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ecwin[it, jt, ix, jx] == m.Jc[it, jt, ix, jx] * m.hsc[it, jt, ix, jx]
    else:

        return Constraint.Skip

# hse
def a5_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.eein[it, jt, ix, jx] == m.Je[it, jt, ix, jx] * m.hse[it, jt, ix, jx]
    else:
        return Constraint.Skip

# nc
def a8_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ccwin[it, jt, ix, jx, k] == m.Jc[it, jt, ix, jx] * m.nc[it, jt, ix, jx, k]
    else:
        return Constraint.Skip

# ne
def a9_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.cein[it, jt, ix, jx, k] == m.Je[it, jt, ix, jx] * m.ne[it, jt, ix, jx, k]
    else:
        return Constraint.Skip

# Je
def a11_rule_2(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.z[it, jt, ix, jx] == m.Je[it, jt, ix, jx] - m.Jc[it, jt, ix, jx]
    else:
        return Constraint.Skip

# delta
def a13_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Gb[it, jt, ix, jx] == m.vb[it, jt, ix, jx] * m.Ax * m.delta[it, jt, ix, jx] * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp)
    else:
        return Constraint.Skip

# Jc
def a14_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Jc[it, jt, ix, jx] == \
               m.fw * m.delta[it, jt, ix, jx] * m.rhos * (1 - m.ed[it, jt, ix, jx]) * m.vb[it, jt, ix, jx]
    else:
        return Constraint.Skip


# yb
def a15_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.cb[it, jt, ix, jx, k] == m.yb[it, jt, ix, jx, k] * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp)
    else:
        return Constraint.Skip

# yc
def a16_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.cc[it, jt, ix, jx, k] == m.yc[it, jt, ix, jx, k] * sum(m.cc[it, jt, ix, jx, kx] for kx in m.sp)
    else:
        return Constraint.Skip

# ye
def a17_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ce[it, jt, ix, jx, k] == m.ye[it, jt, ix, jx, k] * sum(m.ce[it, jt, ix, jx, kx] for kx in m.sp)
    else:
        return Constraint.Skip

# D 'c'
def a22_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.D[it, jt, ix, jx, 'c'] == \
               (0.1593 - 0.1282 * (m.P[it, jt, ix, jx] - 1.4) + 0.001 * (m.Tge[it, jt, ix, jx] - 60) + 0.0964 * (
                   (m.P[it, jt, ix, jx] - 1.4) ** 2) - 0.0006921 * (
                (m.P[it, jt, ix, jx] - 1.4) * (m.Tge[it, jt, ix, jx] - 60)) -
                3.3532e-06 * (m.Tge[it, jt, ix, jx] - 60) ** 2) * m.ye[it, jt, ix, jx, 'h'] / (
                   m.ye[it, jt, ix, jx, 'h'] + m.ye[it, jt, ix, jx, 'n']) + \
               (
               0.1495 - 0.1204 * (m.P[it, jt, ix, jx] - 1.4) + 0.0008896 * (m.Tge[it, jt, ix, jx] - 60) + 0.0906 * (
                   (m.P[it, jt, ix, jx] - 1.4) ** 2) -
               0.0005857 * (m.P[it, jt, ix, jx] - 1.4) * (m.Tge[it, jt, ix, jx] - 60) -
               3.559e-06 * (m.Tge[it, jt, ix, jx] - 60) ** 2) * m.ye[it, jt, ix, jx, 'n'] / (
                   m.ye[it, jt, ix, jx, 'h'] + m.ye[it, jt, ix, jx, 'n'])
    else:
        return Constraint.Skip

# D 'h'
def a23_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.D[it, jt, ix, jx, 'h'] == \
               (0.1593 - 0.1282 * (m.P[it, jt, ix, jx] - 1.4) + 0.001 * (m.Tge[it, jt, ix, jx] - 60) +
                0.0964 * ((m.P[it, jt, ix, jx] - 1.4) ** 2) - 0.0006921 * (
                    (m.P[it, jt, ix, jx] - 1.4) * (m.Tge[it, jt, ix, jx] - 60)) -
                3.3532e-06 * (m.Tge[it, jt, ix, jx] - 60) ** 2) * m.ye[it, jt, ix, jx, 'c'] / (
                   m.ye[it, jt, ix, jx, 'c'] + m.ye[it, jt, ix, jx, 'n']) + \
               (
               0.2165 - 0.1743 * (m.P[it, jt, ix, jx] - 1.4) + 0.001377 * (m.Tge[it, jt, ix, jx] - 60) + 0.13109 * (
                   (m.P[it, jt, ix, jx] - 1.4) ** 2) -
               0.0009115 * (m.P[it, jt, ix, jx] - 1.4) * (m.Tge[it, jt, ix, jx] - 60) -
               4.8394e-06 * (m.Tge[it, jt, ix, jx] - 60) ** 2) * m.ye[it, jt, ix, jx, 'n'] / (
                   m.ye[it, jt, ix, jx, 'c'] + m.ye[it, jt, ix, jx, 'n'])
    else:
        return Constraint.Skip

# D 'n'
def a24_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.D[it, jt, ix, jx, 'n'] == \
               (
               0.1495 - 0.1204 * (m.P[it, jt, ix, jx] - 1.4) + 0.0008896 * (m.Tge[it, jt, ix, jx] - 60) + 0.0906 * (
                   (m.P[it, jt, ix, jx] - 1.4) ** 2) -
               0.0005857 * (m.P[it, jt, ix, jx] - 1.4) * (m.Tge[it, jt, ix, jx] - 60) -
               3.559e-06 * (m.Tge[it, jt, ix, jx] - 60) ** 2) * m.ye[it, jt, ix, jx, 'c'] / (
                   m.ye[it, jt, ix, jx, 'h'] + m.ye[it, jt, ix, jx, 'c']) + \
               (
               0.2165 - 0.1743 * (m.P[it, jt, ix, jx] - 1.4) + 0.001377 * (m.Tge[it, jt, ix, jx] - 60) + 0.13109 * (
                   (m.P[it, jt, ix, jx] - 1.4) ** 2) -
               0.0009115 * (m.P[it, jt, ix, jx] - 1.4) * (m.Tge[it, jt, ix, jx] - 60) -
               4.8394e-06 * (m.Tge[it, jt, ix, jx] - 60) ** 2) * m.ye[it, jt, ix, jx, 'h'] / (
                   m.ye[it, jt, ix, jx, 'h'] + m.ye[it, jt, ix, jx, 'c'])
    else:
        return Constraint.Skip


# rhog
def a25_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rhog[it, jt, ix, jx] == m.P[it, jt, ix, jx] * 100 * (m.ye[it, jt, ix, jx, 'c'] * 44.01 + m.ye[it, jt, ix, jx, 'n'] * 28.01 + m.ye[it, jt, ix, jx, 'h'] * 18.02) \
                                         / (8.314 * (m.Tge[it, jt, ix, jx] + 273.16))
    else:
        return Constraint.Skip

# Ar
def a26_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ar[it, jt, ix, jx] == \
               (m.dp ** 3) * m.rhog[it, jt, ix, jx] * (m.rhos - m.rhog[it, jt, ix, jx]) * m.gc / (m.mug ** 2)
    else:
        return Constraint.Skip

# e
def a27_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return (1 - m.e[it, jt, ix, jx]) == (1 - m.ed[it, jt, ix, jx]) * (1 - m.delta[it, jt, ix, jx])
    else:
        return Constraint.Skip

# vbr
def a28_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.vbr[it, jt, ix, jx] == 0.711 * sqrt(m.gc * m.db[it, jt, ix, jx])
    else:
        return Constraint.Skip

# db0 approx
def a29_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.db0[it, jt] == 1.38 * (m.gc ** (-0.2)) * ((m.vg[it, jt, 1, 1] - m.ve[it, jt, 1, 1]) * m.Ao) ** 0.4
    else:
        return Constraint.Skip

# dbe
def a30_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dbe[it, jt, ix, jx] == (m.Dt / 4) * (-m.g1[it, jt] + m.g3[it, jt, ix, jx]) ** 2
    else:
        return Constraint.Skip

# dbm
def a31_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dbm[it, jt, ix, jx] == 2.59 * (m.gc ** (-0.2)) * ((m.vg[it, jt, ix, jx] - m.ve[it, jt, ix, jx]) * m.Ax) ** 0.4
    else:
        return Constraint.Skip

# g1
def a32_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.g1[it, jt] == 2.56E-2 * sqrt(m.Dt / m.gc) / m.vmf[it, jt]
    else:
        return Constraint.Skip


# g2
def a33_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return 4 * m.g2[it, jt, ix, jx] == m.Dt * (m.g1[it, jt] + m.g3[it, jt, ix, jx]) ** 2
    else:
        return Constraint.Skip

# g3
def a34_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.g3[it, jt, ix, jx] == sqrt(m.g1[it, jt] ** 2 + 4 * m.dbm[it, jt, ix, jx] / m.Dt)
    else:
        return Constraint.Skip


# x included?
# dbu
def a35_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return exp(0.3 * (m.l[ix, jx]) / m.Dt) * \
               (((sqrt(m.dbu[it, jt, ix, jx]) - sqrt(m.dbe[it, jt, ix, jx])) / (sqrt(m.db0[it, jt]) - sqrt(m.dbe[it, jt, ix, jx]))) ** (1 - m.g1[it, jt] / m.g3[it, jt, ix, jx])) == \
               (((sqrt(m.dbu[it, jt, ix, jx]) - sqrt(m.g2[it, jt, ix, jx])) / (sqrt(m.db0[it, jt]) - sqrt(m.g2[it, jt, ix, jx]))) ** -(1 + m.g1[it, jt] / m.g3[it, jt, ix, jx]))
    else:
        return Constraint.Skip

# fc
def a36_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.fc[it, jt, ix, jx] == 3. * (m.vmf[it, jt] / m.emf) / (m.vbr[it, jt, ix, jx] - (m.vmf[it, jt] / m.emf))
    else:
        return Constraint.Skip


# fcw
def a37_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.fcw[it, jt, ix, jx] == m.fc[it, jt, ix, jx] + m.fw
    else:
        return Constraint.Skip

# Kbc
def a38_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Kbc[it, jt, ix, jx, k] == \
               1.32 * 4.5 * (m.vmf[it, jt] / m.db[it, jt, ix, jx]) + 5.85 * (
                   ((m.D[it, jt, ix, jx, k] * 1E-4) ** 0.5) * (m.gc ** 0.25) / (m.db[it, jt, ix, jx] ** (5 / 4)))
    else:
        return Constraint.Skip


# Kce
def a39_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Kce[it, jt, ix, jx, k] == 6.77 * sqrt(m.ed[it, jt, ix, jx] * (m.D[it, jt, ix, jx, k] * 1E-4) * m.vbr[it, jt, ix, jx] / (m.db[it, jt, ix, jx] ** 3))
    else:
        return Constraint.Skip


# Kcebs
def a40_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Kcebs[it, jt, ix, jx] == \
               3 * (1 - m.ed[it, jt, ix, jx]) / ((1 - m.delta[it, jt, ix, jx]) * m.ed[it, jt, ix, jx]) * (m.ve[it, jt, ix, jx] / m.db[it, jt, ix, jx])
    else:
        return Constraint.Skip

# Hbc
def a41_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hbc[it, jt, ix, jx] == 1.32 * 4.5 * m.vmf[it, jt] * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol / m.db[it, jt, ix, jx] + \
                                        5.85 * sqrt((m.kg / 1000) * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol) * (m.gc ** 0.25) / (m.db[it, jt, ix, jx] ** (5 / 4))
    else:
        return Constraint.Skip

# Hce
def a42_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hce[it, jt, ix, jx] == 6.78 * sqrt(m.ed[it, jt, ix, jx] * m.vb[it, jt, ix, jx] * (m.kg / 1000) * sum(m.cc[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol / (m.db[it, jt, ix, jx] ** 3))
    else:
        return Constraint.Skip

# hp
def a43_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Nup[it, jt, ix, jx] == 1000 * exp(m.whp[it, jt, ix, jx]) * m.dp / m.kg
    else:
        return Constraint.Skip

# Red45
def a44_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Red[it, jt, ix, jx] == m.ve[it, jt, ix, jx] * m.dp * m.rhog[it, jt, ix, jx] / m.mug
    else:
        return Constraint.Skip


# Nup
def a45_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Nup[it, jt, ix, jx] == 0.03 * (m.Red[it, jt, ix, jx] ** 1.3)
    else:
        return Constraint.Skip

# kpa
def a46_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.kpa[it, jt, ix, jx] == \
               (3.58 - 2.5 * m.ed[it, jt, ix, jx]) * m.kg * ((m.kp / m.kg) ** (0.46 - 0.46 * m.ed[it, jt, ix, jx]))
    else:
        return Constraint.Skip


# fn
def a47_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.fn[it, jt, ix, jx] == m.vg[it, jt, ix, jx] / m.vmf[it, jt]
    else:
        return Constraint.Skip


# tau
def a48_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.tau[it, jt, ix, jx] == 0.44 * (
        (m.dp * m.gc / ((m.vmf[it, jt] ** 2) * ((m.fn[it, jt, ix, jx] - m.ah) ** 2))) ** 0.14) * (
                                            (m.dp / m.dx) ** 0.225)
    else:
        return Constraint.Skip


# fb
def a49_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.fb[it, jt, ix, jx] == 0.33 * (
        ((m.vmf[it, jt] ** 2) * ((m.fn[it, jt, ix, jx] - m.ah) ** 2) / (m.dp * m.gc)) ** 0.14)
    else:
        return Constraint.Skip


# hd
def a50_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.hd[it, jt, ix, jx] == \
               2 * sqrt((m.kpa[it, jt, ix, jx] / 1000) * m.rhos * m.cps * (1 - m.ed[it, jt, ix, jx]) / (
               m.pi * m.tau[it, jt, ix, jx]))
    else:
        return Constraint.Skip


# hl
def a51_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return 1000 * m.hl[it, jt, ix, jx] * m.dp / m.kg == 0.009 * (m.Ar[it, jt, ix, jx] ** 0.5) * (m.Pr ** 0.33)
    else:
        return Constraint.Skip


# ht
def a52_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ht[it, jt, ix, jx] == m.fb[it, jt, ix, jx] * m.hd[it, jt, ix, jx] + (1 - m.fb[it, jt, ix, jx]) * \
                                                                                     m.hl[it, jt, ix, jx]
    else:
        return Constraint.Skip


# dThx
def a54_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dThx[it, jt, ix, jx] == m.Ttube[it, jt, ix, jx] - m.Tse[it, jt, ix, jx]
    else:
        return Constraint.Skip

# Ttube
def a55_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ht[it, jt, ix, jx] * m.dThx[it, jt, ix, jx] * m.Cr == \
               m.hw * (m.Thx[it, jt, ix, jx] - m.Ttube[it, jt, ix, jx])
    else:
        return Constraint.Skip

# hxh
def a56_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Thx[it, jt, ix, jx] == 33.2104 + 14170.15 * (m.hxh[it, jt, ix, jx] + 0.285)
    else:
        return Constraint.Skip

# vmf
def a57_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return 10 * 1.75 / (m.phis * m.emf ** 3) * (m.dp * m.vmf[it, jt] * m.rhog[it, jt, 1, 1] / m.mug) ** 2 + \
               10 * 150 * (1 - m.emf) / ((m.phis ** 2) * (m.emf ** 3)) * (
               m.dp * m.vmf[it, jt] * m.rhog[it, jt, 1, 1] / m.mug) \
               == \
               10 * m.dp ** 3 * m.rhog[it, jt, 1, 1] * (m.rhos - m.rhog[it, jt, 1, 1]) * m.gc / m.mug ** 2
    else:
        return Constraint.Skip

# k1c
def a58_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        # return m.k1c[it, jt, ix, jx] == \
        #        m.A1 * (m.Tsc[it, jt, ix, jx] + 273.15) * exp(-m.E1 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)))
        return log(m.k1c[it, jt, ix, jx]) - log(m.A1 * (m.Tsc[it, jt, ix, jx] + 273.15)) == (-m.E1 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip

# k2c
def a59_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.wk2c[it, jt, ix, jx] - log(m.A2 * (m.Tsc[it, jt, ix, jx] + 273.15)) == (-m.E2 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip

# k3c
def a60_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.wk3c[it, jt, ix, jx] - log(m.A3 * (m.Tsc[it, jt, ix, jx] + 273.15)) == (-m.E3 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip

# k1e
def a61_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return log(m.k1e[it, jt, ix, jx]) - log(m.A1 * (m.Tse[it, jt, ix, jx] + 273.15)) == (-m.E1 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip

# k2e
def a62_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.wk2e[it, jt, ix, jx] - log(m.A2 * (m.Tse[it, jt, ix, jx] + 273.15)) == (-m.E2 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip

# k3e
def a63_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.wk3e[it, jt, ix, jx] - log(m.A3 * (m.Tse[it, jt, ix, jx] + 273.15)) == (-m.E3 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip

# Ke1c
def a64_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ke1c[it, jt, ix, jx] * m.P[it, jt, ix, jx] * 1E5 == exp(
            -m.dH1 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)) + m.dS1 / m.R)
    else:
        return Constraint.Skip

# Ke2c
# def a65_rule(m, it, jt, ix, jx):
#     if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
#         return m.Ke2c[it, jt, ix, jx] * m.P[it, jt, ix, jx] * 1E5 == exp(
#             -m.dH2 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)) + m.dS2 / m.R)
#     else:
#         return Constraint.Skip

def a65_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.wKe2c[it, jt, ix, jx] + log(m.P[it, jt, ix, jx] * 1E5) == -m.dH2 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)) + m.dS2 / m.R
    else:
        return Constraint.Skip


# Ke3c
def a66_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.wKe3c[it, jt, ix, jx] + log(m.P[it, jt, ix, jx] * 1E5) == -m.dH3 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)) + m.dS3 / m.R
    else:
        return Constraint.Skip

# Ke1e
def a67_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ke1e[it, jt, ix, jx] * m.P[it, jt, ix, jx] * 1E5 == exp(-m.dH1/(m.R * (m.Tse[it, jt, ix, jx] + 273.15)) + m.dS1 / m.R)
    else:
        return Constraint.Skip


# Ke2e
def a68_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.wKe2e[it, jt, ix, jx] + log(m.P[it, jt, ix, jx] * 1E5) == -m.dH2 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)) + m.dS2 / m.R
    else:
        return Constraint.Skip


# Ke3e
def a69_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.wKe3e[it, jt, ix, jx] + log(m.P[it, jt, ix, jx] * 1E5) == -m.dH3 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)) + m.dS3 / m.R
    else:
        return Constraint.Skip


# r1c
def a70_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return (100)*m.r1c[it, jt, ix, jx] / 1E5 == \
               (m.k1c[it, jt, ix, jx] * ((m.P[it, jt, ix, jx] * m.yc[it, jt, ix, jx, 'h']) - (m.nc[it, jt, ix, jx, 'h'] * m.rhos / m.Ke1c[it, jt, ix, jx]) / 1E5))*100
    else:
        return Constraint.Skip


# r2c
def a71_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.r2c[it, jt, ix, jx] / 1E5 == m.k2c[it, jt, ix, jx] * ((1 - 2 * (m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) - (m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) * m.nc[it, jt, ix, jx, 'h'] * m.rhos * m.P[it, jt, ix, jx] * m.yc[it, jt, ix, jx, 'c'] * 1E5 -(((m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) + (m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv))*m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.Ke2c[it, jt, ix, jx]))/1E5
    else:
        return Constraint.Skip


# r3c
def a72_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.r3c[it, jt, ix, jx] == m.k3c[it, jt, ix, jx] * (((1 - 2 * (m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) - (m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) ** 2) * ((m.P[it, jt, ix, jx] * m.yc[it, jt, ix, jx, 'c'] * 1E5) ** m.m1) - ((m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) * ((m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) + (m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) / m.Ke3c[it, jt, ix, jx]))
    else:
        return Constraint.Skip

# r1e
def a73_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return 100 * m.r1e[it, jt, ix, jx] / 1E5 == m.k1e[it, jt, ix, jx] * ((m.P[it, jt, ix, jx] * m.ye[it, jt, ix, jx, 'h']) - (1e-05)*(m.ne[it, jt, ix, jx, 'h'] * m.rhos / m.Ke1e[it, jt, ix, jx]))*100
    else:
        return Constraint.Skip


# r2e
def a74_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return (m.r2e[it, jt, ix, jx] / m.k2e[it, jt, ix, jx]) / (1e+04) == \
               ((1. - 2. * (m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) - (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) * m.ne[it, jt, ix, jx, 'h'] * m.rhos * (m.P[it, jt, ix, jx] * m.ye[it, jt, ix, jx, 'c'] * 1E5) - (((m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) +  (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) * m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.Ke2e[it, jt, ix, jx])) / (1e+04)
    else:
        return Constraint.Skip

# r3e
def a75_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return (m.r3e[it, jt, ix, jx] / m.k3e[it, jt, ix, jx])/(1e+04)  == (((1. - 2. * (m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) - (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) ** 2) * (((m.P[it, jt, ix, jx] * m.ye[it, jt, ix, jx, 'c']) ** m.m1) * 1E5 ** m.m1) -
               ((m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) * ((m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) + (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) /m.Ke3e[it, jt, ix, jx])) / (1e+04)
    else:
        return Constraint.Skip

# rgc 'c'
def a76_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rgc[it, jt, ix, jx, 'c'] == (m.nv * m.r3c[it, jt, ix, jx] + m.r2c[it, jt, ix, jx]) / 1000.
    else:
        return Constraint.Skip

# rge 'c'
def a77_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rge[it, jt, ix, jx, 'c'] == (m.nv * m.r3e[it, jt, ix, jx] + m.r2e[it, jt, ix, jx]) / 1000.
    else:
        return Constraint.Skip

# rsc 'c'
def a78_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rsc[it, jt, ix, jx, 'c'] == m.r2c[it, jt, ix, jx]
    else:
        return Constraint.Skip

# rse 'c'
def a79_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rse[it, jt, ix, jx, 'c'] == m.r2e[it, jt, ix, jx]
    else:
        return Constraint.Skip

# rgc 'h'
def a80_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rgc[it, jt, ix, jx, 'h'] == m.r1c[it, jt, ix, jx] / 1000
    else:
        return Constraint.Skip

# rge 'h'
def a81_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rge[it, jt, ix, jx, 'h'] == m.r1e[it, jt, ix, jx] / 1000
    else:
        return Constraint.Skip

# rsc 'h'
def a82_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rsc[it, jt, ix, jx, 'h'] == m.r1c[it, jt, ix, jx] - m.r2c[it, jt, ix, jx]
    else:
        return Constraint.Skip

# rse 'h'
def a83_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rse[it, jt, ix, jx, 'h'] == m.r1e[it, jt, ix, jx] - m.r2e[it, jt, ix, jx]
    else:
        return Constraint.Skip

# rgc 'n'
def a84_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rgc[it, jt, ix, jx, 'n'] == 0
    else:
        return Constraint.Skip

# rge 'n'
def a85_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rge[it, jt, ix, jx, 'n'] == 0
    else:
        return Constraint.Skip

# rsc 'n'
def a86_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rsc[it, jt, ix, jx, 'n'] == m.nv * m.r3c[it, jt, ix, jx]
    else:
        return Constraint.Skip

# rse 'n'
def a87_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rse[it, jt, ix, jx, 'n'] == m.nv * m.r3e[it, jt, ix, jx]
    else:
        return Constraint.Skip

# hsc
def a88_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.hsc[it, jt, ix, jx] == ((m.nc[it, jt, ix, jx, 'h'] + m.nc[it, jt, ix, jx, 'c']) * (m.cpgcsc['h'] * m.Tsc[it, jt, ix, jx] + m.dH1) +
                                         m.nc[it, jt, ix, jx, 'c'] * (
                                         m.cpgcsc['c'] * m.Tsc[it, jt, ix, jx] + m.dH2) +
                                         m.nc[it, jt, ix, jx, 'n'] * (
                                         m.cpgcsc['c'] * m.Tsc[it, jt, ix, jx] + m.dH3)) * 1E-3 + m.cps * m.Tsc[it, jt, ix, jx]
    else:
        return Constraint.Skip

# hse
def a89_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.hse[it, jt, ix, jx] == ((m.ne[it, jt, ix, jx, 'h'] + m.ne[it, jt, ix, jx, 'c']) * (m.cpgcse['h'] * m.Tse[it, jt, ix, jx] + m.dH1) +
                                         m.ne[it, jt, ix, jx, 'c'] * (
                                         m.cpgcse['c'] * m.Tse[it, jt, ix, jx] + m.dH2) +
                                         m.ne[it, jt, ix, jx, 'n'] * (
                                         m.cpgcse['c'] * m.Tse[it, jt, ix, jx] + m.dH3)) * 1E-3 + m.cps * m.Tse[
            it, jt, ix, jx]
    else:
        return Constraint.Skip

# equation A.3 Gas phase component balance
# dNgc_dt
def de_ngc_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x and k != "n":
        return (1e+04) * m.dNgc_dt[it, jt, ix, jx, k]/m.delta[it, jt, ix, jx] == \
               ((m.Kbc[it, jt, ix, jx, k] * (m.cb[it, jt, ix, jx, k] - m.cc[it, jt, ix, jx, k]) - \
                                           m.Kce[it, jt, ix, jx, k] * (m.cc[it, jt, ix, jx, k] - m.ce[it, jt, ix, jx, k]) - \
                                           m.fcw[it, jt, ix, jx] * (1. - m.ed[it, jt, ix, jx]) * m.rgc[it, jt, ix, jx, k]) * \
                m.hi_t[it]) * (1e+04)
    elif 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x and k == "n":
        return (1e+04) * m.dNgc_dt[it, jt, ix, jx, k]/m.delta[it, jt, ix, jx] == \
               ((m.Kbc[it, jt, ix, jx, k] * (m.cb[it, jt, ix, jx, k] - m.cc[it, jt, ix, jx, k]) - \
                 m.Kce[it, jt, ix, jx, k] * (m.cc[it, jt, ix, jx, k] - m.ce[it, jt, ix, jx, k])) * m.hi_t[it]) * (1e+04)
    else:
        return Constraint.Skip


# equation A.4 Gas phase energy balance
# dHgc_dt
def de_hgc_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dHgc_dt[it, jt, ix, jx] / m.delta[it, jt, ix, jx] == \
               (m.Ax * m.Hbc[it, jt, ix, jx] * (m.Tgb[it, jt, ix, jx] - m.Tgc[it, jt, ix, jx]) - \
                m.Ax * m.Hce[it, jt, ix, jx] * (m.Tgc[it, jt, ix, jx] - m.Tge[it, jt, ix, jx]) - \
                m.Ax * m.fcw[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) * m.rhos * m.ap * exp(m.whp[it, jt, ix, jx]) * (m.Tgc[it, jt, ix, jx] - m.Tsc[it, jt, ix, jx]) - \
                m.Ax * m.fcw[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) * sum(m.rgc[it, jt, ix, jx, k] * m.cpgcgc[k] for k in m.sp2) * m.Tgc[it, jt, ix, jx]) * m.hi_t[it]
    else:
        return Constraint.Skip


# equation A.5 Solid phase adsorbed species balance
# dNse_dt
def de_nsc_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dNsc_dt[it, jt, ix, jx, k] * m.hi_x[ix] == \
               (-m.dccwin_dx[it, jt, ix, jx, k] * m.Ax - m.Ksbulk[it, jt, ix, jx, k] - \
                m.hi_x[ix] * m.Ax * m.delta[it, jt, ix, jx] * m.rhos * m.Kcebs[it, jt, ix, jx] * (m.nc[it, jt, ix, jx, k] - m.ne[it, jt, ix, jx, k]) + \
                m.hi_x[ix] * m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) * m.rsc[it, jt, ix, jx, k]) * m.hi_t[it]
    else:
        return Constraint.Skip

# put derivative space here
# equation A.6 Solid phase energy balance
# dHsc_dt
def de_hsc_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dHsc_dt[it, jt, ix, jx] * m.hi_x[ix] \
               == (-m.decwin_dx[it, jt, ix, jx] * m.Ax - m.Hsbulk[it, jt, ix, jx] - \
                   m.hi_x[ix] * m.Ax * m.delta[it, jt, ix, jx] * m.rhos * m.Kcebs[it, jt, ix, jx] * (m.hsc[it, jt, ix, jx] - m.hse[it, jt, ix, jx]) + \
                   m.hi_x[ix] * m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) * sum((m.rgc[it, jt, ix, jx, k] * m.cpgcgc[k]) for k in m.sp2) * (m.Tgc[it, jt, ix, jx]) + \
                   m.hi_x[ix] * m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (
                   1 - m.ed[it, jt, ix, jx]) * m.rhos * m.ap * exp(m.whp[it, jt, ix, jx]) * (m.Tgc[it, jt, ix, jx] - m.Tsc[it, jt, ix, jx])) * m.hi_t[it]
    else:
        return Constraint.Skip



# equation A.7 Gas phase component balance
# dNge_dt
def de_nge_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x and k != "n":
        return (1e+03) * m.dNge_dt[it, jt, ix, jx, k] \
               == ((m.Ax * m.delta[it, jt, ix, jx] * m.Kce[it, jt, ix, jx, k] * (
        m.cc[it, jt, ix, jx, k] - m.ce[it, jt, ix, jx, k]) - \
                   m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (
                   1. - m.ed[it, jt, ix, jx]) * m.rge[it, jt, ix, jx, k] - \
                   m.Kgbulk[it, jt, ix, jx, k] / m.hi_x[ix]) * m.hi_t[it]) * (1e+03)
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x and k == "n":
        return (1e+03) * m.dNge_dt[it, jt, ix, jx, k] \
               == ((m.Ax * m.delta[it, jt, ix, jx] * m.Kce[it, jt, ix, jx, k] * (
        m.cc[it, jt, ix, jx, k] - m.ce[it, jt, ix, jx, k]) - \
                   m.Kgbulk[it, jt, ix, jx, k] / m.hi_x[ix]) * m.hi_t[it]) * (1e+03)
    else:
        return Constraint.Skip


# equation A.8 Gas phase energy balance
# dHge_dt
def de_hge_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dHge_dt[it, jt, ix, jx] \
               == (m.Ax * m.delta[it, jt, ix, jx] * m.Hce[it, jt, ix, jx] * (
        m.Tgc[it, jt, ix, jx] - m.Tge[it, jt, ix, jx]) - \
                   m.Ax * (1 - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (
                       1. - m.ed[it, jt, ix, jx]) * m.rhos * m.ap * exp(m.whp[it, jt, ix, jx]) * (
                   m.Tge[it, jt, ix, jx] - m.Tse[it, jt, ix, jx]) - \
                   m.Hgbulk[it, jt, ix, jx] / m.hi_x[ix] - \
                   m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (1. - m.ed[it, jt, ix, jx]) * \
                   sum(m.rge[it, jt, ix, jx, k] * m.cpgcge[k] for k in m.sp2) * m.Tge[it, jt, ix, jx]) * m.hi_t[it]
    else:
        return Constraint.Skip



# put derivative space here
# equation A.9 Solid phase adsorbed species balance
# dNse_dt
def de_nse_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dNse_dt[it, jt, ix, jx, k] * m.hi_x[ix] == \
               (m.dcein_dx[it, jt, ix, jx, k] * m.Ax + m.Ksbulk[it, jt, ix, jx, k] + \
                m.hi_x[ix] * m.Ax * m.delta[it, jt, ix, jx] * m.rhos * m.Kcebs[it, jt, ix, jx] * (
                m.nc[it, jt, ix, jx, k] - m.ne[it, jt, ix, jx, k]) + \
                m.hi_x[ix] * m.Ax * (
                1 - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (
                1 - m.ed[it, jt, ix, jx]) * m.rse[it, jt, ix, jx, k]) * m.hi_t[it]
    else:
        return Constraint.Skip

# put derivative space here
# equation A.10 Solid phase energy balance
# dHse_dt
def de_hse_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dHse_dt[it, jt, ix, jx] * m.hi_x[ix] == \
               (m.deein_dx[it, jt, ix, jx] * m.Ax + m.Hsbulk[it, jt, ix, jx] + \
                m.hi_x[ix] * m.Ax * m.delta[it, jt, ix, jx] * m.rhos * m.Kcebs[it, jt, ix, jx] * (
                m.hsc[it, jt, ix, jx] - m.hse[it, jt, ix, jx]) + \
                m.hi_x[ix] * m.Ax * (
                1 - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (
                1 - m.ed[it, jt, ix, jx]) * \
                sum((m.rge[it, jt, ix, jx, k] * m.cpgcge[k]) for k in m.sp2) * m.Tge[it, jt, ix, jx] + \
                m.hi_x[ix] * m.Ax * (
                1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (
                1. - m.ed[it, jt, ix, jx]) * m.rhos * m.ap * exp(m.whp[it, jt, ix, jx]) * (
                m.Tge[it, jt, ix, jx] - m.Tse[it, jt, ix, jx]) + \
                m.hi_x[ix] * m.pi * m.dx * m.ht[it, jt, ix, jx] * m.dThx[it, jt, ix, jx] * m.Nx * m.Cr) * m.hi_t[it]
    else:
        return Constraint.Skip

# shift the AV?
# dz_dx
def dex_z_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dz_dx[it, jt, ix, jx] == 0
    else:
        return Constraint.Skip


# Kgbulk
def i1_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Kgbulk[it, jt, ix, jx, k] == m.K_d * (sum(m.ce[it, jt, ix, jx, kx] for kx in m.sp) - sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp)) * m.yb[it, jt, ix, jx, k]
    else:
        return Constraint.Skip

# Hgbulk
def i2_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hgbulk[it, jt, ix, jx] == m.K_d * (sum(m.ce[it, jt, ix, jx, kx] for kx in m.sp) - sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp)) * m.cpg_mol * \
                                           m.Tgb[it, jt, ix, jx]
    else:
        return Constraint.Skip

# Kgbulk
# oddly derivative looking term here and in the next one
# definetly derivatives e19 and e20 from bfb ss paper
def i3_rule(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.Ksbulk[it, kt, ix, kx, c] == \
               -m.Ax * sum(m.lydot[jx, kx] * m.Jc[it, kt, ix, jx] for jx in m.cp_x if 0 < jx <= m.ncp_x) * m.ne[it, kt, ix, kx, c]
    else:
        return Constraint.Skip


# Hsbulk
# m.Jc[it, jt, ix, jx]-m.Jc[i-1]
def i4_rule(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.Hsbulk[it, kt, ix, kx] == \
               -m.Ax * sum(m.lydot[jx, kx] * m.Jc[it, kt, ix, jx] for jx in m.cp_x if 0 < jx <= m.ncp_x) * m.hse[
                   it, kt, ix, kx]
    # elif j == m.ncp_x:
    #     return m.Hsbulk[it, jt, ix, jx] == -m.Ax * (m.Jc[it, jt, ix, jx] - m.Jc[i, j - 1]) * m.hse[it, jt, ix, jx]
    else:
        return Constraint.Skip


# db
def i5_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.db[it, jt, ix, jx] == m.dbu[it, jt, ix, jx]
    else:
        return Constraint.Skip

# vb
def i6_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.vb[it, jt, ix, jx] == \
               1.55 * ((m.vg[it, jt, ix, jx] - m.vmf[it, jt]) + 14.1 * (m.db[it, jt, ix, jx] + 0.005)) * (
               m.Dte ** 0.32) + m.vbr[it, jt, ix, jx]
    else:
        return Constraint.Skip

# ed
def i7_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return (1 - m.emf) * ((m.dp ** 0.1) * (m.gc ** 0.118) * 2.05 * (m.l[ix, jx] ** 0.043)) == \
               2.54 * (m.mug ** 0.066) * (1. - m.ed[it, jt, ix, jx])
    else:
        return Constraint.Skip

# ve
def i8_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ve[it, jt, ix, jx] * ((m.dp ** 0.568) * (m.gc ** 0.663) * (0.08518 * (m.rhos - m.rhog[it, jt, ix, jx]) + 19.09) * ((m.l[ix, jx]) ** 0.244)) == \
               m.vmf[it, jt] * 188. * 1.02 * (m.mug ** 0.371)
    else:
        return Constraint.Skip


# exchanger pressure drop
# HXIn_h
def e1_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.HXIn_h[it, jt] == -0.2831 - 2.9863e-6 * (m.HXIn_P - 1.3) + 7.3855e-05 * (m.HXIn_T - 60)
    else:
        return Constraint.Skip



# hsint
def e5_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.hsint[it, jt] == \
               ((m.nin['h'] + m.nin['c']) * (m.cpgcst['h'] * m.SolidIn_T + m.dH1) + m.nin['c'] * (m.cpgcst['c'] * m.SolidIn_T + m.dH2) + m.nin['n'] * (m.cpgcst['c'] * m.SolidIn_T + m.dH3)) * 1E-3 + m.cps * m.SolidIn_T
    else:
        return Constraint.Skip


# cein
def fdvar_x_cein_(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dcein_dx[it, kt, ix, kx, c] == \
               sum(m.ldot_x[jx, kx] * m.cein[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# ecwin
def fdvar_x_ecwin_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.decwin_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.ecwin[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# eein
def fdvar_x_eein_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.deein_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.eein[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# hxh:q

def fdvar_x_hxh_(m, it, kt, ix, kx):  #
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dhxh_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.hxh[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# Phx
def fdvar_x_phx_(m, it, kt, ix, kx):  #
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dPhx_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.Phx[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# ccwin
def fdvar_x_ccwin_(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dccwin_dx[it, kt, ix, kx, c] == \
               sum(m.ldot_x[jx, kx] * m.ccwin[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# z
def fdvar_z_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dz_dx[it, kt, ix, kx] == sum(
            m.ldot_x[jx, kx] * m.z[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# ceini0
def fcp_x_cein(m, it, kt, ix, c):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.cein[it, kt, ix + 1, 0, c] == \
               sum(m.l1_x[jx] * m.cein[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# ecwini0
def fcp_x_ecwin(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.ecwin[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.ecwin[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# eeini0
def fcp_x_eein(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.eein[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.eein[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# hxhi0
def fcp_x_hxh(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.hxh[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.hxh[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# Phxi0
def fcp_x_phx(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.Phx[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.Phx[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# ccwini0
def fcp_x_ccwin(m, it, kt, ix, c):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.ccwin[it, kt, ix + 1, 0, c] == \
               sum(m.l1_x[jx] * m.ccwin[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# zi0
def fcp_z_(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.z[it, kt, ix + 1, 0] == sum(m.l1_x[jx] * m.z[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


def fzl_cein(m, it, kt, c):
    if 0 < kt <= m.ncp_t:
        return m.cein_l[it, kt, c] == sum(m.l1_x[jx] * m.cein[it, kt, m.nfe_x, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


def fzl_ecwin(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.ecwin_l[it, kt] == sum(m.l1_x[jx] * m.ecwin[it, kt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_eein(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.eein_l[it, kt] == sum(m.l1_x[jx] * m.eein[it, kt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_hxh(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.hxh_l[it, kt] == sum(m.l1_x[jx] * m.hxh[it, kt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_phx(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.Phx_l[it, kt] == sum(m.l1_x[jx] * m.Phx[it, kt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_ccwin(m, it, kt, c):
    if 0 < kt <= m.ncp_t:
        return m.ccwin_l[it, kt, c] == sum(m.l1_x[jx] * m.ccwin[it, kt, m.nfe_x, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# def fzl_z(m, it, kt):
#     if 0 < kt <= m.ncp_t:
#         return m.z_l[it, kt] ==
#     else:
#         return Constraint.Skip

# hse_l
def fyl_hse(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.hse_l[it, jt] == sum(m.l1y[jx] * m.hse[it, jt, m.nfe_x, jx] for jx in m.cp_x if 0 < jx <= m.ncp_x)
    else:
        return Constraint.Skip

# ne_l
def fyl_ne(m, it, jt, c):
    if 0 < jt <= m.ncp_t:
        return m.ne_l[it, jt, c] == sum(m.l1y[jx] * m.ne[it, jt, m.nfe_x, jx, c] for jx in m.cp_x if 0 < jx <= m.ncp_x)
    else:
        return Constraint.Skip

# cb_l
def fzl_cb(m, it, jt, c):
    if 0 < jt <= m.ncp_t:
        return m.cb_l[it, jt, c] == sum(m.l1_x[jx] * m.cb[it, jt, m.nfe_x, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# ccapture
def cc_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.c_capture[it, jt] == \
               1 - (m.GasOut_F[it, jt] * m.GasOut_z[it, jt, 'c']) / (m.GasIn_F[it] * m.GasIn_z[it, 'c'])
    else:
        return Constraint.Skip


# 1st order Derivative variables in space
# ddx vb
def fdvar_x_vg_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dvg_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.vg[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# ddx Gb
def fdvar_x_Gb_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dGb_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.Gb[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# ddx cb
def fdvar_x_cb_(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dcb_dx[it, kt, ix, kx, c] == \
               sum(m.ldot_x[jx, kx] * m.cb[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# ddx Tb
def fdvar_x_Tgb_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dTgb_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.Tgb[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# ddx P (maybe not)
def fdvar_x_P_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dP_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.P[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def a21_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) == m.P[it, jt, ix, jx] * 100 / (
        8.314 * (m.Tgb[it, jt, ix, jx] + 273.16))
    else:
        return Constraint.Skip

#  Continuation in space
# vb
def fcp_x_vb(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.vg[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.vg[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# cb
def fcp_x_cb(m, it, kt, ix, c):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.cb[it, kt, ix + 1, 0, c] == \
               sum(m.l1_x[jx] * m.cb[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# Tgb
def fcp_x_Tgb(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.Tgb[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.Tgb[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# P
def fcp_x_P(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.P[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.P[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# 2nd order Derivative variables in space
# d2dx2 vb
def fdvar_x_dvg_dx_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dvgx_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.vgx[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# d2dx2 cb
def fdvar_x_dcb_dx_(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dcbx_dx[it, kt, ix, kx, c] == \
               sum(m.ldot_x[jx, kx] * m.cbx[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# d2dx2 Tb
def fdvar_x_dTgb_dx_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dTgbx_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.Tgbx[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

#  Continuation in space d2dx2
# dvb_dx
def fcp_x_dvb_dx(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.vgx[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.vgx[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# dcb_dx
def fcp_x_dcb_dx(m, it, kt, ix, c):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.cbx[it, kt, ix + 1, 0, c] == \
               sum(m.l1_x[jx] * m.cbx[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# dTgb_dx
def fcp_x_dTgb_dx(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.Tgbx[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.Tgbx[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# time vars
def mom_rule(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.mom[it, kt, ix, kx] == (m.cb[it, kt, ix, kx, 'c'] * 44.01 + m.cb[it, kt, ix, kx, 'n'] * 28.01 + m.cb[it, kt, ix, kx,'h'] * 18.02) * m.vg[it, kt, ix, kx]
    else:
        return Constraint.Skip

# Time discretiation mom
def fdvar_t_mom(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dmom_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.mom[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

def fcp_t_mom(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.mom[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.mom[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip


# gas bubble
def ngb_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ngb[it, jt, ix, jx, c] == m.delta[it, jt, ix, jx] * m.cb[it, jt, ix, jx, c]
    else:
        return Constraint.Skip


def hgb_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hgb[it, jt, ix, jx] == \
               m.delta[it, jt, ix, jx] * sum(m.cb[it, jt, ix, jx, c] for c in m.sp) * m.cpg_mol * m.Tgb[it, jt, ix, jx]
    else:
        return Constraint.Skip


# Momentum balance
def alt_de_Gb_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.hi_x[ix] * m.dmom_dt[it, jt, ix, jx] == \
               -m.hi_t[it] * (2 * (m.cb[it, jt, ix, jx, 'c'] * 44.01 + m.cb[it, jt, ix, jx, 'n'] * 28.01 + m.cb[it, jt, ix, jx,'h'] * 18.02) * m.vg[it, jt, ix, jt] * m.dvg_dx[it, jt, ix, jx] + (m.vg[it, jt, ix, jt]**2) * m.drhog_dx[it, jt, ix, jx]) - \
               m.hi_t[it] * m.mug * m.dvgx_dx[it, jt, ix, jx] - m.hi_t[it] * m.dP_dx[it, jt, ix, jx] * 100000 - \
               m.hi_t[it] * m.hi_x[ix] * (1 - m.e[it, jt, ix, jx]) * m.rhos * m.gc
    else:
        return Constraint.Skip


def dum_dex_vg_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dvg_dx[it, jt, ix, jx] == m.hi_x[ix] * m.vgx[it, jt, ix, jx]
    else:
        return Constraint.Skip


# Continuity (mole balance) species
def de_ngb_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.hi_x[ix] * m.dNgb_dt[it, jt, ix, jx, k] == \
               (-m.hi_t[it] * (m.vg[it, jt, ix, jx] * m.dcb_dx[it, jt, ix, jx, k] + m.cb[it, jt, ix, jx, k] * m.dvg_dx[it, jt, ix, jx]) - \
               m.hi_t[it] * m.D[it, jt, ix, jx, k] * m.dcbx_dx[it, jt, ix, jx, k] - \
               m.hi_t[it] * m.hi_x[ix] * m.delta[it, jt, ix, jx] * m.Kbc[it, jt, ix, jx, k] * (m.cb[it, jt, ix, jx, k] - m.cc[it, jt, ix, jx, k]) + \
               m.hi_t[it] * m.Kgbulk[it, jt, ix, jx, k]/m.Ax)
    else:
        return Constraint.Skip


def dum_dex_cb_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dcb_dx[it, jt, ix, jx, c] == m.hi_x[ix] * m.cbx[it, jt, ix, jx, c]
    else:
        return Constraint.Skip


# Energy balance (mole balance)
def de_hgb_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return (1e+02) * (m.hi_x[ix] * m.dHgb_dt[it, jt, ix, jx]) == \
               (-(m.hi_t[it] * m.cpg_mol) * (m.vg[it, jt, ix, jt] * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) * m.dTgb_dx[it, jt, ix, jx] + m.Tgb[it, jt, ix, jx] * (m.vg[it, jt, ix, jt] * sum(m.dcb_dx[it, jt, ix, jx, kx] for kx in m.sp) + sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) * m.dvg_dx[it, jt, ix, jx])) -\
               m.hi_t[it] * 1e-03 * m.kg * m.dTgbx_dx[it, jt, ix, jx] - m.hi_t[it] * m.hi_x[ix] * m.delta[it, jt, ix, jx] * m.Hbc[it, jt, ix, jx] * (m.Tgb[it, jt, ix, jx] - m.Tgc[it, jt, ix, jx]) + \
               m.hi_t[it] * m.Hgbulk[it, jt, ix, jx]/m.Ax) * (1e+02)
    else:
        return Constraint.Skip
    # if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
    #     return m.hi_x[ix] * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) * m.dHgb_dt[it, jt, ix, jx] == \
    #            -(m.hi_t[it] * m.cpg_mol/m.Ax) * (m.Gb[it, jt, ix, jt] * m.dTgb_dx[it, jt, ix, jx]) - \
    #            m.hi_t[it] * m.kg * m.dTgbx_dx[it, jt, ix, jx] - \
    #            m.hi_t[it] * m.hi_x[ix] * m.delta[it, jt, ix, jx] * m.Hbc[it, jt, ix, jx] * (m.Tgb[it, jt, ix, jx] - m.Tgc[it, jt, ix, jx]) + \
    #            m.hi_t[it] * m.Hgbulk[it, jt, ix, jx]/m.Ax
    # else:
    #     return Constraint.Skip


def dum_dex_Tgb_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dTgb_dx[it, jt, ix, jx] == m.hi_x[ix] * m.Tgbx[it, jt, ix, jx]
    else:
        return Constraint.Skip


# Pressure term
def dpdx_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return (m.dP_dx[it, jt, ix, jx]) * 100/8.314 == \
               sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) * m.dTgb_dx[it, jt, ix, jx] + \
               (m.Tgb[it, jt, ix, jx] + 273.16) * sum(m.dcb_dx[it, jt, ix, jx, kx] for kx in m.sp)
    else:
        return Constraint.Skip


# Heat-Exchanger fluid energy balance
# dhxh_dx
def dhxh_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return 0 == (m.HXIn_F / 3600) * m.dhxh_dx[it, jt, ix, jx] - \
                    m.hi_x[ix] * 1E-6 * m.pi * m.dx * m.ht[it, jt, ix, jx] * m.dThx[it, jt, ix, jx] * m.Nx * m.Cr
    else:
        return Constraint.Skip

# dPhx_dx
def dphx_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dPhx_dx[it, jt, ix, jx] == m.hi_x[ix] * m.dPhx + m.hi_x[ix] * m.rhohx * 1E-5
    else:
        return Constraint.Skip


# Gb0
def bc_Gb0_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.Gb[it, jt, 1, 0] == m.GasIn_F[it] / 3600
    else:
        return Constraint.Skip


# Tgb0
def bc_Tgb0_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.Tgb[it, jt, 1, 0] == m.GasIn_T[it]
    else:
        return Constraint.Skip


# cb0
def bc_cb0_rule(m, it, jt, k):
    if 0 < jt <= m.ncp_t:
        return m.cb[it, jt, 1, 0, k] == m.GasIn_z[it, k] * m.GasIn_P[it] * 100 / (8.314 * (m.GasIn_T[it] + 273.16))
    else:
        return Constraint.Skip


# P0
def bc_P0_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.P[it, jt, 1, 0] == m.GasIn_P[it]
    else:
        return Constraint.Skip


# vg0
def bc_vg0_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.vg[it, jt, 1, 0] == m.GasIn_F[it] / (3600 * m.Ax * m.GasIn_P[it] * 100 / (8.314 * (m.GasIn_T[it] + 273.16)))
    else:
        return Constraint.Skip


# bcs ddx vars
# vgx_l
def bc_vgx_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return sum(m.l1_x[jx] * m.vgx[it, jt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x) == 0.0
    else:
        return Constraint.Skip


# cbx_l
def bc_cbx_rule(m, it, jt, c):
    if 0 < jt <= m.ncp_t:
        return sum(m.l1_x[jx] * m.cbx[it, jt, m.nfe_x, jx, c] for jx in m.cp_x if jx <= m.ncp_x) == 0.0
    else:
        return Constraint.Skip


# Tgbx_l
def bc_Tgbx_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return sum(m.l1_x[jx] * m.Tgbx[it, jt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x) == 0.0
    else:
        return Constraint.Skip




# GasOut_F
def e12_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return (1e+04) * sum(m.l1_x[jx] * m.vg[it, jt, m.nfe_x, jx] * sum(m.cb[it, jt, m.nfe_x, jx, kx] for kx in m.sp) for jx in m.cp_x if 0 < jx <= m.ncp_x) == (m.GasOut_F[it, jt] / (3600 * m.Ax))* (1e+04)
    else:
        return Constraint.Skip

# GasOut_T
def e13_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return sum(m.l1_x[jx] * m.Tgb[it, jt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x) == m.GasOut_T[it, jt]
    else:
        return Constraint.Skip

# GasOut_z
def e14_rule(m, it, jt, c):
    if 0 < jt <= m.ncp_t:
        return m.GasOut_z[it, jt, c] * sum(m.cb_l[it, jt, cx] for cx in m.sp) == m.cb_l[it, jt, c]
    else:
        return Constraint.Skip


# Sot -- not bc tough
def e20_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.Sit[it] - m.Sot[it, jt] == sum(m.l1_x[jx] * m.z[it, jt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x) * m.Ax
    else:
        return Constraint.Skip

# ccwin_l or cein_l
def bc_mol_rule(m, it, jt, j):
    if 0 < jt <= m.ncp_t:
        return m.ccwin_l[it, jt, j] * m.Ax + m.Sit[it] * m.nin[j] == m.cein_l[it, jt, j] * m.Ax + m.Sot[it, jt] * m.ne_l[it, jt, j]
    else:
        return Constraint.Skip

# eein_l or eein_l
def bc_ene_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.ecwin_l[it, jt] * m.Ax + m.Sit[it] * m.hsint[it, jt] == m.eein_l[it, jt] * m.Ax + m.Sot[it, jt] * m.hse_l[it, jt]
    else:
        return Constraint.Skip

# ccwin or cein
def bc_mol0_rule(m, it, jt, c):
    if 0 < jt <= m.ncp_t:
        return m.ccwin[it, jt, 1, 0, c] == m.cein[it, jt, 1, 0, c]
    else:
        return Constraint.Skip

# ecwin or eein
def bc_ene0_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.ecwin[it, jt, 1, 0] == m.eein[it, jt, 1, 0]
    else:
        return Constraint.Skip

# z0
def bc_z0_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.z[it, jt, 1, 0] == 0
    else:
        return Constraint.Skip

# HXIn_h
def bc_hxh_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.hxh_l[it, jt] == m.HXIn_h[it, jt]
    else:
        return Constraint.Skip

# Phx_l
def bc_phx_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.Phx_l[it, jt] == m.HXIn_P
    else:
        return Constraint.Skip


def ic_mom_rule(m, ix, jx):
    if 0 < jx <= m.ncp_x:
        return m.mom[1, 0, ix, jx] == m.mom_ic[(ix, jx)]
    else:
        return Constraint.Skip


def drhogx_rule(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.drhog_dx[it, kt, ix, kx] == \
               (m.dcb_dx[it, kt, ix, kx, 'c'] * 44.01 +
                m.dcb_dx[it, kt, ix, kx, 'n'] * 28.01 +
                m.dcb_dx[it, kt, ix, kx,'h'] * 18.02)
    else:
        return Constraint.Skip
