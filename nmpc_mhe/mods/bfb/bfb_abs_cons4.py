
from __future__ import division

from pyomo.core.base import Constraint, sqrt, exp, Expression
from nmpc_mhe.aux.cpoinsc import collptsgen
from nmpc_mhe.aux.lagrange_f import lgr, lgry, lgrdot, lgrydot

"""
Version Note that the reformulation kind of works, i.e. dvar for vg.
Stopped development on this one 
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
        # if c == 'h':
        return m.Ngb[1, 0, ix, jx, c] == m.Ngb_ic[(ix, jx, c)]
        # elif c == 'n':
        #     return m.Ngb[1, 0, ix, jx, c] == m.Ax * m.delta_0[ix, jx] * m.cb_n_0[ix, jx]
        # elif c == 'c':
        #     return m.Ngb[1, 0, ix, jx, c] == m.Ax * m.delta_0[ix, jx] * m.cb_c_0[ix, jx]
    else:
        return Constraint.Skip

def ic_hgb_rule(m, ix, jx):
    if 0 < jx <= m.ncp_x:
        return m.Hgb[1, 0, ix, jx] == m.Hgb_ic[(ix, jx)]
        # return m.Hgb[1, 0, ix, jx] == m.Hgb0_imp[ix, jx, c]
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

# solids in the bed
def ic_ws_rule(m, ix, jx):
    if 0 < jx <= m.ncp_x:
        return m.Ws[1, 0, ix, jx] == m.Ws_ic[(ix, jx)]
    else:
        return Constraint.Skip


# expr ================================================================================================

# expr ================================================================================================

# gas bubble
def ngb_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ngb[it, jt, ix, jx, c] == \
               m.Ax * m.delta[it, jt, ix, jx] * \
               m.cb[it, jt, ix, jx, c]
    else:
        return Constraint.Skip

def hgb_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hgb[it, jt, ix, jx] == \
               m.Ax * m.delta[it, jt, ix, jx] * \
               sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol * m.Tgb[it, jt, ix, jx]
    else:
        return Constraint.Skip

# gas cloud wake
def ngc_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ngc[it, jt, ix, jx, c] == \
               m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * m.ed[it, jt, ix, jx] * \
               m.cc[it, jt, ix, jx, c]
    else:
        return Constraint.Skip

def hgc_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hgc[it, jt, ix, jx] == \
               m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * m.ed[it, jt, ix, jx] * \
               sum(m.cc[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol * m.Tgc[it, jt, ix, jx]
    else:
        return Constraint.Skip

# solid cloud wake
def nsc_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Nsc[it, jt, ix, jx, c] == \
               m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) * m.rhos * \
               m.nc[it, jt, ix, jx, c]
    else:
        return Constraint.Skip

def hsc_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hsc[it, jt, ix, jx] == \
               m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) * m.rhos * \
               m.cps * m.Tsc[it, jt, ix, jx]
    else:
        return Constraint.Skip

# gas emulsion
def nge_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Nge[it, jt, ix, jx, c] == \
               m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * m.ed[
                   it, jt, ix, jx] * \
               m.ce[it, jt, ix, jx, c]
    else:
        return Constraint.Skip

def hge_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hge[it, jt, ix, jx] == \
               m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * m.ed[
                   it, jt, ix, jx] * \
               sum(m.ce[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol * m.Tge[it, jt, ix, jx]
    else:
        return Constraint.Skip

# solids emulsion
def nse_rule(m, it, jt, ix, jx, c):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Nse[it, jt, ix, jx, c] == \
               m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * \
               (1. - m.ed[it, jt, ix, jx]) * m.rhos * \
               m.ne[it, jt, ix, jx, c]
    else:
        return Constraint.Skip

def hse_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hse[it, jt, ix, jx] == \
               m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * \
               (1. - m.ed[it, jt, ix, jx]) * m.rhos * \
               m.cps * m.Tse[it, jt, ix, jx]
    else:
        return Constraint.Skip

# solids in the bed
def ws_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ws[it, jt, ix, jx] == \
               m.Ax * (1. - m.delta[it, jt, ix, jx]) * (1. - m.ed[it, jt, ix, jx]) * m.rhos
    else:
        return Constraint.Skip


# expr ================================================================================================
#

def fdvar_t_ngb(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dNgb_dt[it, kt, ix, kx, c] == \
               sum(m.ldot_t[jt, kt] * m.Ngb[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

def fdvar_t_hgb(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dHgb_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Hgb[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

#
def fdvar_t_ngc(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dNgc_dt[it, kt, ix, kx, c] == \
               sum(m.ldot_t[jt, kt] * m.Ngc[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

def fdvar_t_hgc(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dHgc_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Hgc[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

#
def fdvar_t_nsc(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dNsc_dt[it, kt, ix, kx, c] == \
               sum(m.ldot_t[jt, kt] * m.Nsc[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

def fdvar_t_hsc(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dHsc_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Hsc[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

#
def fdvar_t_nge(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dNge_dt[it, kt, ix, kx, c] == \
               sum(m.ldot_t[jt, kt] * m.Nge[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

def fdvar_t_hge(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dHge_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Hge[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

#
def fdvar_t_nse(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dNse_dt[it, kt, ix, kx, c] == \
               sum(m.ldot_t[jt, kt] * m.Nse[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

def fdvar_t_hse(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dHse_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Hse[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip

def fdvar_t_ws(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dWs_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.Ws[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip



# expr ================================================================================================
#

def fcp_t_ngb(m, it, ix, kx, c):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Ngb[it + 1, 0, ix, kx, c] - \
               sum(m.l1_t[jt] * m.Ngb[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

def fcp_t_hgb(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Hgb[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Hgb[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

#
def fcp_t_ngc(m, it, ix, kx, c):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Ngc[it + 1, 0, ix, kx, c] - \
               sum(m.l1_t[jt] * m.Ngc[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

def fcp_t_hgc(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Hgc[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Hgc[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

#
def fcp_t_nsc(m, it, ix, kx, c):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Nsc[it + 1, 0, ix, kx, c] - \
               sum(m.l1_t[jt] * m.Nsc[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

def fcp_t_hsc(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Hsc[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Hsc[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

#
def fcp_t_nge(m, it, ix, kx, c):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Nge[it + 1, 0, ix, kx, c] - \
               sum(m.l1_t[jt] * m.Nge[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

def fcp_t_hge(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Hge[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Hge[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

#
def fcp_t_nse(m, it, ix, kx, c):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Nse[it + 1, 0, ix, kx, c] - \
               sum(m.l1_t[jt] * m.Nse[it, jt, ix, kx, c] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

def fcp_t_hse(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Hse[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Hse[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

def fcp_t_ws(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.Ws[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.Ws[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip

#

# def fhub_p(m):
#     return min([_maxh, fm.lenleft])
#
#
# self.hub = Param(initialize=ic.fhub_p)
#

# constraint on the upper bound of the lengt
# def fhup(m):
#     return m.hi <= m.hub
#
#
# self.chup = Constraint(rule=fhup)
# ???//////////////////////////////////////////////???

# ???//////////////////////////////////////////////???
#
#
# i, j, k, l
# it, jt, ix, jx, k
def a1_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.vg[it, jt, ix, jx] * m.Ax * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) == sum(m.cbin[it, jt, ix, jx, kx] for kx in m.sp)
    else:
        return Constraint.Skip



# bc_ebin
def a3_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ebin[it, jt, ix, jx] == sum(m.cbin[it, jt, ix, jx, kx] for kx in m.sp)* m.cpg_mol * m.Tgb[it, jt, ix, jx]
    elif 0 < jt <= m.ncp_t and jx == 0 and ix == 1:
        return m.ebin[it, jt, ix, jx] == sum(m.cbin[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol * m.Tgb[it, jt, ix, jx]
    else:
        return Constraint.Skip



# ic
def a4_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ecwin[it, jt, ix, jx] == m.Jc[it, jt, ix, jx] * m.hsc[it, jt, ix, jx]
    else:

        return Constraint.Skip



def a5_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.eein[it, jt, ix, jx] == m.Je[it, jt, ix, jx] * m.hse[it, jt, ix, jx]
    # elif j == 0 and i == 1:
    #     return m.eein[it, jt, ix, jx] == m.Je[it, jt, ix, jx] * m.hse[it, jt, ix, jx]
    else:
        return Constraint.Skip



# bc_cbin

# def a7_rule(m, it, jt, ix, jx, k):
#     if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
#         return m.cbin[it, jt, ix, jx, k] == m.yb[it, jt, ix, jx, k] * sum(m.cbin[it, jt, ix, jx, kx] for kx in m.sp)
#     elif 0 < jt <= m.ncp_t and jx == 0 and ix == 1:
#         return m.cbin[it, jt, ix, jx, k] == m.yb[it, jt, ix, jx, k] * sum(m.cbin[it, jt, ix, jx, kx] for kx in m.sp)
#     else:
#         return Constraint.Skip


# ic_ccwin
def a8_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ccwin[it, jt, ix, jx, k] == m.Jc[it, jt, ix, jx] * m.nc[it, jt, ix, jx, k]
    # elif j == 0 and i == 1:
    #     return m.ccwin[it, jt, ix, jx, k] == m.Jc[it, jt, ix, jx] * m.nc[it, jt, ix, jx, k]
    else:
        return Constraint.Skip


def a9_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.cein[it, jt, ix, jx, k] == m.Je[it, jt, ix, jx] * m.ne[it, jt, ix, jx, k]
    # elif j == 0 and i == 1:
    #     return m.cein[it, jt, ix, jx, k] == m.Je[it, jt, ix, jx] * m.ne[it, jt, ix, jx, k]
    else:
        return Constraint.Skip


def a11_rule_2(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.z[it, jt, ix, jx] == m.Je[it, jt, ix, jx] - m.Jc[it, jt, ix, jx]
        # return m.dJc_dx[i, j] == m.dummyJ[i, j]
    else:
        return Constraint.Skip





def a13_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return sum(m.cbin[it, jt, ix, jx, kx] for kx in m.sp) == m.vb[it, jt, ix, jx] * m.Ax * m.delta[it, jt, ix, jx] * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp)
    else:
        return Constraint.Skip



def a14_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Jc[it, jt, ix, jx] == \
               m.fw * m.delta[it, jt, ix, jx] * m.rhos * (1 - m.ed[it, jt, ix, jx]) * m.vb[it, jt, ix, jx]
    else:
        return Constraint.Skip



def a15_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x and k != "n":
        return m.cb[it, jt, ix, jx, k] == m.yb[it, jt, ix, jx, k] * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp)
    else:
        return Constraint.Skip



def a16_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x and k != "n":
        return m.cc[it, jt, ix, jx, k] == m.yc[it, jt, ix, jx, k] * sum(m.cc[it, jt, ix, jx, kx] for kx in m.sp)
    else:
        return Constraint.Skip


def a17_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x and k != "n":
        return m.ce[it, jt, ix, jx, k] == m.ye[it, jt, ix, jx, k] * sum(m.ce[it, jt, ix, jx, kx] for kx in m.sp)
    else:
        return Constraint.Skip


def a18_rule(m, it, jt, ix, jx):
     if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
# #         return m.cet[it, jt, ix, jx] == sum(m.ce[it, jt, ix, jx, kx] for kx in m.sp)
        return sum(m.ye[it, jt, ix, jx, kx] for kx in m.sp) == 1
     else:
         return Constraint.Skip

def a19_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        # return sum(m.cc[it, jt, ix, jx, kx] for kx in m.sp) == sum(m.cc[it, jt, ix, jx, k] for k in m.sp)
        return sum(m.yc[it, jt, ix, jx, kx] for kx in m.sp) == 1
    else:
        return Constraint.Skip



def a20_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return sum(m.yb[it, jt, ix, jx, kx] for kx in m.sp) == 1
        # return m.cbt[it, jt, ix, jx] == sum(m.cb[it, jt, ix, jx, k] for k in m.sp)
    else:
        return Constraint.Skip



# bc_P
def a21_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) == m.P[it, jt, ix, jx] * 100 / (8.314 * (m.Tgb[it, jt, ix, jx] + 273.16))
    else:
        return Constraint.Skip



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



# density
def a25_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rhog[it, jt, ix, jx] == m.P[it, jt, ix, jx] * 100 * (
            m.ye[it, jt, ix, jx, 'c'] * 44.01 + m.ye[it, jt, ix, jx, 'n'] * 28.01 + m.ye[
                it, jt, ix, jx, 'h'] * 18.02) \
                                         / (8.314 * (m.Tge[it, jt, ix, jx] + 273.16))
    else:
        return Constraint.Skip



def a26_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ar[it, jt, ix, jx] == \
               (m.dp ** 3) * m.rhog[it, jt, ix, jx] * (m.rhos - m.rhog[it, jt, ix, jx]) * m.gc / (m.mug ** 2)
    else:
        return Constraint.Skip



def a27_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return (1 - m.e[it, jt, ix, jx]) == (1 - m.ed[it, jt, ix, jx]) * (1 - m.delta[it, jt, ix, jx])
    else:
        return Constraint.Skip



def a28_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.vbr[it, jt, ix, jx] == 0.711 * sqrt(m.gc * m.db[it, jt, ix, jx])
    else:
        return Constraint.Skip



def a29_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.db0[it, jt] == 1.38 * (m.gc ** (-0.2)) * ((m.vg[it, jt, 1, 1] - m.ve[it, jt, 1, 1]) * m.Ao) ** 0.4
    else:
        return Constraint.Skip



def a30_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dbe[it, jt, ix, jx] == (m.Dt / 4) * (-m.g1[it, jt] + m.g3[it, jt, ix, jx]) ** 2
    else:
        return Constraint.Skip



def a31_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dbm[it, jt, ix, jx] == 2.59 * (m.gc ** (-0.2)) * ((m.vg[it, jt, ix, jx] - m.ve[it, jt, ix, jx]) * m.Ax) ** 0.4
    else:
        return Constraint.Skip


def a32_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.g1[it, jt] == 2.56E-2 * sqrt(m.Dt / m.gc) / m.vmf[it, jt]
    else:
        return Constraint.Skip



def a33_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return 4 * m.g2[it, jt, ix, jx] == m.Dt * (m.g1[it, jt] + m.g3[it, jt, ix, jx]) ** 2
    else:
        return Constraint.Skip



def a34_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.g3[it, jt, ix, jx] == sqrt(m.g1[it, jt] ** 2 + 4 * m.dbm[it, jt, ix, jx] / m.Dt)
    else:
        return Constraint.Skip


# x included?
def a35_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return exp(0.3 * (m.l[ix, jx]) / m.Dt) * (((sqrt(m.dbu[it, jt, ix, jx]) - sqrt(m.dbe[it, jt, ix, jx])) / (
        sqrt(m.db0[it, jt]) - sqrt(m.dbe[it, jt, ix, jx]))) ** (
                    1 - m.g1[it, jt] / m.g3[it, jt, ix, jx])) == \
    (((sqrt(m.dbu[it, jt, ix, jx]) - sqrt(m.g2[it, jt, ix, jx])) / (
    sqrt(m.db0[it, jt]) - sqrt(m.g2[it, jt, ix, jx]))) ** -(1 + m.g1[it, jt] / m.g3[it, jt, ix, jx]))
    else:
        return Constraint.Skip



def a36_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.fc[it, jt, ix, jx] == 3. * (m.vmf[it, jt] / m.emf) / (
        m.vbr[it, jt, ix, jx] - (m.vmf[it, jt] / m.emf))
    else:
        return Constraint.Skip



def a37_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.fcw[it, jt, ix, jx] == m.fc[it, jt, ix, jx] + m.fw
    else:
        return Constraint.Skip



def a38_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Kbc[it, jt, ix, jx, k] == \
               1.32 * 4.5 * (m.vmf[it, jt] / m.db[it, jt, ix, jx]) + 5.85 * (
                   ((m.D[it, jt, ix, jx, k] * 1E-4) ** 0.5) * (m.gc ** 0.25) / (m.db[it, jt, ix, jx] ** (5 / 4)))
    else:
        return Constraint.Skip



def a39_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Kce[it, jt, ix, jx, k] == 6.77 * sqrt(
            m.ed[it, jt, ix, jx] * (m.D[it, jt, ix, jx, k] * 1E-4) * m.vbr[it, jt, ix, jx] / (
            m.db[it, jt, ix, jx] ** 3))
    else:
        return Constraint.Skip



def a40_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Kcebs[it, jt, ix, jx] == 3 * (1 - m.ed[it, jt, ix, jx]) / (
        (1 - m.delta[it, jt, ix, jx]) * m.ed[it, jt, ix, jx]) * (m.ve[it, jt, ix, jx] / m.db[it, jt, ix, jx])
    else:
        return Constraint.Skip



def a41_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hbc[it, jt, ix, jx] == 1.32 * 4.5 * m.vmf[it, jt] * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol / m.db[
            it, jt, ix, jx] + \
                                        5.85 * sqrt((m.kg / 1000) * sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol) * (
                                            m.gc ** 0.25) / (
                                            m.db[it, jt, ix, jx] ** (5 / 4))
    else:
        return Constraint.Skip



def a42_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hce[it, jt, ix, jx] == 6.78 * sqrt(
            m.ed[it, jt, ix, jx] * m.vb[it, jt, ix, jx] * (m.kg / 1000) * sum(m.cc[it, jt, ix, jx, kx] for kx in m.sp) * m.cpg_mol / (
            m.db[it, jt, ix, jx] ** 3))
    else:
        return Constraint.Skip



def a43_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Nup[it, jt, ix, jx] == 1000 * m.hp[it, jt, ix, jx] * m.dp / m.kg
    else:
        return Constraint.Skip



def a44_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Red[it, jt, ix, jx] == m.ve[it, jt, ix, jx] * m.dp * m.rhog[it, jt, ix, jx] / m.mug
    else:
        return Constraint.Skip



def a45_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Nup[it, jt, ix, jx] == 0.03 * (m.Red[it, jt, ix, jx] ** 1.3)
    else:
        return Constraint.Skip



def a46_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.kpa[it, jt, ix, jx] == \
               (3.58 - 2.5 * m.ed[it, jt, ix, jx]) * m.kg * ((m.kp / m.kg) ** (0.46 - 0.46 * m.ed[it, jt, ix, jx]))
    else:
        return Constraint.Skip



def a47_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.fn[it, jt, ix, jx] == m.vg[it, jt, ix, jx] / m.vmf[it, jt]
    else:
        return Constraint.Skip



def a48_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.tau[it, jt, ix, jx] == 0.44 * (
        (m.dp * m.gc / ((m.vmf[it, jt] ** 2) * ((m.fn[it, jt, ix, jx] - m.ah) ** 2))) ** 0.14) * (
                                            (m.dp / m.dx) ** 0.225)
    else:
        return Constraint.Skip



def a49_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.fb[it, jt, ix, jx] == 0.33 * (
        ((m.vmf[it, jt] ** 2) * ((m.fn[it, jt, ix, jx] - m.ah) ** 2) / (m.dp * m.gc)) ** 0.14)
    else:
        return Constraint.Skip



def a50_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.hd[it, jt, ix, jx] == \
               2 * sqrt((m.kpa[it, jt, ix, jx] / 1000) * m.rhos * m.cps * (1 - m.ed[it, jt, ix, jx]) / (
               m.pi * m.tau[it, jt, ix, jx]))
    else:
        return Constraint.Skip



def a51_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return 1000 * m.hl[it, jt, ix, jx] * m.dp / m.kg == 0.009 * (m.Ar[it, jt, ix, jx] ** 0.5) * (m.Pr ** 0.33)
    else:
        return Constraint.Skip



def a52_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ht[it, jt, ix, jx] == m.fb[it, jt, ix, jx] * m.hd[it, jt, ix, jx] + (1 - m.fb[it, jt, ix, jx]) * \
                                                                                     m.hl[it, jt, ix, jx]
    else:
        return Constraint.Skip



# pde
def a53_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dPhx_dx[it, jt, ix, jx] == m.hi_x[ix] * m.dPhx + m.hi_x[ix] * m.rhohx * 1E-5
    else:
        return Constraint.Skip



def a54_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dThx[it, jt, ix, jx] == m.Ttube[it, jt, ix, jx] - m.Tse[it, jt, ix, jx]
    else:
        return Constraint.Skip


def a55_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ht[it, jt, ix, jx] * m.dThx[it, jt, ix, jx] * m.Cr == \
               m.hw * (m.Thx[it, jt, ix, jx] - m.Ttube[it, jt, ix, jx])
    else:
        return Constraint.Skip



def a56_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Thx[it, jt, ix, jx] == 33.2104 + 14170.15 * (m.hxh[it, jt, ix, jx] + 0.285)
    else:
        return Constraint.Skip



def a57_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return 10 * 1.75 / (m.phis * m.emf ** 3) * (m.dp * m.vmf[it, jt] * m.rhog[it, jt, 1, 1] / m.mug) ** 2 + \
               10 * 150 * (1 - m.emf) / ((m.phis ** 2) * (m.emf ** 3)) * (
               m.dp * m.vmf[it, jt] * m.rhog[it, jt, 1, 1] / m.mug) \
               == \
               10 * m.dp ** 3 * m.rhog[it, jt, 1, 1] * (m.rhos - m.rhog[it, jt, 1, 1]) * m.gc / m.mug ** 2
    else:
        return Constraint.Skip




def a58_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.k1c[it, jt, ix, jx] == \
               m.A1 * (m.Tsc[it, jt, ix, jx] + 273.15) * exp(-m.E1 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip



def a59_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.k2c[it, jt, ix, jx] == \
               m.A2 * (m.Tsc[it, jt, ix, jx] + 273.15) * exp(-m.E2 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip



def a60_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.k3c[it, jt, ix, jx] == m.A3 * (m.Tsc[it, jt, ix, jx] + 273.15) * exp(
            -m.E3 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip



def a61_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.k1e[it, jt, ix, jx] == m.A1 * (m.Tse[it, jt, ix, jx] + 273.15) * exp(
            -m.E1 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip



def a62_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.k2e[it, jt, ix, jx] == m.A2 * (m.Tse[it, jt, ix, jx] + 273.15) * exp(
            -m.E2 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip



def a63_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.k3e[it, jt, ix, jx] == m.A3 * (m.Tse[it, jt, ix, jx] + 273.15) * exp(
            -m.E3 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)))
    else:
        return Constraint.Skip



def a64_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ke1c[it, jt, ix, jx] * m.P[it, jt, ix, jx] * 1E5 == exp(
            -m.dH1 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)) + m.dS1 / m.R)
    else:
        return Constraint.Skip



def a65_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ke2c[it, jt, ix, jx] * m.P[it, jt, ix, jx] * 1E5 == exp(
            -m.dH2 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)) + m.dS2 / m.R)
    else:
        return Constraint.Skip



def a66_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ke3c[it, jt, ix, jx] * m.P[it, jt, ix, jx] * 1E5 == exp(
            -m.dH3 / (m.R * (m.Tsc[it, jt, ix, jx] + 273.15)) + m.dS3 / m.R)
    else:
        return Constraint.Skip


def a67_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ke1e[it, jt, ix, jx] * m.P[it, jt, ix, jx] * 1E5 == exp(
            -m.dH1 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)) + m.dS1 / m.R)
    else:
        return Constraint.Skip



def a68_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ke2e[it, jt, ix, jx] * m.P[it, jt, ix, jx] * 1E5 == exp(
            -m.dH2 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)) + m.dS2 / m.R)
    else:
        return Constraint.Skip



def a69_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Ke3e[it, jt, ix, jx] * m.P[it, jt, ix, jx] * 1E5 == exp(
            -m.dH3 / (m.R * (m.Tse[it, jt, ix, jx] + 273.15)) + m.dS3 / m.R)
    else:
        return Constraint.Skip



def a70_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.r1c[it, jt, ix, jx] == \
               m.k1c[it, jt, ix, jx] * ((m.P[it, jt, ix, jx] * m.yc[it, jt, ix, jx, 'h'] * 1E5) - (m.nc[it, jt, ix, jx, 'h'] * m.rhos / m.Ke1c[it, jt, ix, jx]))
    else:
        return Constraint.Skip



def a71_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.r2c[it, jt, ix, jx] == m.k2c[it, jt, ix, jx] * (
        (1 - 2 * (m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) -
         (m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) * m.nc[it, jt, ix, jx, 'h'] * m.rhos * m.P[
            it, jt, ix, jx] *
        m.yc[it, jt, ix, jx, 'c'] * 1E5 -
        (
            ((m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) + (
                m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) *
            m.nc[
                it, jt, ix, jx, 'c'] * m.rhos /
            m.Ke2c[it, jt, ix, jx]))
    else:
        return Constraint.Skip


def a72_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.r3c[it, jt, ix, jx] == m.k3c[it, jt, ix, jx] * (
        ((1 - 2 * (m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) -
          (m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) ** 2) * (
            (m.P[it, jt, ix, jx] * m.yc[it, jt, ix, jx, 'c'] * 1E5) ** m.m1) -
        ((m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) * (
            (m.nc[it, jt, ix, jx, 'n'] * m.rhos / m.nv) + (
                m.nc[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) /
         m.Ke3c[it, jt, ix, jx]))
    else:
        return Constraint.Skip



def a73_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.r1e[it, jt, ix, jx] == m.k1e[it, jt, ix, jx] * ((m.P[it, jt, ix, jx] * m.ye[it, jt, ix, jx, 'h'] * 1E5) - (m.ne[it, jt, ix, jx, 'h'] * m.rhos / m.Ke1e[it, jt, ix, jx]))
    else:
        return Constraint.Skip


def a74_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.r2e[it, jt, ix, jx] == m.k2e[it, jt, ix, jx] * (
        (1. - 2. * (m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) -
         (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) * m.ne[it, jt, ix, jx, 'h'] * m.rhos * (
            m.P[it, jt, ix, jx] * m.ye[it, jt, ix, jx, 'c'] * 1E5) -
        (((m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) +
          (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) * m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.Ke2e[
             it, jt, ix, jx])
        )
    else:
        return Constraint.Skip


def a75_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.r3e[it, jt, ix, jx] == \
               m.k3e[it, jt, ix, jx] * (
                   ((1. - 2. * (m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) -
                     (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) ** 2) * (
                   (m.P[it, jt, ix, jx] * m.ye[it, jt, ix, jx, 'c'] * 1E5) ** m.m1) -
                   ((m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) * (
                       (m.ne[it, jt, ix, jx, 'n'] * m.rhos / m.nv) + (m.ne[it, jt, ix, jx, 'c'] * m.rhos / m.nv)) /
                    m.Ke3e[it, jt, ix, jx]))
    else:
        return Constraint.Skip


def a76_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rgc[it, jt, ix, jx, 'c'] == (m.nv * m.r3c[it, jt, ix, jx] + m.r2c[it, jt, ix, jx]) / 1000.
    else:
        return Constraint.Skip



def a77_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rge[it, jt, ix, jx, 'c'] == (m.nv * m.r3e[it, jt, ix, jx] + m.r2e[it, jt, ix, jx]) / 1000.
    else:
        return Constraint.Skip



def a78_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rsc[it, jt, ix, jx, 'c'] == m.r2c[it, jt, ix, jx]
    else:
        return Constraint.Skip



def a79_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rse[it, jt, ix, jx, 'c'] == m.r2e[it, jt, ix, jx]
    else:
        return Constraint.Skip



def a80_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rgc[it, jt, ix, jx, 'h'] == m.r1c[it, jt, ix, jx] / 1000
    else:
        return Constraint.Skip



def a81_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rge[it, jt, ix, jx, 'h'] == m.r1e[it, jt, ix, jx] / 1000
    else:
        return Constraint.Skip



def a82_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rsc[it, jt, ix, jx, 'h'] == m.r1c[it, jt, ix, jx] - m.r2c[it, jt, ix, jx]
    else:
        return Constraint.Skip


def a83_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rse[it, jt, ix, jx, 'h'] == m.r1e[it, jt, ix, jx] - m.r2e[it, jt, ix, jx]
    else:
        return Constraint.Skip




def a84_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rgc[it, jt, ix, jx, 'n'] == 0
    else:
        return Constraint.Skip



def a85_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rge[it, jt, ix, jx, 'n'] == 0
    else:
        return Constraint.Skip


def a86_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rsc[it, jt, ix, jx, 'n'] == m.nv * m.r3c[it, jt, ix, jx]
    else:
        return Constraint.Skip



def a87_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.rse[it, jt, ix, jx, 'n'] == m.nv * m.r3e[it, jt, ix, jx]
    else:
        return Constraint.Skip



def a88_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.hsc[it, jt, ix, jx] == ((m.nc[it, jt, ix, jx, 'h'] + m.nc[it, jt, ix, jx, 'c']) * (m.cpgcsc['h'] * m.Tsc[it, jt, ix, jx] + m.dH1) +
                                         m.nc[it, jt, ix, jx, 'c'] * (m.cpgcsc['c'] * m.Tsc[it, jt, ix, jx] + m.dH2) +
                                         m.nc[it, jt, ix, jx, 'n'] * (
                                         m.cpgcsc['c'] * m.Tsc[it, jt, ix, jx] + m.dH3)) * 1E-3 + m.cps * m.Tsc[it, jt, ix, jx]
    else:
        return Constraint.Skip



def a89_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.hse[it, jt, ix, jx] == ((m.ne[it, jt, ix, jx, 'h'] + m.ne[it, jt, ix, jx, 'c']) * (
        m.cpgcse['h'] * m.Tse[it, jt, ix, jx] + m.dH1) +
                                         m.ne[it, jt, ix, jx, 'c'] * (
                                         m.cpgcse['c'] * m.Tse[it, jt, ix, jx] + m.dH2) +
                                         m.ne[it, jt, ix, jx, 'n'] * (
                                         m.cpgcse['c'] * m.Tse[it, jt, ix, jx] + m.dH3)) * 1E-3 + m.cps * m.Tse[
            it, jt, ix, jx]
    else:
        return Constraint.Skip



# d_rules diff?
# put derivative space here
# equation A.1 Gas phase component balance
# IC
def de_ngb_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dNgb_dt[it, jt, ix, jx, k] * m.hi_x[ix] == (-m.dcbin_dx[it, jt, ix, jx, k] + m.hi_x[ix] * (
            -m.Ax * m.delta[it, jt, ix, jx] * m.Kbc[it, jt, ix, jx, k] * (
            m.cb[it, jt, ix, jx, k] - m.cc[it, jt, ix, jx, k])
        ) + m.Kgbulk[it, jt, ix, jx, k]) * m.hi_t[it]
    else:
        return Constraint.Skip



# put derivative space here
# equation A.2 Gas phase energy balance
# IC
def de_hgb_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dHgb_dt[it, jt, ix, jx] * m.hi_x[ix] == (-m.debin_dx[it, jt, ix, jx] + \
                                                          -m.hi_x[ix] * m.Ax * m.delta[it, jt, ix, jx] * m.Hbc[
                                                              it, jt, ix, jx] * (
                                                          m.Tgb[it, jt, ix, jx] - m.Tgc[it, jt, ix, jx]) + m.Hgbulk[
                                                              it, jt, ix, jx]) * m.hi_t[it]
    else:
        return Constraint.Skip


# equation A.3 Gas phase component balance
def de_ngc_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dNgc_dt[it, jt, ix, jx, k] == \
               (m.delta[it, jt, ix, jx] * m.Kbc[it, jt, ix, jx, k] * (
               m.cb[it, jt, ix, jx, k] - m.cc[it, jt, ix, jx, k]) - \
                m.delta[it, jt, ix, jx] * m.Kce[it, jt, ix, jx, k] * (
                m.cc[it, jt, ix, jx, k] - m.ce[it, jt, ix, jx, k]) - \
                m.delta[it, jt, ix, jx] * m.fcw[it, jt, ix, jx] * (1. - m.ed[it, jt, ix, jx]) * m.rgc[
                    it, jt, ix, jx, k]) * \
               m.hi_t[it]
    else:
        return Constraint.Skip



# equation A.4 Gas phase energy balance
def de_hgc_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dHgc_dt[it, jt, ix, jx] == \
               (m.Ax * m.delta[it, jt, ix, jx] * \
                m.Hbc[it, jt, ix, jx] * (m.Tgb[it, jt, ix, jx] - m.Tgc[it, jt, ix, jx]) - \
                m.Ax * m.delta[it, jt, ix, jx] * \
                m.Hce[it, jt, ix, jx] * (m.Tgc[it, jt, ix, jx] - m.Tge[it, jt, ix, jx]) - \
                m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) * \
                m.rhos * m.ap * m.hp[it, jt, ix, jx] * (m.Tgc[it, jt, ix, jx] - m.Tsc[it, jt, ix, jx]) - \
                m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) * \
                sum(m.rgc[it, jt, ix, jx, k] * m.cpgcgc[k] for k in m.sp) * m.Tgc[it, jt, ix, jx]) * m.hi_t[it]
    else:
        return Constraint.Skip



# equation A.5 Solid phase adsorbed species balance
# IC
def de_nsc_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dNsc_dt[it, jt, ix, jx, k] * m.hi_x[ix] == \
               (-m.dccwin_dx[it, jt, ix, jx, k] * m.Ax - m.Ksbulk[it, jt, ix, jx, k] - \
                m.hi_x[ix] * m.Ax * m.delta[it, jt, ix, jx] * m.rhos * m.Kcebs[it, jt, ix, jx] * (
                m.nc[it, jt, ix, jx, k] - m.ne[it, jt, ix, jx, k]) + \
                m.hi_x[ix] * m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (1 - m.ed[it, jt, ix, jx]) *
                m.rsc[it, jt, ix, jx, k]) * m.hi_t[it]
    else:
        return Constraint.Skip



# put derivative space here
# equation A.6 Solid phase energy balance
def de_hsc_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dHsc_dt[it, jt, ix, jx] * m.hi_x[ix] \
               == (-m.decwin_dx[it, jt, ix, jx] * m.Ax - m.Hsbulk[it, jt, ix, jx] - \
                   m.hi_x[ix] * m.Ax * m.delta[it, jt, ix, jx] * m.rhos * m.Kcebs[it, jt, ix, jx] * (
                   m.hsc[it, jt, ix, jx] - m.hse[it, jt, ix, jx]) + \
                   m.hi_x[ix] * m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (
                   1 - m.ed[it, jt, ix, jx]) * sum(
                       (m.rgc[it, jt, ix, jx, k] * m.cpgcgc[k]) for k in m.sp) * (m.Tgc[it, jt, ix, jx]) + \
                   m.hi_x[ix] * m.Ax * m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] * (
                   1 - m.ed[it, jt, ix, jx]) * m.rhos * m.ap * m.hp[it, jt, ix, jx] * (
                       m.Tgc[it, jt, ix, jx] - m.Tsc[it, jt, ix, jx])) * m.hi_t[it]
    else:
        return Constraint.Skip



# equation A.7 Gas phase component balance
def de_nge_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dNge_dt[it, jt, ix, jx, k] \
               == (m.Ax * m.delta[it, jt, ix, jx] * m.Kce[it, jt, ix, jx, k] * (
        m.cc[it, jt, ix, jx, k] - m.ce[it, jt, ix, jx, k]) - \
                   m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (
                   1. - m.ed[it, jt, ix, jx]) * m.rge[
                       it, jt, ix, jx, k] - \
                   m.Kgbulk[it, jt, ix, jx, k] / m.hi_x[ix]) * m.hi_t[it]
    else:
        return Constraint.Skip



# equation A.8 Gas phase energy balance
def de_hge_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dHge_dt[it, jt, ix, jx] \
               == (m.Ax * m.delta[it, jt, ix, jx] * m.Hce[it, jt, ix, jx] * (
        m.Tgc[it, jt, ix, jx] - m.Tge[it, jt, ix, jx]) - \
                   m.Ax * (1 - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (
                       1. - m.ed[it, jt, ix, jx]) * m.rhos * m.ap * m.hp[it, jt, ix, jx] * (
                   m.Tge[it, jt, ix, jx] - m.Tse[it, jt, ix, jx]) - \
                   m.Hgbulk[it, jt, ix, jx] / m.hi_x[ix] - \
                   m.Ax * (1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (
                   1. - m.ed[it, jt, ix, jx]) * \
                   sum(m.rge[it, jt, ix, jx, k] * m.cpgcge[k] for k in m.sp) * m.Tge[it, jt, ix, jx]) * m.hi_t[it]
    else:
        return Constraint.Skip



# put derivative space here
# equation A.9 Solid phase adsorbed species balance
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
def de_hse_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dHse_dt[it, jt, ix, jx] * m.hi_x[ix] == \
               (m.deein_dx[it, jt, ix, jx] * m.Ax + m.Hsbulk[it, jt, ix, jx] + \
                m.hi_x[ix] * m.Ax * m.delta[it, jt, ix, jx] * m.rhos * m.Kcebs[it, jt, ix, jx] * (
                m.hsc[it, jt, ix, jx] - m.hse[it, jt, ix, jx]) + \
                m.hi_x[ix] * m.Ax * (
                1 - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (
                1 - m.ed[it, jt, ix, jx]) * \
                sum((m.rge[it, jt, ix, jx, k] * m.cpgcge[k]) for k in m.sp) * m.Tge[it, jt, ix, jx] + \
                m.hi_x[ix] * m.Ax * (
                1. - m.fcw[it, jt, ix, jx] * m.delta[it, jt, ix, jx] - m.delta[it, jt, ix, jx]) * (
                1. - m.ed[it, jt, ix, jx]) * m.rhos * m.ap * m.hp[it, jt, ix, jx] * (
                m.Tge[it, jt, ix, jx] - m.Tse[it, jt, ix, jx]) + \
                m.hi_x[ix] * m.pi * m.dx * m.ht[it, jt, ix, jx] * m.dThx[it, jt, ix, jx] * m.Nx * m.Cr) * m.hi_t[it]
    else:
        return Constraint.Skip

# shift the AV?


def de_ws_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dz_dx[it, jt, ix, jx] * m.Ax * m.hi_t[it] == m.dWs_dt[it, jt, ix, jx] * m.hi_x[ix]
    else:
        return Constraint.Skip



def i1_rule(m, it, jt, ix, jx, k):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Kgbulk[it, jt, ix, jx, k] == m.K_d * (sum(m.ce[it, jt, ix, jx, kx] for kx in m.sp) - sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp)) * m.yb[it, jt, ix, jx, k]
    else:
        return Constraint.Skip



def i2_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.Hgbulk[it, jt, ix, jx] == m.K_d * (sum(m.ce[it, jt, ix, jx, kx] for kx in m.sp) - sum(m.cb[it, jt, ix, jx, kx] for kx in m.sp)) * m.cpg_mol * \
                                           m.Tgb[it, jt, ix, jx]
    else:
        return Constraint.Skip



# oddly derivative looking term here and in the next one
# definetly derivatives e19 and e20 from bfb ss paper
def i3_rule(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.Ksbulk[it, kt, ix, kx, c] == \
               -m.Ax * sum(m.lydot[jx, kx] * m.Jc[it, kt, ix, jx] for jx in m.cp_x if 0 < jx <= m.ncp_x) \
               * m.ne[it, kt, ix, kx, c]
    else:
        return Constraint.Skip



# sum(m.ldot[j, k] * m.cbin[it, jt, ix, jx, c] for j in m.cp_x if j <= m.ncp_x)
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

# else:
#         return Constraint.Skip


def i5_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.db[it, jt, ix, jx] == m.dbu[it, jt, ix, jx]
    else:
        return Constraint.Skip


def i6_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.vb[it, jt, ix, jx] == \
               1.55 * ((m.vg[it, jt, ix, jx] - m.vmf[it, jt]) + 14.1 * (m.db[it, jt, ix, jx] + 0.005)) * (
               m.Dte ** 0.32) + m.vbr[it, jt, ix, jx]
    else:
        return Constraint.Skip


def i7_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return (1 - m.emf) * (
            (m.dp ** 0.1) * (m.gc ** 0.118) * 2.05 * (m.l[ix, jx] ** 0.043)) == \
               2.54 * (m.mug ** 0.066) * (1. - m.ed[it, jt, ix, jx])
    else:
        return Constraint.Skip


def i8_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.ve[it, jt, ix, jx] * (
        (m.dp ** 0.568) * (m.gc ** 0.663) * (0.08518 * (m.rhos - m.rhog[it, jt, ix, jx]) + 19.09) *
        ((m.l[ix, jx]) ** 0.244)) == \
               m.vmf[it, jt] * 188. * 1.02 * (m.mug ** 0.371)
    else:
        return Constraint.Skip


# exchanger pressure drop

def e1_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.HXIn_h[it, jt] == -0.2831 - 2.9863e-6 * (m.HXIn_P - 1.3) + 7.3855e-05 * (m.HXIn_T - 60)
    else:
        return Constraint.Skip

# bc_Phx
def e2_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.Phx_l[it, jt] == m.HXIn_P
    else:
        return Constraint.Skip



# Heat-Exchanger fluid energy balance
# pde
def e3_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return 0 == (m.HXIn_F / 3600) * m.dhxh_dx[it, jt, ix, jx] - \
                    m.hi_x[ix] * 1E-6 * m.pi * m.dx * m.ht[it, jt, ix, jx] * m.dThx[it, jt, ix, jx] * m.Nx * m.Cr
    else:
        return Constraint.Skip

# bc_hxh
def e4_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return 0 == m.HXIn_h[it, jt] - m.hxh_l[it, jt]
    else:
        return Constraint.Skip



def e5_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.hsint[it, jt] == ((m.nin['h'] + m.nin['c']) * (m.cpgcst['h'] * m.SolidIn_T + m.dH1) +
                                   m.nin['c'] * (m.cpgcst['c'] * m.SolidIn_T + m.dH2) +
                                   m.nin['n'] * (m.cpgcst['c'] * m.SolidIn_T + m.dH3)) * 1E-3 + m.cps * m.SolidIn_T
    else:
        return Constraint.Skip



# def e6_rule(m, i):
#     if i == 1:
#         return m.hsinb == ((m.nin['h'] + m.nin['c']) * (m.cpgcsb['h'] * m.SolidIn_T + m.dH1) +
#                        m.nin['c'] * (m.cpgcsb['c'] * m.SolidIn_T + m.dH2) +
#                        m.nin['n'] * (m.cpgcsb['c'] * m.SolidIn_T + m.dH3)) * 1E-3 + m.cps * m.SolidIn_T
#     else:
#         return Constraint.Skip


# self.e6 = Constraint(self.fe_x, rule=e6_rule)


# def e7_rule(m, it, jt):
#     # return m.P[1, 0] == 1.31514238316
#     if 0 < jt <= m.ncp_t:
#         return m.GasIn_P[it, jt] == m.P[it, jt, 1, 0] + 0.034
#     else:
#         return Constraint.Skip

# def e7_rule(m):
#     return m.GasIn_P == (m.P[1, 0] - m.P_l)*0.2 + m.P[1, 0]
#


# def _gasinf_rule(m):
#     return m.GasIn_F == 9950


# self.x_1 = Constraint(rule=_gasinf_rule)


def e8_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.Gb[it, jt, 1, 0] == m.GasIn_F[it]
    else:
        return Constraint.Skip

# bc for tgb
def e9_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.Tgb[it, jt, 1, 0] == m.GasIn_T[it]
    else:
        return Constraint.Skip

def e10_rule(m, it, jt, k):
    if 0 < jt <= m.ncp_t:
        return m.yb[it, jt, 1, 0, k] == m.GasIn_z[it, k]
    else:
        return Constraint.Skip



# def x_3_rule(m, it, jt):
#     if 0 < jt <= m.ncp_t:
#          return m.GasOut_P[it, jt] == m.P_l[it, jt]
#     else:
#         return Constraint.Skip

# # bc
def e12_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.Gb_l[it, jt] == m.GasOut_F[it, jt]
    else:
        return Constraint.Skip

# # bc
def e13_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.GasOut_T[it, jt] == m.Tgb_l[it, jt]
    else:
        return Constraint.Skip

# # bc
def e14_rule(m, it, jt, c):
    if 0 < jt <= m.ncp_t:
        return m.GasOut_z[it, jt, c] == m.yb_l[it, jt, c]
    else:
        return Constraint.Skip



# def e15_rule(m, it, jt):
#     if 0 < jt <= m.ncp_t:
#         return m.Sit[it, jt] == m.SolidIn_Fm[it, jt] / 3600
#     else:
#         return Constraint.Skip

# for v7


# def e16_rule(m, it, jt):
#     if 0 < jt <= m.ncp_t:
#         return m.SolidIn_P[it, jt] == m.GasOut_P[it, jt]
#     else:
#         return Constraint.Skip

#
#
# for v3rule

# # bc Jc_l Je_l
def e20_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.Sit[it, jt] - m.Sot[it, jt] == m.z_l[it, jt] * m.Ax
        # return 0 == m.z_l[it, jt] * m.Ax
    else:
        return Constraint.Skip


def e25_rule(m, it, jt, j):
    #     # if i == m.ND:
    if 0 < jt <= m.ncp_t:
        return m.ccwin_l[it, jt, j] * m.Ax + m.Sit[it, jt] * m.nin[j] == \
               m.cein_l[it, jt, j] * m.Ax + m.Sot[it, jt] * m.ne_l[it, jt, j]
    else:
        return Constraint.Skip

# # bc
# bc_eein
def e26_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.ecwin_l[it, jt] * m.Ax + m.Sit[it, jt] * m.hsint[it, jt] == \
               m.eein_l[it, jt] * m.Ax + m.Sot[it, jt] * m.hse_l[it, jt]
    else:
        return Constraint.Skip


# eqn_for_gasin_f
# def v1_rule(m, it, jt):
#     if 0 < jt <= m.ncp_t:
#         return (m.GasIn_F[it, jt] / 3600) == \
#                (m.CV_1 * (m.per_opening1[it] / 100) * ((m.flue_gas_P - m.GasIn_P[it, jt]) / m.rhog_in[it, jt]) ** 0.5)
#     else:
#         return Constraint.Skip


# def v2_rule(m, it, jt):
#     if 0 < jt <= m.ncp_t:
#         return m.rhog_in[it, jt] == \
#                m.GasIn_P[it, jt] * 100 * (
#                m.GasIn_z[it, 'c'] * 44.01 + m.GasIn_z[it, 'n'] * 28.01 + m.GasIn_z[it, 'h'] * 18.02) / (
#                    8.314 * (m.GasIn_T[it] + 273.16))
#     else:
#         return Constraint.Skip



# def v4_rule(m, it, jt):
#     if 0 < jt <= m.ncp_t:
#         return m.GasOut_F[it, jt] / 3600 == \
#                m.CV_2 * (m.per_opening2[it] / 100) * ((m.GasOut_P[it, jt] - m.Out2_P) / m.rhog_out[it, jt]) ** 0.5
#     else:
#         return Constraint.Skip



# def v5_rule(m, it, jt):
#     if 0 < jt <= m.ncp_t:
#         return m.rhog_out[it, jt] == m.GasOut_P[it, jt] * 100 * (
#             m.GasOut_z[it, jt, 'c'] * 44.01 + m.GasOut_z[it, jt, 'n'] * 28.01 + m.GasOut_z[it, jt, 'h'] * 18.02) / \
#                                      (8.314 * (m.GasOut_T[it, jt] + 273.16))
#     else:
#         return Constraint.Skip
# def v3_rule(m, it, jt):
#     if 0 < jt <= m.ncp_t:
#         return (m.SolidIn_Fm[it, jt] / 3600) == m.CV_3 * (m.per_opening3[it] / 100) * ((m.sorbent_P - m.SolidIn_P[it, jt]) / (2. * m.rhos)) ** 0.5
#     else:
#         return Constraint.Skip
# fdvar_t_cb
# if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
def fdvar_x_cbin_(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dcbin_dx[it, kt, ix, kx, c] == \
               sum(m.ldot_x[jx, kx] * m.cbin[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fdvar_x_cein_(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dcein_dx[it, kt, ix, kx, c] == \
               sum(m.ldot_x[jx, kx] * m.cein[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fdvar_x_ebin_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.debin_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.ebin[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fdvar_x_ecwin_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.decwin_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.ecwin[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fdvar_x_eein_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.deein_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.eein[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fdvar_x_hxh_(m, it, kt, ix, kx):  #
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dhxh_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.hxh[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fdvar_x_p_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dP_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.P[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fdvar_x_phx_(m, it, kt, ix, kx):  #
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dPhx_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.Phx[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fdvar_x_ccwin_(m, it, kt, ix, kx, c):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dccwin_dx[it, kt, ix, kx, c] == \
               sum(m.ldot_x[jx, kx] * m.ccwin[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


def fdvar_z_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dz_dx[it, kt, ix, kx] == sum(
            m.ldot_x[jx, kx] * m.z[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


def fcp_x_cbin(m, it, kt, ix, c):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.cbin[it, kt, ix + 1, 0, c] == \
               sum(m.l1_x[jx] * m.cbin[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


def fcp_x_cein(m, it, kt, ix, c):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.cein[it, kt, ix + 1, 0, c] == \
               sum(m.l1_x[jx] * m.cein[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fcp_x_ebin(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.ebin[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.ebin[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fcp_x_ecwin(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.ecwin[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.ecwin[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fcp_x_eein(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.eein[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.eein[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fcp_x_hxh(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.hxh[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.hxh[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# def fcp_x_p(m, it, kt, ix):
#     if 0 < kt <= m.ncp_t and ix < m.nfe_x:
#         return m.P[it, kt, ix + 1, 0] == \
#                sum(m.l1_x[jx] * m.P[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
#     else:
#         return Constraint.Skip

def fcp_x_phx(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.Phx[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.Phx[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fcp_x_ccwin(m, it, kt, ix, c):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.ccwin[it, kt, ix + 1, 0, c] == \
               sum(m.l1_x[jx] * m.ccwin[it, kt, ix, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fcp_z_(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.z[it, kt, ix + 1, 0] == sum(m.l1_x[jx] * m.z[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_1(m, it, kt, c):
    if 0 < kt <= m.ncp_t:
        return m.cbin_l[it, kt, c] == sum(
            m.l1_x[jx] * m.cbin[it, kt, m.nfe_x, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_2(m, it, kt, c):
    if 0 < kt <= m.ncp_t:
        return m.cein_l[it, kt, c] == sum(
            m.l1_x[jx] * m.cein[it, kt, m.nfe_x, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_3(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.ebin_l[it, kt] == sum(m.l1_x[jx] * m.ebin[it, kt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_4(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.ecwin_l[it, kt] == sum(m.l1_x[jx] * m.ecwin[it, kt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_5(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.eein_l[it, kt] == sum(m.l1_x[jx] * m.eein[it, kt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_6(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.hxh_l[it, kt] == sum(m.l1_x[jx] * m.hxh[it, kt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

# def fzl_8(m, it, kt):
#     if 0 < kt <= m.ncp_t:
#         return m.P_l[it, kt] == sum(m.l1_x[jx] * m.P[it, kt, m.nfe_x, jx] for jx in m.cp_x if 0 < jx <= m.ncp_x)
#     else:
#         return Constraint.Skip

def fzl_9(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.Phx_l[it, kt] == sum(m.l1_x[jx] * m.Phx[it, kt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fzl_10(m, it, kt, c):
    if 0 < kt <= m.ncp_t:
        return m.ccwin_l[it, kt, c] == sum(m.l1_x[jx] * m.ccwin[it, kt, m.nfe_x, jx, c] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


def fzl_z_(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.z_l[it, kt] == sum(m.l1_x[jx] * m.z[it, kt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fyl_hse(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.hse_l[it, jt] == sum(m.l1y[jx] * m.hse[it, jt, m.nfe_x, jx] for jx in m.cp_x if 0 < jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fyl_ne(m, it, jt, c):
    if 0 < jt <= m.ncp_t:
        return m.ne_l[it, jt, c] == sum(m.l1y[jx] * m.ne[it, jt, m.nfe_x, jx, c] for jx in m.cp_x if 0 < jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fyl_gb(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.Gb_l[it, jt] == sum(m.cbin_l[it, jt, c] for c in m.sp) * 3600
        # return m.Gb_l[it, jt] == sum(m.l1_x[jx] * m.Gb[it, jt, m.nfe_x, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

def fyl_tgb(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.ebin_l[it, jt] == (m.Gb_l[it, jt] / 3600) * m.cpg_mol * m.Tgb_l[it, jt]
    else:
        return Constraint.Skip

def fyl_yb(m, it, jt, c):
    if 0 < jt <= m.ncp_t:
        return m.cbin_l[it, jt, c] == m.yb_l[it, jt, c] * m.Gb_l[it, jt] / 3600
    else:
        return Constraint.Skip
        # return m.yb_l[c] == sum(m.l1y[j]*m.yb[m.nfe_x, j, c] for j in m.cp_x if 0 < j <= m.ncp_x)


def ic_jn(m, it, jt, c):
    if 0 < jt <= m.ncp_t:
        return m.ccwin[it, jt, 1, 0, c] == m.cein[it, jt, 1, 0, c]
    else:
        return Constraint.Skip


def ic_jh(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.ecwin[it, jt, 1, 0] == m.eein[it, jt, 1, 0]
    else:
        return Constraint.Skip


def ic_jejc(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.z[it, jt, 1, 0] == 0
    else:
        return Constraint.Skip
        # return m.Jc[1, 0] - m.Je[1, 0] <= m.dummy_alp


def cc_rule(m, it, jt):
    if 0 < jt <= m.ncp_t:
        return m.c_capture[it, jt] == \
               1 - (m.GasOut_F[it, jt] * m.GasOut_z[it, jt, 'c']) / (m.GasIn_F[it] * m.GasIn_z[it, 'c'])
    else:
        return Constraint.Skip


# ddx variables --------------------------------------------------------------------------------------------------------
# vg cont.eqn.
def fdvar_x_vg_(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dvg_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.vg[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


# vg ddx
def fcp_x_vg(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.vg[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.vg[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip





def fdvar_t_vg(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dvg_dt[it, kt, ix, kx] == \
               sum(m.ldot_t[jt, kt] * m.vg[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Constraint.Skip


def fcp_t_vg(m, it, ix, kx):
    if it < m.nfe_t and 0 < kx <= m.ncp_x:
        return m.vg[it + 1, 0, ix, kx] - \
               sum(m.l1_t[jt] * m.vg[it, jt, ix, kx] for jt in m.cp_t if jt <= m.ncp_t)
    else:
        return Expression.Skip


def vg_de_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dvg_dx[it, jt, ix, jx] == m.hi_x[ix] * m.vgbar[it, jt, ix, jx]
    else:
        return Constraint.Skip

def l_rule(m, it, jt, ix, jx):
    if 0 < jt <= m.ncp_t and 0 < jx <= m.ncp_x:
        return m.dvg_dt[it, jt, ix, jx] + 2 * m.vg[it, jt, ix, jx] * m.vgbar[it, jt, ix, jx] * m.hi_t[it] == 2 * m.mug * m.dvgbar_dx[it, jt, ix, jx] * m.hi_t[it] + m.dP_dx[it, jt, ix, jx] * 100000 * m.hi_t[it] + m.hi_x[ix] * m.hi_t[it] * (1 - m.e[it, jt, ix, jx]) * m.rhos * m.gc
    else:
        return Constraint.Skip


def fcp_x_vbar(m, it, kt, ix):
    if 0 < kt <= m.ncp_t and ix < m.nfe_x:
        return m.vgbar[it, kt, ix + 1, 0] == \
               sum(m.l1_x[jx] * m.vgbar[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip


def vgbar_rule(m, it, kt, ix, kx):
    if 0 < kt <= m.ncp_t and 0 < kx <= m.ncp_x:
        return m.dvgbar_dx[it, kt, ix, kx] == \
               sum(m.ldot_x[jx, kx] * m.vgbar[it, kt, ix, jx] for jx in m.cp_x if jx <= m.ncp_x)
    else:
        return Constraint.Skip

####
def vd_bc0(m, it, kt):
    if 0 < kt <= m.ncp_t:
        return m.vg[it, kt, 1, 0] == m.GasIn_F[it] * 8.314/(100*m.Ax*3600) * (m.GasIn_T[it] + 273.16)/m.GasIn_P[it]
