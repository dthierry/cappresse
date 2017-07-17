# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
from pyomo.core.base import ConcreteModel, Set, Constraint, Var,\
    Param, Objective, minimize, sqrt, exp, value
import re

def dist_col_Rodrigo_ss(init0, bnd_set):
    # collocation polynomial parameters

    # polynomial roots (Radau)

    m = ConcreteModel()
    m.i_flag = init0
    # set of finite elements and collocation points
    m.fe = Set(initialize=[1])
    m.cp = Set(initialize=[1])

    m.bnd_set = bnd_set

    if bnd_set:
        print("Bounds_set")

    Ntray = 42

    m.Ntray = Ntray

    m.tray = Set(initialize=[i for i in range(1, Ntray + 1)])

    def __init_feed(m, t):
        if t == 21:
            return 57.5294
        else:
            return 0


    m.feed = Param(m.tray, initialize=__init_feed)
    m.xf = Param(initialize=0.32) # feed mole fraction
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

    m.alpha = Param(m.tray, initialize=_alpha_init)

    ME0 = {}
    ME0[1] = 123790.826443232
    ME0[2] = 3898.34923206106
    ME0[3] = 3932.11766868415
    ME0[4] = 3950.13107445914
    ME0[5] = 3960.01212104318
    ME0[6] = 3965.37146944881
    ME0[7] = 3968.25340380767
    ME0[8] = 3969.78910997468
    ME0[9] = 3970.5965548502
    ME0[10] = 3971.0110096803
    ME0[11] = 3971.21368740283
    ME0[12] = 3971.30232788932
    ME0[13] = 3971.32958547037
    ME0[14] = 3971.32380573089
    ME0[15] = 3971.30024105555
    ME0[16] = 3971.26709591428
    ME0[17] = 3971.22878249852
    ME0[18] = 3971.187673073
    ME0[19] = 3971.14504284211
    ME0[20] = 3971.10157713182
    ME0[21] = 3971.05764415189
    ME0[22] = 3611.00216267141
    ME0[23] = 3766.84741932423
    ME0[24] = 3896.87907072814
    ME0[25] = 4004.98630195624
    ME0[26] = 4092.49383654928
    ME0[27] = 4161.86560059956
    ME0[28] = 4215.98509169956
    ME0[29] = 4257.69470716792
    ME0[30] = 4289.54901779038
    ME0[31] = 4313.71557755738
    ME0[32] = 4331.9642075775
    ME0[33] = 4345.70190802884
    ME0[34] = 4356.02621744716
    ME0[35] = 4363.78165047072
    ME0[36] = 4369.61159802674
    ME0[37] = 4374.00266939603
    ME0[38] = 4377.32093116489
    ME0[39] = 4379.84068162411
    ME0[40] = 4381.76685527968
    ME0[41] = 4383.25223100374
    ME0[42] = 4736.04924276762

    m.M_pred = Param(m.tray, initialize=ME0)

    XE0 = {}
    XE0[1] = 0.306547877605746
    XE0[2] = 0.398184778678485
    XE0[3] = 0.416675004386508
    XE0[4] = 0.42676332128531
    XE0[5] = 0.432244548463899
    XE0[6] = 0.435193762178033
    XE0[7] = 0.436764699693985
    XE0[8] = 0.437589297877498
    XE0[9] = 0.438010896454752
    XE0[10] = 0.43821522113022
    XE0[11] = 0.438302495819782
    XE0[12] = 0.438326730875504
    XE0[13] = 0.438317008813347
    XE0[14] = 0.438288981487008
    XE0[15] = 0.438251069561153
    XE0[16] = 0.438207802087721
    XE0[17] = 0.438161614415035
    XE0[18] = 0.438113815737636
    XE0[19] = 0.438065109638753
    XE0[20] = 0.438015874079915
    XE0[21] = 0.437966311972983
    XE0[22] = 0.724835538043496
    XE0[23] = 0.788208485334881
    XE0[24] = 0.838605564838572
    XE0[25] = 0.87793558673077
    XE0[26] = 0.908189470853012
    XE0[27] = 0.931224584141055
    XE0[28] = 0.948635083197147
    XE0[29] = 0.961724712952285
    XE0[30] = 0.971527857048483
    XE0[31] = 0.978848914860811
    XE0[32] = 0.984304939392599
    XE0[33] = 0.98836476845163
    XE0[34] = 0.991382214572503
    XE0[35] = 0.993622983870866
    XE0[36] = 0.995285909293636
    XE0[37] = 0.996519395295701
    XE0[38] = 0.997433995899531
    XE0[39] = 0.998111951760656
    XE0[40] = 0.998614376770054
    XE0[41] = 0.998986649363
    XE0[42] = 0.999262443919619

    m.x_pred = Param(m.tray, initialize=XE0)

    # hold in each tray

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

    m.M = Var(m.fe, m.cp, m.tray, initialize=__m_init)
    # m.M_0 = Var(m.fe, m.tray, initialize=1e07)

    # temperatures

    def __t_init(m, i, j, t):
        if m.i_flag:
            return ((370.781 - 335.753)/m.Ntray)*t + 370.781
        else:
            return 10.

    m.T = Var(m.fe, m.cp, m.tray, initialize=__t_init)


    # saturation pressures
    m.pm = Var(m.fe, m.cp, m.tray, initialize=1e4)
    m.pn = Var(m.fe, m.cp, m.tray, initialize=1e4)

    # define l-v flowrate

    def _v_init(m, i, j, t):
        if m.i_flag:
            return 44.
        else:
            return 0.

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

    m.L = Var(m.fe, m.cp, m.tray, initialize=_l_init)

    # mol frac l-v

    def __x_init(m, i, j, t):
        if m.i_flag:
            return (0.999/m.Ntray)*t
        else:
            return 1

    m.x = Var(m.fe, m.cp, m.tray, initialize=__x_init)

    #m.x_0 = Var(m.fe, m.tray)

    # av

    def __y_init(m, i, j, t):
        if m.i_flag:
            return ((0.99-0.005)/m.Ntray)*t + 0.005
        else:
            return 1

    m.y = Var(m.fe, m.cp, m.tray, initialize=__y_init)
    # enthalpy
    m.hl = Var(m.fe, m.cp, m.tray, initialize=10000.)

    def __hv_init(m, i, j, t):
        if m.i_flag:
            if t < m.Ntray:
                return 5e4
        else:
            return 0.0

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

    def _bound_set(m):
        if m.bnd_set:
            for key, value in m.M.iteritems():
                value.setlb(1.0)
                value.setub(1e7)
            for key, value in m.Vm.iteritems():
                value.setlb(-1.0)
                value.setub(1e4)
            for key, value in m.Mv.iteritems():
                value.setlb(0.155 + 1e-06)
                value.setub(1e4)
            for key, value in m.Mv1.iteritems():
                value.setlb(8.5 + 1e-06)
                value.setub(1e4)
            for key, value in m.Mvn.iteritems():
                value.setlb(0.17 + 1e-06)
                value.setub(1e4)
            for key, value in m.y.iteritems():
                value.setlb(0.0)
                value.setub(1.0)
            for key, value in m.x.iteritems():
                value.setlb(0.0)
                value.setub(1.0)
            for key, value in m.L.iteritems():
                value.setlb(0.0)
            for key, value in m.V.iteritems():
                value.setlb(0.0)




    _bound_set(m)

    m.Rec = Param(m.fe, initialize=7.72700925775773761472464684629813e-01)
    # m.Rec = Param(m.fe, initialize=0.05)
    m.Qr = Param(m.fe, initialize=1.78604740940007800236344337463379E+06)
    # m.Qr = Param(m.fe, initialize=1.5e+02)

    # mass balances
    def _MODEtr(m, i, j, k):
        if j > 0 and 1 < k < Ntray:
            return 0.0 == (m.V[i, j, k - 1] - m.V[i, j, k] + m.L[i, j, k + 1] - m.L[i, j, k] + m.feed[k])
        else:
            return Constraint.Skip

    m.MODEtr = Constraint(m.fe, m.cp, m.tray, rule=_MODEtr)

    # m.L[i, j, 1] = B
    def _MODEr(m, i, j):
        if j > 0:
            return 0.0 == (m.L[i, j, 2] - m.L[i, j, 1] - m.V[i, j, 1])
        else:
            return Constraint.Skip

    m.MODEr = Constraint(m.fe, m.cp, rule=_MODEr)

    def _MODEc(m, i, j):
        if j > 0:
            return 0.0 == (m.V[i, j, Ntray - 1] - m.L[i, j, Ntray] - m.D[i, j])
        else:
            return Constraint.Skip

    m.MODEc = Constraint(m.fe, m.cp, rule=_MODEc)


    def _XODEtr(m, i, j, t):
        if j > 0 and 1 < t < Ntray:
            return 0.0 == (m.V[i, j, t - 1] * (m.y[i, j, t - 1] - m.x[i, j, t]) + \
                                                     m.L[i, j, t + 1] * (m.x[i, j, t + 1] - m.x[i, j, t]) - \
                                                     m.V[i, j, t] * (m.y[i, j, t] - m.x[i, j, t]) + \
                                                     m.feed[t] * (m.xf - m.x[i, j, t]))
        else:
            return Constraint.Skip

    m.XODEtr = Constraint(m.fe, m.cp, m.tray, rule=_XODEtr)

    def _xoder(m, i, j):
        if j > 0:
            return 0.0 == (m.L[i, j, 2] * (m.x[i, j, 2] - m.x[i, j, 1]) - \
                                                     m.V[i, j, 1] * (m.y[i, j, 1] - m.x[i, j, 1]))
        else:
            return Constraint.Skip

    m.xoder = Constraint(m.fe, m.cp, rule=_xoder)

    def _xodec(m, i, j):
        if j > 0:
            return 0.0 == \
                   (m.V[i, j, Ntray - 1] * (m.y[i, j, Ntray - 1] - m.x[i, j, Ntray]))
        else:
            return Constraint.Skip

    m.xodec = Constraint(m.fe, m.cp, rule=_xodec)


    def _hrc(m, i, j):
        if j > 0:
            return m.D[i, j] - m.Rec[i]*m.L[i, j, Ntray] == 0
        else:
            return Constraint.Skip

    m.hrc = Constraint(m.fe, m.cp, rule=_hrc)

    # Energy balance
    def _gh(m, i, j, t):
        if j > 0 and 1 < t < Ntray:
            return 0.0 \
                   == (m.V[i, j, t-1] * (m.hv[i, j, t-1] - m.hl[i, j, t]) + m.L[i, j, t+1] * (m.hl[i, j, t+1] - m.hl[i, j, t]) - m.V[i, j, t] * (m.hv[i, j, t] - m.hl[i, j, t]) + m.feed[t] * (m.hf - m.hl[i, j, t]))
            # return m.M[i, j, t] * (m.xdot[i, j, t] * ((m.hlm0 - m.hln0) * m.T[i, j, t]**3 + (m.hlma - m.hlna) * m.T[i, j, t]**2 + (m.hlmb - m.hlnb) * m.T[i, j, t] + m.hlmc - m.hlnc) + m.Tdot[i, j, t]*(3*m.hln0*m.T[i, j, t]**2 + 2*m.hlna * m.T[i, j, t] + m.hlnb + m.x[i, j, t]*(3*(m.hlm0 - m.hln0) * m.T[i, j, t]**2 + 2 * (m.hlma - m.hlna) * m.T[i, j, t] + m.hlmb - m.hlnb))) \
            #        M[i, q, c] * (  xdot[i, q, c] * ((  hlm0 -   hln0) *   T[i, q, c] ^ 3 +  (  hlma -   hlna) *   T[i, q, c] ^ 2 +  (  hlmb -   hlnb) *   T[i, q, c] +   hlmc -   hlnc) +   Tdot[i, q, c]*(3*  hln0 * T[i, q, c] ^ 2 +2 * hlna *   T[i, q, c] +   hlnb +   x[i, q, c]*(3*(  hlm0 -   hln0) *   T[i, q, c] ^ 2 +    2 * (  hlma -   hlna) *   T[i, q, c] +   hlmb -   hlnb))) =
            #            V[i - 1, q, c]*(hv[i - 1, q, c] -   hl[i, q, c]) +   L[i + 1, q, c]*(hl[i + 1, q, c] -   hl[i, q, c]) -   V[i, q, c] * (  hv[i, q, c] -   hl[i, q, c]) +   feed[i] * (  hf -   hl[i, q, c]);


        else:
            return Constraint.Skip
        # M[i,q,c]    *(  xdot[i,q,c]*      ((hlm0 -     hln0) *   T[i,q,c]^3 +    (  hlma - hlna)     * T[i,q,c]^2 +    (  hlmb -   hlnb) *   T[i,q,c]   +   hlmc -   hlnc) +   Tdot[i,q,c]  *(3*  hln0*  T[i,q,c]^2    + 2*  hlna*    T[i,q,c] +     hlnb +   x[i,q,c]*  (3*(  hlm0 -   hln0)    *T[i,q,c]^2    + 2*   (hlma  -   hlna)    *T[i,q,c]+      hlmb -   hlnb)))
        # V[i-1,q,c]    *(  hv[i-1,q,c] -     hl[i,q,c]  ) +   L[i+1,q,c]*    (  hl[i+1,q,c] -     hl[i,q,c]  ) -   V[i,q,c]   * (  hv[i,q,c]   -   hl[i,q,c])   +   feed[i] * (hf   -   hl[i,q,c])

    m.gh = Constraint(m.fe, m.cp, m.tray, rule=_gh)

    def _ghb(m, i, j):
        if j > 0:
            return 0.0 == \
                   (m.L[i, j, 2] * (m.hl[i, j, 2] - m.hl[i, j, 1]) - m.V[i, j, 1] * (m.hv[i, j, 1] - m.hl[i, j, 1]) + m.Qr[i])
            #    M[1,q,c]*  (  xdot[1,q,c]  * ((  hlm0 -   hln0)   *T[1,q,c]^3 +    (hlma     - hlna)*  T[1,q,c]^2 +    (  hlmb -   hlnb)  *T[1,q,c]     + hlmc -   hlnc) +   Tdot[1,q,c]*    (3*    hln0 *   T[1,q,c]^2 +   2 *   hlna *   T[1,q,c]   +   hlnb +   x[1,q,c]    * (3 * (  hlm0 -   hln0) *   T[1,q,c]^2    + 2*(  hlma -   hlna)    *T[1,q,c]     + hlmb - hlnb))) =
            #        L[2,q,c]*    (  hl[2,q,c]   -   hl[1,q,c]  ) -   V[1,q,c]   * (  hv[1,q,c]   -   hl[1,q,c]  ) +   Qr[q] ;
        else:
            return Constraint.Skip

    m.ghb = Constraint(m.fe, m.cp, rule=_ghb)

    def _ghc(m, i, j):
        if j > 0:
            return 0.0 == \
                   (m.V[i, j, Ntray - 1] * (m.hv[i, j, Ntray - 1] - m.hl[i, j, Ntray]) - m.Qc[i, j])
                    #M[Ntray, q, c] * (xdot[Ntray, q, c]   * ((hlm0 -     hln0) *   T[Ntray, q, c] ^ 3 + (hlma - hlna) * T[Ntray, q, c] ^ 2 +     (hlmb - hlnb) *       T[Ntray, q, c] + hlmc - hlnc) +       Tdot[Ntray, q, c] * (3 * hln0 * T[Ntray, q, c] ^ 2 +    2 * hlna *    T[Ntray, q, c] +   hlnb +   x[Ntray, q, c] * (3 * (  hlm0 -   hln0) * T[Ntray, q, c] ^ 2 + 2 *    (hlma -   hlna) *   T[Ntray, q, c] +   hlmb -   hlnb))) =
                   # V[Ntray - 1, q, c] * (hv[Ntray - 1, q, c]     - hl[Ntray, q, c]) -  Qc[q, c];
        else:
            return Constraint.Skip
                    #M[Ntray, q, c] * (  xdot[Ntray, q, c] * ((  hlm0 -   hln0) *   T[Ntray, q, c] ^ 3 + (  hlma -   hlna) *  T[Ntray, q, c] ^ 2 +(hlmb   - hlnb) *     T[Ntray, q, c] +   hlmc - hlnc) +     Tdot[Ntray, q, c] * (3 * hln0 * T[Ntray, q, c] ^ 2 + 2 * hlna * T[Ntray, q, c] +         hlnb + x[Ntray, q, c] *   (3 * (  hlm0 -   hln0) * T[Ntray, q, c] ^ 2 + 2 * (hlma - hlna)      *   T[Ntray, q, c] + hlmb     - hlnb))) =
            # V[Ntray - 1, q, c] * (hv[Ntray - 1, q, c] - hl[Ntray, q, c]) - Qc[q, c];

    m.ghc = Constraint(m.fe, m.cp, rule=_ghc)

    def _hkl(m, i, j, t):
        if j > 0:
            return m.hl[i, j, t] == m.x[i, j, t]*(m.hlm0*m.T[i, j, t]**3 + m.hlma * m.T[i, j, t]**2 + m.hlmb * m.T[i, j, t] + m.hlmc) + (1 - m.x[i, j, t])*(m.hln0 * m.T[i, j, t]**3 + m.hlna*m.T[i, j, t]**2 + m.hlnb * m.T[i, j, t] + m.hlnc)
#                    hl[i, q, c] =    x[i, q, c]*(  hlm0 * T[i, q, c] ^ 3 +  hlma * T[i, q, c] ^ 2 + hlmb * T[i, q, c] + hlmc       ) + (1 - x[i, q, c]  )*(  hln0 * T[i, q, c] ^ 3 +    hlna*  T[i, q, c] ^ 2  + hlnb *   T[i, q, c] +   hlnc);
        else:
            return Constraint.Skip
            #    hl[i,q,c]    =   x[i,q,c]*  (  hlm0*  T[i,q,c]^3 +      hlma *   T[i,q,c]^2 +      hlmb*   T[i,q,c] +      hlmc) + (1 - x[i,q,c])*    (  hln0    *T[i,q,c] ^  3 +   hlna*  T[i,q,c]^2 + hlnb         *T[i,q,c]   +   hlnc) ;

    m.hkl = Constraint(m.fe, m.cp, m.tray, rule=_hkl)

    def _hkv(m, i, j, t):
        if j > 0 and t < Ntray:
            return m.hv[i, j, t] == m.y[i, j, t] * (m.hlm0 * m.T[i, j, t]**3 + m.hlma * m.T[i, j, t]**2 + m.hlmb * m.T[i, j, t] + m.hlmc + m.r * m.Tkm * sqrt(1 - (m.p[t]/m.Pkm) * (m.Tkm/m.T[i, j, t])**3)*(m.a - m.b * m.T[i, j, t]/m.Tkm + m.c1 * (m.T[i, j , t]/m.Tkm)**7 + m.gm * (m.d - m.l * m.T[i, j, t]/m.Tkm + m.f*(m.T[i, j, t]/m.Tkm)**7 ))) + (1 - m.y[i, j, t]) * (m.hln0 * m.T[i, j, t]**3 + m.hlna * m.T[i, j, t]**2 + m.hlnb * m.T[i, j, t] + m.hlnc + m.r * m.Tkn * sqrt(1 - (m.p[t]/m.Pkn)*(m.Tkn/m.T[i, j, t])**3)*(m.a - m.b * m.T[i, j, t]/m.Tkn + m.c1 * (m.T[i, j, t]/m.Tkn)**7 + m.gn*(m.d - m.l * m.T[i, j, t]/m.Tkn + m.f* (m.T[i, j, t]/m.Tkn)**7)))
        else:
            return Constraint.Skip
    #            hv[i,q,c] =      y[i,q,c]   *     (hlm0 *   T[i,q,c]^3    +   hlma*    T[i,q,c]^2    +   hlmb *   T[i,q,c] +     hlmc +   r*    Tkm*    sqrt(1 - (  p[i]/  Pkm)*  ( Tkm/  T[i,q,c]) ^ 3  )*(a   -   b*    T[i,q,c]/    Tkm +   c1*  (  T[i,q,c]   /  Tkm) ^7 +   gm*  (  d -   l *   T[i,q,c]  /Tkm +     f*(  T[i,q,c] /   Tkm)^7  ))) + (1 - y[i,q,c]    )*  (  hln0 *   T[i,q,c]  ^3 +    hlna*    T[i,q,c]^   2 +   hlnb*    T[i,q,c]  +    hlnc +   r *   Tkn*  sqrt(1 - (  p[i]/Pkn  )*(  Tkn/  T[i,q,c]  )^3 )*(  a -   b *   T[i,q,c]  /  Tkn   + c1 * (  T[i,q,c]  /  Tkn)^7 +    gn*(  d -   l*    T[i,q,c]/    Tkn +    f*(  T[i,q,c]  / Tkn) ^7 ))) ;

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

    def _gy0(m, i, j):
        if j > 0:
            return m.y[i, j, 1] == m.x[i, j, 1] * m.pm[i, j, 1]/m.p[1]
        else:
            return Constraint.Skip

    m.gy0 = Constraint(m.fe, m.cp, rule=_gy0)

    def _gy(m, i, j, t):
        if j > 0 and 1 < t < Ntray:
            return m.y[i, j, t] == m.alpha[t] * m.x[i, j, t] * m.pm[i, j, t] / m.p[t] + (1 - m.alpha[t]) * m.y[i, j, t - 1]
                #y[i, q, c] =    alpha[i] *   x[i, q, c] *   pm[i, q, c] /   p[i] + (1 -  alpha[i]) *   y[i - 1, q, c];
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
            #        L[i,q,c]*      Vm[i,q,c] =    0.166 * (  Mv[i,q,c]   - 0.155)^1.5 ;
        else:
            return Constraint.Skip

    m.hyd = Constraint(m.fe, m.cp, m.tray, rule=_hyd)

    def _hyd1(m, i, j):
        if j > 0:
            return m.L[i, j, 1] * m.Vm[i, j, 1] == 0.166 * (m.Mv1[i, j] - 8.5) ** 1.5
        else:
            return Constraint.Skip
                 #  L[i,q,c]*Vm[i,q,c] = 0.166*(Mv[i,q,c] - 0.155)^1.5 ;

    m.hyd1 = Constraint(m.fe, m.cp, rule=_hyd1)

    def _hydN(m, i, j):
        if j > 0:
            return m.L[i, j, Ntray] * m.Vm[i, j, Ntray] == 0.166 * (m.Mvn[i, j] - 0.17) ** 1.5
        else:
            return Constraint.Skip
            #  L[i,q,c]*Vm[i,q,c] = 0.166*(Mv[i,q,c] - 0.155)^1.5 ;

    m.hydN = Constraint(m.fe, m.cp, rule=_hydN)

    def _dvm(m, i, j, t):
        if j > 0:
            return m.Vm[i, j, t] == m.x[i, j, t] * ((1/2288) * 0.2685**(1 + (1 - m.T[i, j, t]/512.4)**0.2453)) + (1 - m.x[i, j, t]) * ((1/1235) * 0.27136**(1 + (1 - m.T[i, j, t]/536.4)**0.24))
        else:
            return Constraint.Skip
             #   Vm[i,q,c] =          x[i,q,c]   * ( 1/2288 *  0.2685^ (1 + (1 -   T[i,q,c]  /512.4)^0.2453)) +   (1 -   x[i,q,c])   * (1/1235 * 0.27136^ (1 + (1 - T[i,q,c]    /536.4)^ 0.24)) ;

    m.dvm = Constraint(m.fe, m.cp, m.tray, rule=_dvm)

    return m


from pyomo.environ import *
from pyomo.opt import SolverFactory
from pyomo.core.base import Suffix

somemodel = dist_col_Rodrigo_ss(True, True)


# solver = SolverFactory("asl:ipopt")
# solver.options["halt_on_ampl_error"] = "yes"

solver = SolverFactory("ipopt")
solver.options["linear_solver"] = "ma57"
solver.options["option_file_name"] = "ipopt.opt"

def parse_ig_ampl(file_i):
    lines = file_i.readlines()
    dict = {}
    for line in lines:
        kk = re.split('(?:let)|[:=\s\[\]]', line)
        try:
            var = kk[2]
            key = kk[3]
            key = re.split(',', key)
            actual_key = []
            for k in key:
                actual_key.append(int(k))
            actual_key.append(actual_key.pop(0))
            actual_key = tuple(actual_key)

            value = kk[8]
            value = float(value)
            dict[var,actual_key] = value
        except IndexError:
            continue
    file_i.close()
    return dict



file_tst = open("iv_ss.txt", "r")
somedict = parse_ig_ampl(file_tst)
somemodel.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
somemodel.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)

somemodel.display(filename="somefile0.txt")

for var in somemodel.component_objects(Var, active=True):
    vx = getattr(somemodel, str(var))
    for v, k in var.iteritems():
        # print(str(var), v)
        try:
            vx[v] = somedict[str(var), v]
        except KeyError:
            continue


someresults = solver.solve(somemodel, tee=True)
# somemodel.pprint(filename="pprint.txt")
# somemodel.display(filename="somefile1.txt")
# someresults = solver.solve(somemodel, tee=True)
file = open('d.py', 'w')
file.write('dic_ss = {}\n')
dic_ss = {}
for var in somemodel.component_objects(Var, active=True):
    act_var = getattr(somemodel, str(var))
    for key in act_var:
        try:
            dic_ss[str(var), key[-1]] = value(act_var[key])
            file.write('dic_ss[\'' + str(var) + '\', ' + str(key[-1]) + '] = ' + str(value(act_var[key])))
            file.write('\n')
        except ValueError:
            dic_ss[str(var), key[-1]] = 1


file.close()
somemodel.display(filename="somefile_.txt")
# try:
    #     print("Result {:_<10}\t{:06.2f}".format(str(var), value(var)))
    # except ValueError:
    #     pass

# someresults = solver.solve(somemodel, tee=True)


