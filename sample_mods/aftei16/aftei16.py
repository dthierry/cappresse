#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from pyomo.core.base import Param, ConcreteModel, Var, Constraint, Set, Suffix, Expression
from pyomo.opt import SolverFactory
from pyomo.core.kernel.expr import exp, sqrt
from pyomo.core.kernel.numvalue import value
from pyomo.opt import ProblemFormat
from nmpc_mhe.aux.cpoinsc import collptsgen
from nmpc_mhe.aux.lagrange_f import lgr, lgry, lgrdot, lgrydot
from six import itervalues, iterkeys, iteritems
import re, os


class aftei16(ConcreteModel):
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

        self.fe_t = Set(initialize=[ii for ii in range(0, self.nfe_t)])

        # collocation points
        # collocation points for differential variables
        self.cp_t = Set(initialize=[ii for ii in range(0, self.ncp_t + 1)])
        self.nx = Set(initialize=[ii for ii in range(0, 4)])
        self.nu = Set(initialize=[ii for ii in range(0, 2)])
        # collocation points for algebraic variables
        self.cp_ta = Set(within=self.cp_t, initialize=[ii for ii in range(1, self.ncp_t + 1)])

        # create collocation param
        self.taucp_t = Param(self.cp_t, initialize=self.tau_i_t)

        self.ldot_t = Param(self.cp_t, self.cp_t, initialize=
        (lambda m, j, k: lgrdot(k, m.taucp_t[j], self.ncp_t, self._alp_gauB_t, self._bet_gauB_t)))  #: watch out for this!

        self.l1_t = Param(self.cp_t, initialize=
        (lambda m, j: lgr(j, 1, self.ncp_t, self._alp_gauB_t, self._bet_gauB_t)))

        # --------------------------------------------------------------------------------------------------------------
        self.x = Var(self.fe_t, self.cp_t, self.nx, initialize=lambda m, i, j, k: 1.0)

        #: Initial state-Param
        zero2 = dict.fromkeys(self.fe_t * self.cp_t * self.nx)
        zero_x = dict.fromkeys(self.nx)

        for key in zero2.keys():
            zero2[key] = 0.0

        self.x_ic = zero_x if steady else Param(self.nx, initialize=0.0, mutable=True)

        #:  Derivative-var
        self.dx_dt = zero2 if steady else Var(self.fe_t, self.cp_t, self.nx, initialize=0.0)
        self.u1 = Param(self.fe_t, mutable=True, default=1.1)  #: Dummy
        self.u2 = Param(self.fe_t, mutable=True, default=2.2)  #: Dummy
        self.u = Var(self.fe_t, self.nu, initialize=1.0)

        A = dict()
        A[0, 0] = -0.0151
        A[0, 1] = -60.5651
        A[0, 2] = 0.0
        A[0, 3] = -32.174

        A[1, 0] = -0.0001
        A[1, 1] = -1.3411
        A[1, 2] = 0.9929
        A[1, 3] = 0.0

        A[2, 0] = 0.00018
        A[2, 1] = 43.2541
        A[2, 2] = -0.869639
        A[2, 3] = 0.0

        A[3, 0] = 0.0
        A[3, 1] = 0.0
        A[3, 2] = 1.0
        A[3, 3] = 0.0

        B = dict()
        B[0, 0] = -2.516
        B[0, 1] = -13.136

        B[1, 0] = -0.1689
        B[1, 1] = -0.2514

        B[2, 0] = -17.251
        B[2, 1] = -1.5766

        B[3, 0] = 0
        B[3, 1] = 0

        self.A = Param(self.nx, self.nx, initialize=A)
        self.B = Param(self.nx, self.nu, initialize=B)

        def x_ode(m, i, j, nx):
            if j > 0:
                return self.dx_dt[i, j, nx] == sum(self.A[nx, nxx] * self.x[i, j, nxx] for nxx in self.nx) + \
                       sum(self.B[nx, nu] * self.u[i, nu] for nu in self.nu)

        self.de_x = Constraint(self.fe_t, self.cp_ta, self.nx, rule=x_ode)

        def x_coll(m, i, j, nx):
            if j > 0:
                return m.dx_dt[i, j, nx] == \
                       sum(m.ldot_t[j, k] * m.x[i, k, nx] for k in m.cp_t)
            else:
                return Constraint.Skip

        self.dvar_t_x = None if steady else Constraint(self.fe_t, self.cp_ta, self.nx, rule=x_coll)

        def x_cont(m, i, nx):
            if i < m.nfe_t and m.nfe_t > 1:
                return m.x[i + 1, 0, nx] - sum(m.l1_t[j] * m.x[i, j, nx] for j in m.cp_t)
            else:
                return Expression.Skip

        if self.nfe_t > 1:
            #: Noisy expressions
            self.noisy_x = None if steady else Expression(self.fe_t, self.nx, rule=x_cont)

            #: Continuation equations
            self.cp_x = None if steady else \
                Constraint(self.fe_t, self.nx,
                           rule=lambda m, i, nx: self.noisy_x[i, nx] == 0.0 if i < self.nfe_t else Constraint.Skip)

        def acx(m, nx):
            return m.x[0, 0, nx] == m.x_ic[nx]
        #: Initial condition-Constraints
        self.x_icc = None if steady else Constraint(self.nx, rule=acx)

        self.u1_e = Expression(self.fe_t, rule=lambda m, i: self.u[i, 0])
        self.u2_e = Expression(self.fe_t, rule=lambda m, i: self.u[i, 1])

        self.u1_c = Constraint(self.fe_t, rule=lambda m, i: self.u1[i] == self.u1_e[i])
        self.u2_c = Constraint(self.fe_t, rule=lambda m, i: self.u2[i] == self.u2_e[i])

        self.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        self.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        self.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        self.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)


    def write_nl(self, name):
        """Writes the nl file and the respective row & col"""
        name = str(self.__class__.__name__) + ".nl"
        self.write(filename=name,
                   format=ProblemFormat.nl,
                   io_options={"symbolic_solver_labels": True})

    def create_bounds(self):
        """Creates bounds for the variables"""
        return

    def init_steady_ref(self):
        """If the model is steady, we try to initialize it with an initial guess from ampl"""
        cur_dir = os.path.dirname(__file__)
        ampl_ig = os.path.join(cur_dir, "iv_ss.txt")
        file_tst = open(ampl_ig, "r")
        if self.nfe_t == 1 and self.ncp_t == 1:
            somedict = self.parse_ig_ampl(file_tst)
            for var in self.component_objects(Var, active=True):
                vx = getattr(self, str(var))
                for v, k in iteritems(var):
                    try:
                        vx[v] = somedict[str(var), v]
                    except KeyError:
                        continue
            solver = SolverFactory('ipopt')
            someresults = solver.solve(self, tee=True)

    def equalize_u(self, direction="u_to_r"):
        """set current controls to the values of their respective dummies"""
        if direction == "u_to_r":
            for i in iterkeys(self.u1):
                self.u[i, 1].set_value(value(self.u1[i]))
            for i in iterkeys(self.u2):
                self.u[i, 2].set_value(value(self.u2[i]))
        elif direction == "r_to_u":
            for i in iterkeys(self.u1):
                self.u1[i].value = value(self.u[i, 0])
            for i in iterkeys(self.u2):
                self.u2[i].value = value(self.u[i, 1])
