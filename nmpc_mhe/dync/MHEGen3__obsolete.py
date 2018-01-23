#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix, ConstraintList
from pyomo.core.base.sets import SimpleSet
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from nmpc_mhe.dync.DynGen import DynGen
from nmpc_mhe.dync.NMPCGen import NmpcGen
import numpy as np
from itertools import product
import sys, os, time

__author__ = "David M Thierry @dthierry"
"""Not yet. Our people, they don't understand."""


class MheGen(NmpcGen):
    def __init__(self, **kwargs):
        NmpcGen.__init__(self, **kwargs)
        self.int_file_mhe_suf = int(time.time())-1

        # Need a list of relevant measurements y

        self.y = kwargs.pop('y', [])
        self.y_vars = kwargs.pop('y_vars', {})

        # Need a list or relevant noisy-states z

        self.x_noisy = kwargs.pop('x_noisy', [])
        self.x_vars = kwargs.pop('x_vars', {})
        self.deact_ics = kwargs.pop('del_ics', True)
        self.diag_Q_R = kwargs.pop('diag_QR', True)  #: By default use diagonal matrices for Q and R matrices
        self.u = kwargs.pop('u', [])
        self.IgnoreProcessNoise = kwargs.pop('IgnoreProcessNoise', False)


        print("-" * 120)
        print("I[[create_lsmhe]] lsmhe (full) model created.")
        print("-" * 120)
        nstates = sum(len(self.x_vars[x]) for x in self.x_noisy)

        self.journalizer("I", self._c_it, "MHE with \t", str(nstates) + "states")
        self.journalizer("I", self._c_it, "MHE with \t", str(nstates*self.nfe_t*self.ncp_t) + "noise vars")
        self.lsmhe = self.d_mod(self.nfe_t, self.ncp_t, _t=self._t)
        self.lsmhe.name = "LSMHE (Least-Squares MHE)"
        self.lsmhe.create_bounds()
        #: create x_pi constraint

        #: Create list of noisy-states vars
        self.xkN_l = []
        self.xkN_nexcl = []
        self.xkN_key = {}
        k = 0
        for x in self.x_noisy:
            n_s = getattr(self.lsmhe, x)  #: Noisy-state
            for jth in self.x_vars[x]:  #: the jth variable
                self.xkN_l.append(n_s[(1, 0) + jth])
                self.xkN_nexcl.append(1)  #: non-exclusion list for active bounds
                self.xkN_key[(x, jth)] = k
                k += 1

        self.lsmhe.xkNk_mhe = Set(initialize=[i for i in range(0, len(self.xkN_l))])  #: Create set of noisy_states
        self.lsmhe.x_0_mhe = Param(self.lsmhe.xkNk_mhe, initialize=0.0, mutable=True)  #: Prior-state
        self.lsmhe.wk_mhe = Param(self.lsmhe.fe_t, self.lsmhe.cp_ta, self.lsmhe.xkNk_mhe, initialize=0.0) \
            if self.IgnoreProcessNoise else Expression(self.lsmhe.fe_t, self.lsmhe.cp_ta, self.lsmhe.xkNk_mhe)  #: Model disturbance
        self.lsmhe.PikN_mhe = Param(self.lsmhe.xkNk_mhe, self.lsmhe.xkNk_mhe,
                                initialize=lambda m, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Prior-Covariance
        self.lsmhe.Q_mhe = Param(range(1, self.nfe_t), self.lsmhe.xkNk_mhe, initialize=1, mutable=True) if self.diag_Q_R\
            else Param(range(1, self.nfe_t), self.lsmhe.xkNk_mhe, self.lsmhe.xkNk_mhe,
                             initialize=lambda m, t, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Disturbance-weight
        j = 0
        for i in self.x_noisy:
            de_exp = getattr(self.lsmhe, "de_" + i)
            for k in self.x_vars[i]:
                for tfe in range(1, self.nfe_t+1):
                    for tcp in range(1, self.ncp_t + 1):
                        self.lsmhe.wk_mhe[tfe, tcp, j].set_value(de_exp[(tfe, tcp) + k]._body)
                        de_exp[(tfe, tcp) + k].deactivate()
                j += 1



        #: Create list of measurements vars
        self.yk_l = {}
        self.yk_key = {}
        k = 0
        self.yk_l[1] = []
        for y in self.y:
            m_v = getattr(self.lsmhe, y)  #: Measured "state"
            for jth in self.y_vars[y]:  #: the jth variable
                self.yk_l[1].append(m_v[(1, self.ncp_t) + jth])
                self.yk_key[(y, jth)] = k  #: The key needs to be created only once, that is why the loop was split
                k += 1

        for t in range(2, self.nfe_t + 1):
            self.yk_l[t] = []
            for y in self.y:
                m_v = getattr(self.lsmhe, y)  #: Measured "state"
                for jth in self.y_vars[y]:  #: the jth variable
                    self.yk_l[t].append(m_v[(t, self.ncp_t) + jth])

        self.lsmhe.ykk_mhe = Set(initialize=[i for i in range(0, len(self.yk_l[1]))])  #: Create set of measured_vars
        self.lsmhe.nuk_mhe = Var(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=0.0)   #: Measurement noise
        self.lsmhe.yk0_mhe = Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=1.0, mutable=True)
        self.lsmhe.hyk_c_mhe = Constraint(self.lsmhe.fe_t, self.lsmhe.ykk_mhe,
                                          rule=
                                          lambda mod, t, i:mod.yk0_mhe[t, i] - self.yk_l[t][i] - mod.nuk_mhe[t, i] == 0.0)
        self.lsmhe.hyk_c_mhe.deactivate()
        self.lsmhe.R_mhe = Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=1.0, mutable=True) if self.diag_Q_R else \
            Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, self.lsmhe.ykk_mhe,
                             initialize=lambda mod, t, i, ii: 1.0 if i == ii else 0.0, mutable=True)
        f = open("file_cv.txt", "w")
        f.close()

        #: Constraints for the input noise
        for u in self.u:
            # cv = getattr(self.lsmhe, u)  #: Get the param
            # c_val = [value(cv[i]) for i in cv.keys()]  #: Current value
            # self.lsmhe.del_component(cv)  #: Delete the param
            # self.lsmhe.add_component(u + "_mhe", Var(self.lsmhe.fe_t, initialize=lambda m, i: c_val[i-1]))
            self.lsmhe.add_component("w_" + u + "_mhe", Var(self.lsmhe.fe_t, initialize=0.0))  #: Noise for input
            self.lsmhe.add_component("w_" + u + "c_mhe", Constraint(self.lsmhe.fe_t))
            self.lsmhe.equalize_u(direction="r_to_u")
            # cc = getattr(self.lsmhe, u + "_c")  #: Get the constraint for input
            con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
            var_w = getattr(self.lsmhe, "w_" + u + "_mhe")  #: Get the constraint-noisy
            ce = getattr(self.lsmhe, u + "_e")  #: Get the expression
            cp = getattr(self.lsmhe, u)  #: Get the param

            con_w.rule = lambda m, i: cp[i] == ce[i] + var_w[i]
            con_w.reconstruct()
            con_w.deactivate()

            # con_w.rule = lambda m, i: cp[i] == cv[i] + var_w[i]
            # con_w.reconstruct()
            # with open("file_cv.txt", "a") as f:
            #     cc.pprint(ostream=f)
            #     con_w.pprint(ostream=f)
                # f.close()

        self.lsmhe.U_mhe = Param(range(1, self.nfe_t + 1), self.u, initialize=1, mutable=True)

        #: Deactivate icc constraints
        if self.deact_ics:
            pass
            # for i in self.states:
            #     self.lsmhe.del_component(i + "_icc")
        #: Maybe only for a subset of the states
        else:
            for i in self.states:
                if i in self.x_noisy:
                    ic_con = getattr(self.lsmhe, i + "_icc")
                    for k in self.x_vars[i]:
                        ic_con[k].deactivate()

        #: Put the noise in the continuation equations (finite-element)
        j = 0
        self.lsmhe.noisy_cont = ConstraintList()
        for i in self.x_noisy:
            # cp_con = getattr(self.lsmhe, "cp_" + i)
            cp_exp = getattr(self.lsmhe, "noisy_" + i)
            # self.lsmhe.del_component(cp_con)
            for k in self.x_vars[i]:  #: This should keep the same order
                for t in range(1, self.nfe_t):
                    self.lsmhe.noisy_cont.add(cp_exp[t, k] == 0.0)
                    # self.lsmhe.noisy_cont.add(cp_exp[t, k] == 0.0)
                j += 1
            # cp_con.reconstruct()
        j = 0
        self.lsmhe.noisy_cont.deactivate()

        #: Expressions for the objective function (least-squares)
        self.lsmhe.Q_e_mhe = 0.0 if self.IgnoreProcessNoise else Expression(
            expr=0.5 * sum(
                sum(
                    sum(self.lsmhe.Q_mhe[1, k] * self.lsmhe.wk_mhe[i, j, k]**2 for k in self.lsmhe.xkNk_mhe) for j in range(1, self.ncp_t +1))
                for i in range(1, self.nfe_t+1))) if self.diag_Q_R else Expression(
            expr=sum(sum(self.lsmhe.wk_mhe[i, j] *
                         sum(self.lsmhe.Q_mhe[i, j, k] * self.lsmhe.wk_mhe[i, 1, k] for k in self.lsmhe.xkNk_mhe)
                         for j in self.lsmhe.xkNk_mhe) for i in range(1, self.nfe_t)))

        self.lsmhe.R_e_mhe = Expression(
            expr=0.5 * sum(
                sum(
                    self.lsmhe.R_mhe[i, k] * self.lsmhe.nuk_mhe[i, k]**2 for k in self.lsmhe.ykk_mhe)
                for i in self.lsmhe.fe_t)) if self.diag_Q_R else Expression(
            expr=sum(sum(self.lsmhe.nuk_mhe[i, j] *
                         sum(self.lsmhe.R_mhe[i, j, k] * self.lsmhe.nuk_mhe[i, k] for k in self.lsmhe.ykk_mhe)
                         for j in self.lsmhe.ykk_mhe) for i in self.lsmhe.fe_t))
        expr_u_obf = 0
        for i in self.lsmhe.fe_t:
            for u in self.u:
                var_w = getattr(self.lsmhe, "w_" + u + "_mhe")  #: Get the constraint-noisy
                expr_u_obf += self.lsmhe.U_mhe[i, u] * var_w[i] ** 2

        self.lsmhe.U_e_mhe = Expression(expr=0.5 * expr_u_obf)  # how about this
        # with open("file_cv.txt", "a") as f:
        #     self.lsmhe.U_e_mhe.pprint(ostream=f)
        #     f.close()

        self.lsmhe.Arrival_e_mhe = Expression(
            expr=0.5 * sum((self.xkN_l[j] - self.lsmhe.x_0_mhe[j]) *
                     sum(self.lsmhe.PikN_mhe[j, k] * (self.xkN_l[k] - self.lsmhe.x_0_mhe[k]) for k in self.lsmhe.xkNk_mhe)
                     for j in self.lsmhe.xkNk_mhe))

        self.lsmhe.Arrival_dummy_e_mhe = Expression(
            expr=100000.0 * sum((self.xkN_l[j] - self.lsmhe.x_0_mhe[j]) ** 2 for j in self.lsmhe.xkNk_mhe))

        self.lsmhe.obfun_dum_mhe_deb = Objective(sense=minimize,
                                             expr=self.lsmhe.Q_e_mhe)
        self.lsmhe.obfun_dum_mhe = Objective(sense=minimize,
                                             expr=self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.U_e_mhe) # no arrival
        self.lsmhe.obfun_dum_mhe.deactivate()

        self.lsmhe.obfun_mhe_first = Objective(sense=minimize,
                                         expr=self.lsmhe.Arrival_dummy_e_mhe + self.lsmhe.Q_e_mhe)
        self.lsmhe.obfun_mhe_first.deactivate()


        self.lsmhe.obfun_mhe = Objective(sense=minimize,
                                         expr=self.lsmhe.Arrival_dummy_e_mhe + self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.U_e_mhe)
        self.lsmhe.obfun_mhe.deactivate()

        # with open("file_cv.txt", "a") as f:
        #     self.lsmhe.obfun_mhe.pprint(ostream=f)
        #     f.close()

        self._PI = {}  #: Container of the KKT matrix
        self.xreal_W = {}
        self.curr_m_noise = {}   #: Current measurement noise
        self.curr_y_offset = {}  #: Current offset of measurement
        for y in self.y:
            for j in self.y_vars[y]:
                self.curr_m_noise[(y, j)] = 0.0
                self.curr_y_offset[(y, j)] = 0.0

        self.s_estimate = {}
        self.s_real = {}
        for x in self.x_noisy:
            self.s_estimate[x] = []
            self.s_real[x] = []

        self.y_estimate = {}
        self.y_real = {}
        self.y_noise_jrnl = {}
        self.yk0_jrnl = {}
        for y in self.y:
            self.y_estimate[y] = []
            self.y_real[y] = []
            self.y_noise_jrnl[y] = []
            self.yk0_jrnl[y] = []

    def initialize_xreal(self, ref):
        """Wanted to keep the states in a horizon-like window, this should be done in the main dyngen class"""
        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.name = "Dummy [xreal]"
        self.load_d_d(ref, dum, 1)
        for fe in range(1, self._window_keep):
            for i in self.states:
                pn = i + "_ic"
                p = getattr(dum, pn)
                vs = getattr(dum, i)
                for ks in p.iterkeys():
                    p[ks].value = value(vs[(1, self.ncp_t) + (ks,)])
            #: Solve
            self.solve_d(dum, o_tee=False)
            for i in self.states:
                self.xreal_W[(i, fe)] = []
                xs = getattr(dum, i)
                for k in xs.keys():
                    if k[1] == self.ncp_t:
                        print(i)
                        self.xreal_W[(i, fe)].append(value(xs[k]))

    def init_lsmhe_prep(self, ref):
        """Initializes the lsmhe in preparation phase
        Args:
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model"""
        self.journalizer("I", self._c_it, "initialize_lsmhe", "Preparation phase MHE")
        dum = self.d_mod(1, self.ncp_t, _t=self.hi_t)
        dum.name = "Dummy I"
        #: Load current solution
        self.load_d_d(ref, dum, 1)
        #: Patching of finite elements
        for finite_elem in range(1, self.nfe_t + 1):
            #: Cycle ICS
            for i in self.states:
                pn = i + "_ic"
                p = getattr(dum, pn)
                vs = getattr(dum, i)
                for ks in p.iterkeys():
                    p[ks].value = value(vs[(1, self.ncp_t) + (ks,)])
            if finite_elem == 1:
                for i in self.states:
                    pn = i + "_ic"
                    p = getattr(self.lsmhe, pn)  #: Target
                    vs = getattr(dum, i)  #: Source
                    for ks in p.iterkeys():
                        p[ks].value = value(vs[(1, self.ncp_t) + (ks,)])
            self.patch_meas_mhe(finite_elem, src=self.d1)
            #: Solve
            self.solve_d(dum, o_tee=False)
            #: Patch
            self.load_d_d(dum, self.lsmhe, finite_elem)
            self.load_input_mhe("mod", src=dum, fe=finite_elem)
        self.lsmhe.name = "Preparation MHE"   #: Pretty much simulation
        tst = self.solve_d(self.lsmhe, skip_update=False,
                  max_cpu_time=12000,
                  halt_on_ampl_error=False,
                  jacobian_regularization_value=1e-02,
                  jacobian_regularization_exponent=2.,
                  tol=1e-03,
                  iter_max=100000,
                  # mu_target=1e-03,
                  output_file="file_prepmhe.txt")  #: Pre-loaded mhe solve
        # with open("cons_0.txt", "w") as f:
        #     for con in self.lsmhe.component_objects(Constraint, active=True):
        #         con.pprint(ostream=f)
        #     for obj in self.lsmhe.component_objects(Objective, active=True):
        #         obj.pprint(ostream=f)
        #     f.close()
        if tst != 0:
            self.lsmhe.write_nl(name="failed_mhe.nl")
            sys.exit()
        self.lsmhe.name = "LSMHE (Least-Squares MHE)"

        self.lsmhe.obfun_dum_mhe_deb.deactivate()

        self.lsmhe.obfun_mhe_first.activate()
        self.deact_icc_mhe()
        self.lsmhe.hyk_c_mhe.activate()


        for u in self.u:
            cc = getattr(self.lsmhe, u + "_c")  #: Get the constraint for input
            con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
            cc.deactivate()
            con_w.activate()

        # if self.deact_ics:
        #     for i in self.states:
        #         self.lsmhe.del_component(i + "_icc")

        self.journalizer("I", self._c_it, "initialize_lsmhe", "Attempting to initialize lsmhe Done")

    def patch_meas_mhe(self, t, **kwargs):
        """Mechanism to assign a value of y0 to the current mhe from the dynamic model
        Args:
            t (int): int The current collocation point
        Returns:
            meas_dict (dict): A dictionary containing the measurements list by meas_var
        """
        src = kwargs.pop("src", None)
        skip_update = kwargs.pop("skip_update", False)
        noisy = kwargs.pop("noisy", True)

        meas_dic = dict.fromkeys(self.y)
        l = []
        for i in self.y:
            lm = []
            var = getattr(src, i)
            for j in self.y_vars[i]:
                lm.append(value(var[(1, self.ncp_t,) + j]))
                l.append(value(var[(1, self.ncp_t,) + j]))
            meas_dic[i] = lm

        if not skip_update:  #: Update the mhe model
            self.journalizer("I", self._c_it, "patch_meas_mhe", "Measurement patched to " + str(t))
            y0dest = getattr(self.lsmhe, "yk0_mhe")
            # print("there is an update", file=sys.stderr)
            for i in self.y:
                for j in self.y_vars[i]:
                    k = self.yk_key[(i, j)]
                    #: Adding noise to the mhe measurement
                    y0dest[t, k].value = l[k] + self.curr_m_noise[(i, j)] if noisy else l[k]
        return meas_dic

    def adjust_nu0_mhe(self):
        """Adjust the initial guess for the nu variable"""
        for t in self.lsmhe.fe_t:
            k = 0
            for i in self.y:
                for j in self.y_vars[i]:
                    target = value(self.lsmhe.yk0_mhe[t, k]) - value(self.yk_l[t][k])
                    self.lsmhe.nuk_mhe[t, k].set_value(target)
                    k += 1

    def adjust_w_mhe(self):
        return
        for i in range(1, self.nfe_t+1):
            j = 0
            for x in self.x_noisy:
                x_var = getattr(self.lsmhe, x)
                for k in self.x_vars[x]:
                    x1pvar_val = value(x_var[(i+1, 0), k])
                    x1var_val = value(x_var[(i, self.ncp_t), k])
                    if self.IgnoreProcessNoise:
                        pass
                    else:
                        self.lsmhe.wk_mhe[i, j].set_value(0.0)
                    j += 1

    def set_covariance_meas(self, cov_dict):
        """Sets covariance(inverse) for the measurements.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(meas_name, j), (meas_name, k), time]
        Returns:
            None
        """
        rtarget = getattr(self.lsmhe, "R_mhe")
        for key in cov_dict:
            vni = key[0]
            vnj = key[1]
            _t = key[2]

            v_i = self.yk_key[vni]
            v_j = self.yk_key[vnj]
            # try:
            if self.diag_Q_R:
                rtarget[_t, v_i] = 1 / cov_dict[vni, vnj, _t]
            else:
                rtarget[_t, v_i, v_j] = cov_dict[vni, vnj, _t]
            # except KeyError:
            #     print("Key error, {:} {:} {:}".format(vni, vnj, _t))

    def set_covariance_disturb(self, cov_dict):
        """Sets covariance(inverse) for the states.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(state_name, j), (state_name, k), time]
        Returns:
            None
        """
        qtarget = getattr(self.lsmhe, "Q_mhe")
        for key in cov_dict:
            vni = key[0]
            vnj = key[1]
            _t = key[2]
            v_i = self.xkN_key[vni]
            v_j = self.xkN_key[vnj]
            if self.diag_Q_R:
                qtarget[_t, v_i] = 1 / cov_dict[vni, vnj, _t]
            else:
                qtarget[_t, v_i, v_j] = cov_dict[vni, vnj, _t]

    def set_covariance_u(self, cov_dict):
        """Sets covariance(inverse) for the states.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(state_name, j), (state_name, k), time]
        Returns:
            None
        """
        qtarget = getattr(self.lsmhe, "U_mhe")
        for key in cov_dict:
            vni = key[0]
            _t = key[1]
            qtarget[_t, vni] = 1 / cov_dict[vni, _t]

    def shift_mhe(self):
        """Shifts current initial guesses of variables for the mhe problem"""
        for v in self.lsmhe.component_objects(Var, active=True):
            if type(v.index_set()) == SimpleSet:  #: Don't want simple sets
                continue
            else:
                kl = v.keys()
                if len(kl[0]) < 2:
                    continue
                for k in kl:
                    if k[0] < self.nfe_t:
                        try:
                            v[k].set_value(v[(k[0] + 1,) + k[1:]])
                        except ValueError:
                            continue

    def shift_measurement_input_mhe(self):
        """Shifts current measurements for the mhe problem"""
        y0 = getattr(self.lsmhe, "yk0_mhe")
        for i in range(2, self.nfe_t + 1):
            for j in self.lsmhe.yk0_mhe.keys():
                y0[i-1, j[1:]].value = value(y0[i, j[1:]])
            for u in self.u:
                umhe = getattr(self.lsmhe, u)
                umhe[i-1] = value(umhe[i])
        self.adjust_nu0_mhe()


    def load_input_mhe(self, src_kind, **kwargs):
        """Loads inputs into the mhe model"""
        src = kwargs.pop("src", self.d1)
        fe = kwargs.pop("fe", 1)
        # src_kind = kwargs.pop("src_kind", "mod")
        if src_kind == "mod":
            for u in self.u:
                usrc = getattr(src, u)
                utrg = getattr(self.lsmhe, u)
                utrg[fe].value = value(usrc[1])
        elif src_kind == "self.dict":
            for u in self.u:
                utrg = getattr(self.lsmhe, u)
                utrg[fe].value = value(self.curr_u[u])

    def init_step_mhe(self, tgt, i, patch_pred_y=False):
        """Takes the last state-estimate from the mhe to perform an open-loop simulation
        that initializes the last slice of the mhe horizon
        Args:
            tgt (pyomo.core.base.PyomoModel.ConcreteModel): The target model
            i (int): finite element of lsmhe
            patch_y (bool): If true, patch the measurements as well"""
        src = self.lsmhe
        for vs in src.component_objects(Var, active=True):
            if vs.getname()[-4:] == "_mhe":
                continue
            vd = getattr(tgt, vs.getname())
            # there are two cases: 1 key 1 elem, several keys 1 element
            vskeys = vs.keys()
            if len(vskeys) == 1:
                #: One key
                for ks in vskeys:
                    for v in vd.itervalues():
                        v.set_value(value(vs[ks]))
            else:
                k = 0
                for ks in vskeys:
                    if k == 0:
                        if type(ks) != tuple:
                            #: Several keys of 1 element each!!
                            vd[1].set_value(value(vs[vskeys[-1]]))  #: This has got to be true
                            break
                        k += 1
                    kj = ks[2:]
                    if vs.getname() in self.states:  #: States start at 0
                        for j in range(0, self.ncp_t + 1):
                            vd[(1, j) + kj].set_value(value(vs[(i, j) + kj]))
                    else:
                        for j in range(1, self.ncp_t + 1):
                            vd[(1, j) + kj].set_value(value(vs[(i, j) + kj]))

        for u in self.u:  #: This should update the inputs
            usrc = getattr(src, u)
            utgt = getattr(tgt, u)
            utgt[1] = (value(usrc[i]))
        for x in self.states:
            pn = x + "_ic"
            p = getattr(tgt, pn)
            vs = getattr(self.lsmhe, x)
            for ks in p.iterkeys():
                p[ks].value = value(vs[(i, self.ncp_t) + (ks,)])

        test = self.solve_d(tgt, o_tee=False, stop_if_nopt=False, max_cpu_time=300,
                            jacobian_regularization_value=1e-04,
                            jacobian_regularization_exponent=2.,
                            halt_on_ampl_error=False,
                            output_file="init_mhe.txt")
        # if test != 0:
        #     self.lsmhe.write_nl(name="failed_mhe.nl")
        self.load_d_d(tgt, self.lsmhe, self.nfe_t)



        if patch_pred_y:
            self.journalizer("I", self._c_it, "init_step_mhe", "Prediction for advanced-step.. Ready")
            self.patch_meas_mhe(self.nfe_t, src=tgt, noisy=True)
        self.adjust_nu0_mhe()
        self.adjust_w_mhe()

    def create_rh_sfx(self, set_suffix=True):
        """Creates relevant suffixes for k_aug (prior at fe=2) (Reduced_Hess)
        Args:
            set_suffix (bool): True if update must be done
        Returns:
            None
        """
        if hasattr(self.lsmhe, "dof_v"):
            self.lsmhe.dof_v.clear()
        else:
            self.lsmhe.dof_v = Suffix(direction=Suffix.EXPORT)  #: dof_v
        if hasattr(self.lsmhe, "rh_name"):
            self.lsmhe.rh_name.clear()
        else:
            self.lsmhe.rh_name = Suffix(direction=Suffix.IMPORT)  #: Red_hess_name

        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)

        if set_suffix:
            for key in self.x_noisy:
                var = getattr(self.lsmhe, key)
                for j in self.x_vars[key]:
                    var[(2, 0) + j].set_suffix_value(self.lsmhe.dof_v, 1)

    def create_sens_suffix_mhe(self, set_suffix=True):
        """Creates relevant suffixes for k_aug (Sensitivity)
        Args:
            set_suffix (bool): True if update must be done
        Returns:
            None"""
        if hasattr(self.lsmhe, "dof_v"):
            self.lsmhe.dof_v.clear()
        else:
            self.lsmhe.dof_v = Suffix(direction=Suffix.EXPORT)  #: dof_v
        if hasattr(self.lsmhe, "rh_name"):
            self.lsmhe.rh_name.clear()
        else:
            self.lsmhe.rh_name = Suffix(direction=Suffix.IMPORT)  #: Red_hess_name
        if set_suffix:
            for key in self.x_noisy:
                var = getattr(self.lsmhe, key)
                for j in self.x_vars[key]:
                    var[(self.nfe_t, self.ncp_t) + j].set_suffix_value(self.lsmhe.dof_v, 1)

    def check_active_bound_noisy(self):
        """Checks if the dof_(super-basic) have active bounds, if so, add them to the exclusion list"""
        if hasattr(self.lsmhe, "dof_v"):
            self.lsmhe.dof_v.clear()
        else:
            self.lsmhe.dof_v = Suffix(direction=Suffix.EXPORT)  #: dof_v
        if hasattr(self.lsmhe, "rh_name"):
            self.lsmhe.rh_name.clear()
        else:
            self.lsmhe.rh_name = Suffix(direction=Suffix.IMPORT)  #: Red_hess_name

        self.xkN_nexcl = []
        k = 0
        for x in self.x_noisy:
            v = getattr(self.lsmhe, x)
            for j in self.x_vars[x]:
                active_bound = False
                if v[(2, 0) + j].lb:
                    if v[(2, 0) + j].value - v[(2, 0) + j].lb < 1e-08:
                        active_bound = True
                if v[(2, 0) + j].ub:
                    if v[(2, 0) + j].ub - v[(2, 0) + j].value < 1e-08:
                        active_bound = True
                if active_bound:
                    print("Active bound {:s}, {:d}, value {:f}".format(x, j[0], v[(2, 0) + j].value), file=sys.stderr)
                    v[(2, 0) + j].set_suffix_value(self.lsmhe.dof_v, 0)
                    self.xkN_nexcl.append(0)
                    k += 1
                else:
                    v[(2, 0) + j].set_suffix_value(self.lsmhe.dof_v, 1)
                    self.xkN_nexcl.append(1)  #: Not active, add it to the non-exclusion list.
        if k > 0:
            print("I[[check_active_bound_noisy]] {:d} Active bounds.".format(k))

    def deact_icc_mhe(self):
        """Deactivates the icc constraints in the mhe problem"""
        if self.deact_ics:
            for i in self.x_noisy:
                try:
                    ic_con = getattr(self.lsmhe, i + "_icc")
                    for k in self.x_vars[i]:
                        ic_con[k].deactivate()
                    # self.lsmhe.del_component(ic_con[k])
                except AttributeError:
                    continue


        #: Maybe only for a subset of the states
        else:
            for i in self.x_noisy:
                # if i in self.x_noisy:
                ic_con = getattr(self.lsmhe, i + "_icc")
                for k in self.x_vars[i]:
                        ic_con[k].deactivate()

    def regen_objective_fun(self):
        """Given the exclusion list, regenerate the expression for the arrival cost"""
        self.lsmhe.Arrival_e_mhe.set_value(0.5 * sum((self.xkN_l[j] - self.lsmhe.x_0_mhe[j]) *
                                                     sum(self.lsmhe.PikN_mhe[j, k] *
                                                         (self.xkN_l[k] - self.lsmhe.x_0_mhe[k]) for k in
                                                         self.lsmhe.xkNk_mhe if self.xkN_nexcl[k])
                                                     for j in self.lsmhe.xkNk_mhe if self.xkN_nexcl[j]))
        self.lsmhe.obfun_mhe.set_value(self.lsmhe.Arrival_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.R_e_mhe)
        if self.lsmhe.obfun_dum_mhe.active:
            self.lsmhe.obfun_dum_mhe.deactivate()
        if not self.lsmhe.obfun_mhe.active:
            self.lsmhe.obfun_mhe.activate()

    def load_covariance_prior(self):
        """Computes the reduced-hessian (inverse of the prior-covariance)
        Reads the result_hessian.txt file that contains the covariance information"""
        self.journalizer("I", self._c_it, "load_covariance_prior", "K_AUG w red_hess")
        self.k_aug.options["compute_inv"] = ""
        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        self.create_rh_sfx()
        self.k_aug.solve(self.lsmhe, tee=True)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)

        self._PI.clear()
        with open("inv_.in", "r") as rh:
            ll = []
            l = rh.readlines()
            row = 0
            for i in l:
                ll = i.split()
                col = 0
                for j in ll:
                    self._PI[row, col] = float(j)
                    col += 1
                row += 1
            rh.close()
        print("-" * 120)
        print("I[[load covariance]] e-states nrows {:d} ncols {:d}".format(len(l), len(ll)))
        print("-" * 120)

    def set_state_covariance(self):
        """Sets covariance(inverse) for the prior_state.
        Args:
            None
        Return:
            None
        """
        pikn = getattr(self.lsmhe, "PikN_mhe")
        for key_j in self.x_noisy:
            for key_k in self.x_noisy:
                vj = getattr(self.lsmhe, key_j)
                vk = getattr(self.lsmhe, key_k)
                for j in self.x_vars[key_j]:
                    if vj[(2, 0) + j].get_suffix_value(self.lsmhe.dof_v) == 0:
                        #: This state is at its bound, skip
                        continue
                    for k in self.x_vars[key_k]:
                        if vk[(2, 0) + k].get_suffix_value(self.lsmhe.dof_v) == 0:
                            #: This state is at its bound, skip
                            print("vj {:s} {:d} .sfx={:d}, vk {:s} {:d}.sfx={:d}"
                                  .format(key_j, j[0], vj[(2, 0) + j].get_suffix_value(self.lsmhe.dof_v),
                                          key_k, k[0], vk[(2, 0) + k].get_suffix_value(self.lsmhe.dof_v),))
                            continue
                        row = vj[(2, 0) + j].get_suffix_value(self.lsmhe.rh_name)
                        col = vk[(2, 0) + k].get_suffix_value(self.lsmhe.rh_name)
                        #: Ampl does not give you back 0's
                        if not row:
                            row = 0
                        if not col:
                            col = 0

                        # print((row, col), (key_j, j), (key_k, k))
                        q0j = self.xkN_key[key_j, j]
                        q0k = self.xkN_key[key_k, k]
                        pi = self._PI[row, col]
                        try:
                            pikn[q0j, q0k] = pi
                        except KeyError:
                            errk = key_j + "_" + str(j) + ", " + key_k + "_" + str(k)
                            print("Kerror, var {:}".format(errk))
                            pikn[q0j, q0k] = 0.0

    def set_prior_state_from_prior_mhe(self):
        """Mechanism to assign a value to x0 (prior-state) from the previous mhe
        Args:
            None
        Returns:
            None
        """
        for x in self.x_noisy:
            var = getattr(self.lsmhe, x)
            for j in self.x_vars[x]:
                z0dest = getattr(self.lsmhe, "x_0_mhe")
                z0 = self.xkN_key[x, j]
                z0dest[z0] = value(var[(2, 0,) + j])

    def update_noise_meas(self, mod, cov_dict):
        self.journalizer("I", self._c_it, "introduce_noise_meas", "Noise introduction")
        # f = open("m0.txt", "w")
        # f1 = open("m1.txt", "w")
        for y in self.y:
            vy = getattr(mod,  y)
            # vy.display(ostream=f)
            for j in self.y_vars[y]:
                vv = value(vy[(1, self.ncp_t) + j])
                sigma = cov_dict[(y, j), (y, j), 1]
                self.curr_m_noise[(y, j)] = np.random.normal(0, sigma)
                # noise = np.random.normal(0, sigma)
                # # print(noise)
                # vv += noise
                # vy[(1, self.ncp_t) + j].set_value(vv)
            # vy.display(ostream=f1)
        # f.close()
        # f1.close()

    def print_r_mhe(self):
        self.journalizer("I", self._c_it, "print_r_mhe", "Results at" + os.getcwd())
        self.journalizer("I", self._c_it, "print_r_mhe", "Results suffix " + self.res_file_suf)
        for x in self.x_noisy:
            elist = []
            rlist = []
            xe = getattr(self.lsmhe, x)
            xr = getattr(self.d1, x)
            for j in self.x_vars[x]:
                elist.append(value(xe[(self.nfe_t, self.ncp_t) + j]))
                rlist.append(value(xr[(1, self.ncp_t) + j]))
            self.s_estimate[x].append(elist)
            self.s_real[x].append(rlist)

        # with open("res_mhe_ee.txt", "w") as f:
        #     for x in self.x_noisy:
        #         for j in range(0, len(self.s_estimate[x][0])):
        #             for i in range(0, len(self.s_estimate[x])):
        #                 xvs = str(self.s_estimate[x][i][j])
        #                 f.write(xvs)
        #                 f.write('\t')
        #             f.write('\n')
        #     f.close()

        with open("res_mhe_es_" + self.res_file_suf + ".txt", "a") as f:
            for x in self.x_noisy:
                for j in self.s_estimate[x][-1]:
                    xvs = str(j)
                    f.write(xvs)
                    f.write('\t')
            f.write('\n')
            f.close()
        with open("res_mhe_rs_" + self.res_file_suf + ".txt", "a") as f:
            for x in self.x_noisy:
                for j in self.s_real[x][-1]:
                    xvs = str(j)
                    f.write(xvs)
                    f.write('\t')
            f.write('\n')
            f.close()
        with open("res_mhe_eoff_" + self.res_file_suf + ".txt", "a") as f:
            for x in self.x_noisy:
                for j in range(0, len(self.s_estimate[x][-1])):
                    e = self.s_estimate[x][-1][j]
                    r = self.s_real[x][-1][j]
                    xvs = str(e-r)
                    f.write(xvs)
                    f.write('\t')
            f.write('\n')
            f.close()
        # with open("res_mhe_ereal.txt", "w") as f:
        #     for x in self.x_noisy:
        #         for j in range(0, len(self.s_real[x][0])):
        #             for i in range(0, len(self.s_real[x])):
        #                 xvs = str(self.s_real[x][i][j])
        #                 f.write(xvs)
        #                 f.write('\t')
        #             f.write('\n')
        #     f.close()

        for y in self.y:
            elist = []
            rlist = []
            nlist = []
            yklst = []
            ye = getattr(self.lsmhe, y)
            yr = getattr(self.d1, y)
            for j in self.y_vars[y]:
                elist.append(value(ye[(self.nfe_t, self.ncp_t) + j]))
                rlist.append(value(yr[(1, self.ncp_t) + j]))
                nlist.append(self.curr_m_noise[(y, j)])
                yklst.append(value(self.lsmhe.yk0_mhe[self.nfe_t, self.yk_key[(y, j)]]))
            self.y_estimate[y].append(elist)
            self.y_real[y].append(rlist)
            self.y_noise_jrnl[y].append(nlist)
            self.yk0_jrnl[y].append(yklst)

        # with open("res_mhe_ey.txt", "w") as f:
        #     for y in self.y:
        #         for j in range(0, len(self.y_estimate[y][0])):
        #             for i in range(0, len(self.y_estimate[y])):
        #                 yvs = str(self.y_estimate[y][i][j])
        #                 f.write(yvs)
        #                 f.write('\t')
        #             f.write('\n')
        #     f.close()

        with open("res_mhe_ey_" + self.res_file_suf + ".txt", "a") as f:
            for y in self.y:
                for j in self.y_estimate[y][-1]:
                    yvs = str(j)
                    f.write(yvs)
                    f.write('\t')
            f.write('\n')
            f.close()

        with open("res_mhe_yreal_" + self.res_file_suf + ".txt", "a") as f:
            for y in self.y:
                for j in self.y_real[y][-1]:
                    yvs = str(j)
                    f.write(yvs)
                    f.write('\t')
            f.write('\n')
            f.close()

        with open("res_mhe_yk0_" + self.res_file_suf + ".txt", "a") as f:
            for y in self.y:
                for j in self.yk0_jrnl[y][-1]:
                    yvs = str(j)
                    f.write(yvs)
                    f.write('\t')
            f.write('\n')
            f.close()

        with open("res_mhe_ynoise_" + self.res_file_suf + ".txt", "a") as f:
            for y in self.y:
                for j in self.y_noise_jrnl[y][-1]:
                    yvs = str(j)
                    f.write(yvs)
                    f.write('\t')
            f.write('\n')
            f.close()

        with open("res_mhe_yoffset_" + self.res_file_suf + ".txt", "a") as f:
            for y in self.y:
                for j in self.y_vars[y]:
                    yvs = str(self.curr_y_offset[(y, j)])
                    f.write(yvs)
                    f.write('\t')
            f.write('\n')
            f.close()

        with open("res_mhe_unoise_" + self.res_file_suf + ".txt", "a") as f:
            for u in self.u:
                # u_mhe = getattr(self.lsmhe, u)
                ue_mhe = getattr(self.lsmhe, "w_" + u + "_mhe")
                for i in self.lsmhe.fe_t:
                    dv = value(ue_mhe[i])
                    dstr = str(dv)
                    f.write(dstr)
                    f.write('\t')
            f.write('\n')
            f.close()

    def compute_y_offset(self, noisy=True):
        mhe_y = getattr(self.lsmhe, "yk0_mhe")
        for y in self.y:
            plant_y = getattr(self.d1, y)
            for j in self.y_vars[y]:
                k = self.yk_key[(y, j)]
                mhe_yval = value(mhe_y[self.nfe_t, k])
                plant_yval = value(plant_y[(1, self.ncp_t) + j])
                y_noise = self.curr_m_noise[(y, j)] if noisy else 0.0
                self.curr_y_offset[(y, j)] = mhe_yval - plant_yval - y_noise

    def sens_dot_mhe(self):
        """Updates suffixes, solves using the dot_driver"""
        self.journalizer("I", self._c_it, "sens_dot_mhe", "Set-up")

        if hasattr(self.lsmhe, "npdp"):
            self.lsmhe.npdp.clear()
        else:
            self.lsmhe.npdp = Suffix(direction=Suffix.EXPORT)
        self.create_sens_suffix_mhe()
        for y in self.y:
            for j in self.y_vars[y]:
                k = self.yk_key[(y, j)]
                self.lsmhe.hyk_c_mhe[self.nfe_t, k].set_suffix_value(self.lsmhe.npdp, self.curr_y_offset[(y, j)])



        # with open("somefile0.txt", "w") as f:
        #     self.lsmhe.x.display(ostream=f)
        #     self.lsmhe.M.display(ostream=f)
        #     f.close()
        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        #: Looks for the file with the timestamp
        self.lsmhe.set_suffix_value(self.lsmhe.f_timestamp, self.int_file_mhe_suf)

        self.lsmhe.f_timestamp.display(ostream=sys.stderr)

        self.journalizer("I", self._c_it, "sens_dot_mhe", self.lsmhe.name)

        results = self.dot_driver.solve(self.lsmhe, tee=True, symbolic_solver_labels=True)
        self.lsmhe.solutions.load_from(results)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)

        # with open("somefile1.txt", "w") as f:
        #     self.lsmhe.x.display(ostream=f)
        #     self.lsmhe.M.display(ostream=f)
        #     f.close()

        ftiming = open("timings_dot_driver.txt", "r")
        s = ftiming.readline()
        ftiming.close()
        k = s.split()
        self._dot_timing = k[0]

    def sens_k_aug_mhe(self):
        self.journalizer("I", self._c_it, "sens_k_aug_mhe", "k_aug sensitivity")
        self.lsmhe.ipopt_zL_in.update(self.lsmhe.ipopt_zL_out)
        self.lsmhe.ipopt_zU_in.update(self.lsmhe.ipopt_zU_out)
        self.journalizer("I", self._c_it, "sens_k_aug_mhe", self.lsmhe.name)

        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        #: Now, the sensitivity step will have the timestamp for dot_in

        self.lsmhe.set_suffix_value(self.lsmhe.f_timestamp, self.int_file_mhe_suf)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)
        self.create_sens_suffix_mhe()
        results = self.k_aug_sens.solve(self.lsmhe, tee=True, symbolic_solver_labels=True)
        self.lsmhe.solutions.load_from(results)
        self.lsmhe.f_timestamp.display(ostream=sys.stderr)
        ftimings = open("timings_k_aug.txt", "r")
        s = ftimings.readline()
        ftimings.close()
        self._k_timing = s.split()

    def update_state_mhe(self, as_nmpc_mhe_strategy=False):
        # Improvised strategy
        if as_nmpc_mhe_strategy:
            self.journalizer("I", self._c_it, "update_state_mhe", "offset ready for asnmpcmhe")
            for x in self.states:
                xvar = getattr(self.lsmhe, x)
                x0 = getattr(self.olnmpc, x + "_ic")
                for j in self.state_vars[x]:
                    # self.curr_state_offset[(x, j)] = self.curr_estate[(x, j)] - value(xvar[self.nfe_t, self.ncp_t, j])
                    self.curr_state_offset[(x, j)] = value(x0[j] )- value(xvar[self.nfe_t, self.ncp_t, j])
                    print("state !", self.curr_state_offset[(x, j)])

        for x in self.states:
            xvar = getattr(self.lsmhe, x)
            for j in self.state_vars[x]:
                self.curr_estate[(x, j)] = value(xvar[self.nfe_t, self.ncp_t, j])


    def method_for_mhe_simulation_step(self):
        pass

    def deb_alg_sys(self):
        """Debugging the algebraic system"""
        # Fix differential states
        # Deactivate ODEs de_
        # Deactivate FE cont cp_
        # Deactivate IC _icc
        # Deactivate coll dvar_t_

        # Deactivate hyk
        for i in self.x_noisy:
            x = getattr(self.lsmhe, i)
            x.fix()
            cp_con = getattr(self.lsmhe, "cp_" + i)
            cp_con.deactivate()
            de_con = getattr(self.lsmhe, "de_" + i)
            de_con.deactivate()
            icc_con = getattr(self.lsmhe, i + "_icc")
            icc_con.deactivate()
            dvar_con = getattr(self.lsmhe, "dvar_t_" + i)
            dvar_con.deactivate()

        self.lsmhe.obfun_dum_mhe.deactivate()
        self.lsmhe.obfun_dum_mhe_deb.activate()
        self.lsmhe.hyk_c_mhe.deactivate()
        self.lsmhe.noisy_cont.deactivate()

        for u in self.u:
            cc = getattr(self.lsmhe, u + "_c")  #: Get the constraint for input
            con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")  #: Get the constraint-noisy
            cc.deactivate()
            con_w.deactivate()

        # self.lsmhe.pprint(filename="algeb_mod.txt")
