# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, Set, Constraint, Expression, Param, Suffix, ConstraintList, TransformationFactory
from pyomo.core.base.sets import SimpleSet
from pyomo.dae import *
from pyomo.core.kernel.numvalue import value
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from nmpc_mhe.pyomo_dae.NMPCGen_pyDAE import NmpcGen_DAE
import numpy as np
from itertools import product
import sys, os, time
from pyutilib.common._exceptions import ApplicationError
from nmpc_mhe.aux.utils import t_ij
from nmpc_mhe.aux.utils import fe_compute, load_iguess, augment_model
from copy import deepcopy
__author__ = "David Thierry @dthierry" #: March 2018


class MheGen_DAE(NmpcGen_DAE):
    def __init__(self, d_mod, hi_t, states, controls, noisy_states, measurements, **kwargs):
        NmpcGen_DAE.__init__(self, d_mod, hi_t, states, controls, **kwargs)
        self.int_file_mhe_suf = int(time.time())-1

        # Need a list of relevant measurements y
        self.y = measurements
        self.y_vars = dict()

        # Need a list or relevant noisy-states z
        self.x_noisy = noisy_states
        self.x_vars = dict()
        self.deact_ics = kwargs.pop('del_ics', True)
        self.diag_Q_R = kwargs.pop('diag_QR', True)  #: By default use diagonal matrices for Q and R matrices
        if self.diag_Q_R:
            self.journalist('W', self._iteration_count, "Initializing MHE", "The Q_MHE and R_MHE matrices are diagonal")
        # self.u = kwargs.pop('u', [])

        self.IgnoreProcessNoise = kwargs.pop('IgnoreProcessNoise', False)
        # One can specify different discretization lenght
        self.nfe_tmhe = kwargs.pop('nfe_tmhe', self.nfe_t)  #: Specific number of finite elements
        self.ncp_tmhe = kwargs.pop('ncp_tmhe', self.ncp_t)  #: Specific number of collocation points

        # nstates = sum(len(self.x_vars[x]) for x in self.x_noisy)
        # self.journalist("I", self._iteration_count, "MHE with \t", str(nstates) + "states")

        _t_mhe = self.nfe_tmhe * self.hi_t

        self.lsmhe = self.d_mod(self.nfe_tmhe, self.ncp_tmhe, _t=_t_mhe)
        # self.dum_mhe = self.d_mod(1, self.ncp_tmhe, _t=self.hi_t)

        self.dum_mhe = self.lsmhe.clone()

        self.dum_mhe.t._bounds = (0, self.hi_t)
        self.dum_mhe.t.clear()
        self.dum_mhe.t.construct()
        self.dum_mhe.t.pprint()
        self.dum_mhe.name = "Dummy [MHE]"
        for i in self.dum_mhe.component_objects(Var):
            i.clear()
            i.construct()
        for i in self.dum_mhe.component_objects(Constraint):
            i.clear()
            i.construct()

        augment_model(self.lsmhe)
        augment_model(self.dum_mhe)

        d = TransformationFactory('dae.collocation')
        d.apply_to(self.lsmhe, nfe=self.nfe_tmhe, ncp=self.ncp_tmhe, scheme="LAGRANGE-RADAU")

        self.lsmhe.name = "LSMHE (Least-Squares MHE)"
        self.lsmhe.create_bounds()
        #: create x_pi constraint
        #: Create list of noisy-states vars
        self.xkN_l = []
        self.xkN_nexcl = []
        self.xkN_key = {}
        tS_mhe = self.lsmhe.t
        k = 0
        for x in self.x_noisy:
            n_s = getattr(self.lsmhe, x)  #: Noisy-state
            self.x_vars[x] = list()
            if not n_s._implicit_subsets:
                raise Exception("not right")
            if tS_mhe not in n_s._implicit_subsets:
                raise Exception("not right")
            remaining_set = n_s._implicit_subsets[1]
            for jth_set in range(2, len(n_s._implicit_subsets)):
                remaining_set *= jth_set
            # for jth in self.lsmhe.fe_t:  #: the jth variable
            for kth in remaining_set:
                kth = kth if isinstance(kth, tuple) else (kth,)
                self.x_vars[x].append(kth)
                self.xkN_l.append(n_s[(0,) + kth])
                self.xkN_nexcl.append(1)  #: non-exclusion list for active bounds
                self.xkN_key[(x,) + kth] = k
                k += 1

        #: self.lsmhe.fe_t won't work with the pyomo.dae; re-define a new one
        self.lsmhe.fe_t = Set(initialize=[i for i in range(0, self.nfe_tmhe)])  #: Set for the MHE stuff
        self.lsmhe.xkNk_mhe = Set(initialize=[i for i in range(0, len(self.xkN_l))])  #: Create set of noisy_states
        self.lsmhe.x_0_mhe = Param(self.lsmhe.xkNk_mhe, initialize=0.0, mutable=True)  #: Prior-state
        self.lsmhe.wk_mhe = Param(self.lsmhe.fe_t, self.lsmhe.xkNk_mhe, initialize=0.0) \
            if self.IgnoreProcessNoise else Var(self.lsmhe.fe_t, self.lsmhe.xkNk_mhe, initialize=0.0)  #: Model disturbance
        self.lsmhe.PikN_mhe = Param(self.lsmhe.xkNk_mhe, self.lsmhe.xkNk_mhe,
                                initialize=lambda m, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Prior-Covariance
        self.lsmhe.Q_mhe = Param(range(0, (self.nfe_tmhe - 1)), self.lsmhe.xkNk_mhe, initialize=1, mutable=True) if self.diag_Q_R\
            else Param(range(0, (self.nfe_tmhe - 1)), self.lsmhe.xkNk_mhe, self.lsmhe.xkNk_mhe,
                             initialize=lambda m, t, i, ii: 1. if i == ii else 0.0, mutable=True)  #: Disturbance-weight

        #: Create list of measurements vars
        self.yk_l = {}
        self.yk_key = {}
        k = 0
        self.yk_l[0] = []
        for y in self.y:
            m_v = getattr(self.lsmhe, y)  #: Measured "state"
            self.y_vars[y] = list()
            if not m_v._implicit_subsets:
                raise Exception("not right")
            if tS_mhe not in m_v._implicit_subsets:
                raise Exception("not right")
            remaining_set = m_v._implicit_subsets[1]
            for jth_set in range(2, len(m_v._implicit_subsets)):
                remaining_set *= jth_set
            # for jth in self.lsmhe.fe_t:  #: the jth variable
            t = t_ij(tS_mhe, 0, self.ncp_tmhe)
            for kth in remaining_set:
                kth = kth if isinstance(kth, tuple) else (kth,)
                self.y_vars[y].append(kth)
                self.yk_l[0].append(m_v[(t,) + kth])
                #: position of the variable in the list
                self.yk_key[(y,) + kth] = k  #: The key needs to be created only once, that is why the loop was split
                k += 1

        for fe in self.lsmhe.fe_t:
            if fe == 0:
                continue
            self.yk_l[fe] = []
            for y in self.y:
                m_v = getattr(self.lsmhe, y)  #: Measured "state"
                remaining_set = m_v._implicit_subsets[1]
                for jth_set in range(2, len(m_v._implicit_subsets)):
                    remaining_set *= jth_set
                t = t_ij(tS_mhe, fe, self.ncp_tmhe)
                for kth in remaining_set:
                    kth = kth if isinstance(kth, tuple) else (kth,)
                    self.yk_l[fe].append(m_v[(t,) + kth])

        self.lsmhe.ykk_mhe = Set(initialize=[i for i in range(0, len(self.yk_l[0]))])  #: Create set of measured_vars
        self.lsmhe.nuk_mhe = Var(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=0.0)   #: Measurement noise
        self.lsmhe.yk0_mhe = Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=1.0, mutable=True)
        self.lsmhe.hyk_c_mhe = Constraint(self.lsmhe.fe_t, self.lsmhe.ykk_mhe,
                                          rule=
                                          lambda mod, t, i:mod.yk0_mhe[t, i] - self.yk_l[t][i] - mod.nuk_mhe[t, i] == 0.0)
        #: This will work because yk_l is indexed by fe
        self.lsmhe.hyk_c_mhe.deactivate()
        self.lsmhe.R_mhe = Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, initialize=1.0, mutable=True) if self.diag_Q_R else \
            Param(self.lsmhe.fe_t, self.lsmhe.ykk_mhe, self.lsmhe.ykk_mhe,
                             initialize=lambda mod, t, i, ii: 1.0 if i == ii else 0.0, mutable=True)

        #: Constraints for the input noise
        tfe_mhe_dic = dict()
        for t in self.lsmhe.t:
            if t == max(self.lsmhe.t):
                tfe_mhe_dic[t] = fe_compute(tS_mhe, t - 1)
            else:
                tfe_mhe_dic[t] = fe_compute(tS_mhe, t)

        for u in self.u:  #: u only has one index
            cv = getattr(self.lsmhe, u)  #: Get the param
            t_u = [t_ij(tS_mhe, i, 0) for i in range(0, self.lsmhe.nfe_t)]
            c_val = [value(cv[t_u[i]]) for i in self.lsmhe.fe_t]  #: Current value
            dumm_eq = getattr(self.lsmhe, u + '_cdummy')
            dexpr = dumm_eq[0].expr._args[0]
            control_var = getattr(self.lsmhe, dexpr.parent_component().name)
            if isinstance(control_var, Var): #: all good
                pass
            else:
                raise ValueError  #: Some exception here
            dumm_eq.clear()
            self.lsmhe.del_component(cv)  #: Delete the dummy_param
            # self.lsmhe.del_component(dumm_eq)  #: Delete the dummy_constraint
            #: Change this guy to mutable parameter [piece-wise constant]

            self.lsmhe.add_component(u, Param(self.lsmhe.fe_t, mutable=True, initialize=lambda m, i: c_val[i]))
            self.lsmhe.add_component('w_' + u + '_mhe', Var(self.lsmhe.fe_t, initialize=0.0))
            cv_param = getattr(self.lsmhe, u)  #: Get the new variable
            cv_noise = getattr(self.lsmhe, 'w_' + u + '_mhe')
            self.lsmhe.add_component(u + '_cdummy_mhe', Constraint(self.lsmhe.t,
                                                               rule=lambda m, i:
                                                               cv_param[tfe_mhe_dic[i]] == control_var[i] + cv_noise[tfe_mhe_dic[i]]))
            dumm_eq.rule = lambda m, i: cv_param[tfe_mhe_dic[i]] == control_var[i]
            cv_con = getattr(self.lsmhe, u + '_cdummy_mhe')
            cv_con.deactivate()

        self.lsmhe.U_mhe = Param(self.lsmhe.fe_t, self.u, initialize=1, mutable=True)

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

        #: Put the noise in the ode
        #: Cant make it a set of individual constraints bc. I can't access invidual components...
        #: of a constraint
        self.lsmhe.noisy_cont = ConstraintList()
        j = 0
        for i in self.x_noisy:
            oc_e = getattr(self.lsmhe, i + "dot_disc_eq")
            for k in self.x_vars[i]:  #: This should keep the same order
                for t in self.lsmhe.t:
                    if t == 0 or tfe_mhe_dic[t] == self.lsmhe.nfe_t - 1:
                        continue
                    e = oc_e[(t,) + k].expr
                    j = self.xkN_key[(i,) + k]
                    self.lsmhe.noisy_cont.add(e._args[0] == self.lsmhe.wk_mhe[tfe_mhe_dic[t], j])
                # j += 1
        self.lsmhe.noisy_cont.deactivate()

        #: Expressions for the objective function (least-squares)
        self.lsmhe.Q_e_mhe = 0.0 if self.IgnoreProcessNoise else Expression(
            expr=0.5 * sum(
                sum(
                    self.lsmhe.Q_mhe[i, k] * self.lsmhe.wk_mhe[i, k]**2 for k in self.lsmhe.xkNk_mhe)
                for i in range(0, self.nfe_tmhe - 1))) if self.diag_Q_R else Expression(
            expr=sum(sum(self.lsmhe.wk_mhe[i, j] *
                         sum(self.lsmhe.Q_mhe[i, j, k] * self.lsmhe.wk_mhe[i, k] for k in self.lsmhe.xkNk_mhe)
                         for j in self.lsmhe.xkNk_mhe) for i in range(0, self.nfe_tmhe - 1)))

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

        self.lsmhe.Arrival_e_mhe = Expression(
            expr=0.5 * sum((self.xkN_l[j] - self.lsmhe.x_0_mhe[j]) *
                     sum(self.lsmhe.PikN_mhe[j, k] * (self.xkN_l[k] - self.lsmhe.x_0_mhe[k]) for k in self.lsmhe.xkNk_mhe)
                     for j in self.lsmhe.xkNk_mhe))

        self.lsmhe.Arrival_dummy_e_mhe = Expression(
            expr=100000.0 * sum((self.xkN_l[j] - self.lsmhe.x_0_mhe[j]) ** 2 for j in self.lsmhe.xkNk_mhe))

        self.lsmhe.obfun_dum_mhe_deb = Objective(sense=minimize, expr=1.0)
        #: no arrival
        self.lsmhe.obfun_dum_mhe = Objective(sense=minimize,
                                             expr=self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.U_e_mhe)
        self.lsmhe.obfun_dum_mhe.deactivate()

        self.lsmhe.obfun_mhe_first = Objective(sense=minimize,
                                         expr=self.lsmhe.Arrival_dummy_e_mhe)
        self.lsmhe.obfun_mhe_first.deactivate()

        self.lsmhe.obfun_mhe = Objective(sense=minimize,
                                         expr=self.lsmhe.Arrival_e_mhe + self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe +
                                              self.lsmhe.U_e_mhe)
        self.lsmhe.obfun_mhe.deactivate()

        self._PI = {}  #: Container of the KKT matrix
        self.xreal_W = {}
        self.curr_m_noise = {}   #: Current measurement noise
        self.curr_y_offset = {}  #: Current offset of measurement
        self.curr_u_offset = {}  #: Current offset of the input

        for y in self.y:
            for j in self.y_vars[y]:
                self.curr_m_noise[(y, j)] = 0.0
                self.curr_y_offset[(y, j)] = 0.0
                self.curr_meas[(y, j)] = 0.0

        for u in self.u:
            self.curr_u_offset[u] = 0.0

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


        with open("res_mhe_label_" + self.res_file_suf + ".txt", "w") as f:
            for x in self.x_noisy:
                for j in self.x_vars[x]:
                    jth = (x, j)
                    jth = str(jth)
                    f.write(jth)
                    f.write('\t')
            f.close()

    def init_lsmhe_prep(self, ref, update=True):
        """Initializes the lsmhe in preparation phase
        Args:
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model"""
        self.journalist("I", self._iteration_count, "init_lsmhe_prep", "Preparation phase MHE")
        dum = self.dum_mhe
        if not 'tau_points' in dum.t.get_discretization_info().keys():
            d = TransformationFactory('dae.collocation')
            d.apply_to(dum, nfe=1, ncp=self.ncp_tmhe, scheme='LAGRANGE-RADAU')

        #: Load current solution
        print(dum.t.get_discretization_info())
        print(ref.t.get_discretization_info())
        load_iguess(ref, dum, 0, 0)
        self.load_init_state_gen(dum, src_kind="mod", ref=ref, fe=0)
        #: Patching of finite elements
        t0ncp = t_ij(self.lsmhe.t, 0, self.ncp_tmhe)
        for finite_elem in range(0, self.nfe_tmhe):
            #: Cycle ICS
            for i in self.states:
                pn = i + "_ic"
                p = getattr(dum, pn)
                vs = getattr(dum, i)
                for ks in p.keys():
                    p[ks].value = value(vs[(t0ncp,) + (ks,)])
            if finite_elem == 0:
                for i in self.states:
                    pn = i + "_ic"
                    p = getattr(self.lsmhe, pn)  #: Target
                    vs = getattr(dum, i)  #: Source
                    for ks in p.keys():
                        p[ks].value = value(vs[(t0ncp,) + (ks,)])
            self.patch_meas_mhe(self.PlantSample, fe=finite_elem)
            #: Solve
            self.solve_dyn(dum, o_tee=True)
            #: Patch
            load_iguess(dum, self.lsmhe, 0, finite_elem)
            self.patch_input_mhe("mod", src=dum, fe=finite_elem)

        self.lsmhe.name = "Preparation MHE"   #: Pretty much simulation
        tst = self.solve_dyn(self.lsmhe,
                             o_tee=True,
                             skip_update=False,
                             max_cpu_time=600,
                             jacobian_regularization_value=1e-06,
                             jacobian_regularization_exponent=2.,
                             halt_on_ampl_error=True,
                             output_file="prep_mhe.txt",
                             mu_strategy="adaptive",
                             ma57_pre_alloc=5)

        if tst != 0:
            self.lsmhe.write_nl(name="failed_mhe.nl")
            sys.exit()
        self.lsmhe.name = "LSMHE (Least-Squares MHE)"
        if update:
              cut_off_time = t_ij(self.lsmhe.t, self.lsmhe.nfe_t - 1, 0)
              for i in self.x_noisy:  # only deactivate the relevant noisy-state continuity conditions
                  cp_con = getattr(self.lsmhe, i + "dot_disc_eq")
                  for ii in self.x_vars[i]:
                      for t in self.lsmhe.t:
                          if t >= cut_off_time or t == 0:
                              continue
                          cp_con[(t,) + ii].deactivate()
              self.lsmhe.noisy_cont.activate()  # activate new noisy-state continuity conditions
              self.lsmhe.obfun_dum_mhe_deb.deactivate()
              self.lsmhe.obfun_dum_mhe.activate()
              # self.deact_icc_mhe()
              self.lsmhe.hyk_c_mhe.activate()
              for u in self.u:
                  # need to keep both!!
                  cc = getattr(self.lsmhe, u + "_cdummy")  #: Get the constraint for input
                  con_w = getattr(self.lsmhe, u + "_cdummy_mhe")  #: Get the constraint-noisy
                  cc.deactivate()
                  con_w.activate()
        self.journalist("I", self._iteration_count, "initialize_lsmhe", "Attempting to initialize lsmhe Done")
        
    def preparation_phase_mhe(self, as_strategy=False):
        """Method that prepares the mhe problem; shift; update u and y; initialize last fe"""
        self.shift_mhe()
        self.shift_measurement_input_mhe()
        #: if as_ patch to a different position
        if as_strategy:
            self.patch_meas_mhe(None, use_dict=True, fe=self.nfe_tmhe - 2)
            self.patch_input_mhe(src_kind="dict")
            self.init_step_mhe(patch_pred_y=True)
            self.journalist("I", self._iteration_count, "preparation_phase_mhe", "asMHE: Ready")
        else:
            self.patch_meas_mhe(None, use_dict=True)
            self.patch_input_mhe(src_kind="dict")  #: At this point it doesn't matter if this is the wrong input
            self.init_step_mhe(patch_pred_y=False)  #: Just for initialization purposes
            self.journalist("I", self._iteration_count, "preparation_phase_mhe", "idMHE: Ready")

    def patch_meas_mhe(self, src, **kwargs):
        """Mechanism to assign a value of y0 to the current mhe from the dynamic model
        By default load the measurement to the last finite element of the lsmhe
        Args:
            src(ConcreteModel): The reference model
        Returns:
            meas_dict (dict): A dictionary containing the measurements list by meas_var
        """
        y0dest = getattr(self.lsmhe, "yk0_mhe")  #: Param containing data.
        fe = kwargs.pop("fe", self.nfe_tmhe - 1)
        use_dict = kwargs.pop("use_dict", False)
        #: Override patching
        if use_dict:
            self.journalist("I", self._iteration_count, "patch_meas_mhe", "use_dict")
            for y in self.y:
                for j in self.y_vars[y]:
                    k = self.yk_key[(y, j)]
                    y0dest[fe, k].value = self.curr_meas[(y, j)]
            return dict()

        skip_update = kwargs.pop("skip_update", False)
        noisy = kwargs.pop("noisy", True)
        cp = getattr(src, "ncp_t")
        cpa = cp  #: From the source

        meas_dic = dict.fromkeys(self.y)
        l = []
        tcpa = t_ij(src.t, 0, cpa)
        for y in self.y:
            lm = []
            var = getattr(src, y)
            for j in self.y_vars[y]:
                # self.curr_meas[(y, j)] = value(var[(0, cpa,) + j])
                lm.append(value(var[(tcpa,) + j]))
                l.append(value(var[(tcpa,) + j]))
            meas_dic[y] = lm
            
        # if not skip_update:  #: Update the mhe model
        self.journalist("I", self._iteration_count, "patch_meas_mhe", "Measurement to:" + str(fe))

        for i in self.y:
            for j in self.y_vars[i]:
                k = self.yk_key[(i,) + j]
                #: Adding noise to the mhe measurement
                y0dest[fe, k].value = l[k] + self.curr_m_noise[(i, j)] if noisy else l[k]
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
        for i in range(0, self.nfe_tmhe - 1):
            j = 0
            for x in self.x_noisy:
                x_var = getattr(self.lsmhe, x)
                for k in self.x_vars[x]:
                    x1pvar_val = value(x_var[(i+1, 0), k])
                    x1var_val = value(x_var[(i, self.ncp_tmhe), k])
                    if self.IgnoreProcessNoise:
                        pass
                    else:
                        self.lsmhe.wk_mhe[i, j].set_value(x1pvar_val - x1var_val)
                    j += 1

    def set_covariance_meas(self, cov_dict):
        """Sets covariance(inverse) for the measurements.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(meas_name, j), (meas_name, k), time]
        Returns:
            None
        """
        rtarget = getattr(self.lsmhe, "R_mhe")
        if self.diag_Q_R:
            for i in range(0, self.nfe_tmhe):
                for y in self.y:
                    for jth in self.y_vars[y]:  #: the jth variable
                        v_i = self.yk_key[(y,) + jth]
                        if cov_dict[y] == 0:
                            raise ZeroDivisionError('wrong covariance')
                        rtarget[i, v_i] = 1 / cov_dict[y]
        else:
            sys.exit(1)

        # for key in cov_dict:
        #     vni = key[0]
        #     vnj = key[1]
        #     _t = key[2]
        #
        #     v_i = self.yk_key[vni]
        #     v_j = self.yk_key[vnj]
        #     # try:
        #     if self.diag_Q_R:
        #         rtarget[_t, v_i] = 1 / cov_dict[vni, vnj, _t]
        #     else:
        #         rtarget[_t, v_i, v_j] = cov_dict[vni, vnj, _t]

    def set_covariance_disturb(self, cov_dict):
        """Assign values to the covariance of the disturbance.

        For now we only take diagonal covariance(variance) matrices.
        Args:
            cov_dict (dict): The values of the covariance

        Returns:
            None:
        """
        qtarget = getattr(self.lsmhe, "Q_mhe")
        if self.diag_Q_R:
            for i in range(0, self.nfe_tmhe - 1):
                for x in self.x_noisy:
                    for jth in self.x_vars[x]:  #: the jth variable
                        v_i = self.xkN_key[(x,) + jth]
                        if cov_dict != 0.0:
                            qtarget[i, v_i] = 1 / cov_dict[x]
                        else:
                            raise ZeroDivisionError

        else:
            raise Exception("Not yet implemented [set_covariance_disturb]")


    def set_covariance_u(self, cov_dict):
        """Sets covariance(inverse) for the states.
        Args:
            cov_dict (dict): a dictionary with the following key structure [(state_name, j), (state_name, k), time]
        Returns:
            None
        """
        qtarget = getattr(self.lsmhe, "U_mhe")
        for key in qtarget:
            _t = key[0]
            vni = key[1]
            if cov_dict[vni] == 0.0:
                raise ZeroDivisionError
            qtarget[_t, vni] = 1 / cov_dict[vni]

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
                    if k[0] < self.nfe_tmhe - 1:
                        try:
                            v[k].set_value(v[(k[0] + 1,) + k[1:]])
                        except ValueError:
                            continue

    def shift_measurement_input_mhe(self):
        """Shifts current measurements for the mhe problem"""
        y0 = getattr(self.lsmhe, "yk0_mhe")
        #: Start from the second fe
        for i in range(1, self.nfe_tmhe):
            for j in self.lsmhe.yk0_mhe.keys():
                y0[i-1, j[1:]].value = value(y0[i, j[1:]])
            for u in self.u:
                umhe = getattr(self.lsmhe, u)
                umhe[i-1] = value(umhe[i])
        self.adjust_nu0_mhe()

    def patch_input_mhe(self, src_kind, **kwargs):
        """Loads inputs into the mhe model, by default takes the last finite element"""
        src = kwargs.pop("src", self.PlantSample)
        fe = kwargs.pop("fe", self.nfe_tmhe - 1)
        if src_kind == "mod":
            for u in self.u:
                usrc = getattr(src, u)
                utrg = getattr(self.lsmhe, u)
                utrg[fe].value = value(usrc[0])  #: This has to work
        elif src_kind == "dict":
            for u in self.u:
                utrg = getattr(self.lsmhe, u)
                utrg[fe].value = self.curr_u[u]
        else:
            raise ValueError("Either use mod or dict %s" % src_kind)

    def init_step_mhe(self, patch_pred_y=False, **kwargs):
        """Takes the last state-estimate from the mhe to perform an open-loop simulation
        that initializes the last slice of the mhe horizon. By default the last finite element will be taken as ref.
        Args:
            patch_y (bool): If true, patch the measurements as well"""
        tgt = self.dum_mhe
        src = self.lsmhe
        # fe_t = getattr(src, "fe_t")
        # l = max(fe_t)  #: Maximum fe value
        fe_src = kwargs.pop("fe", self.nfe_tmhe - 1)
        #: Load initial guess to tgt
        for vs in src.component_objects(Var, active=True):
            if vs.getname()[-4:] == "_mhe":
                continue
            vd = getattr(tgt, vs.getname())
            # there are two cases: 1 key 1 elem, several keys 1 element
            vskeys = vs.keys()
            if len(vskeys) == 1:
                #: One key
                for ks in vskeys:
                    for v in vd.keys():
                        v.set_value(value(vs[ks]))
            else:
                k = 0
                for ks in vskeys:
                    if k == 0:
                        if type(ks) != tuple:
                            #: Several keys of 1 element each!!
                            vd[0].set_value(value(vs[vskeys[-1]]))  #: This has got to be true
                            break
                        k += 1
                    kj = ks[2:]
                    if vs.getname() in self.states:  #: States start at 0
                        for j in range(0, self.ncp_tmhe + 1):
                            vd[(0, j) + kj].set_value(value(vs[(fe_src, j) + kj]))
                    else:
                        for j in range(1, self.ncp_tmhe + 1):
                            vd[(0, j) + kj].set_value(value(vs[(fe_src, j) + kj]))
        #: Set values for inputs
        for u in self.u:  #: This should update the inputs
            usrc = getattr(src, u)
            utgt = getattr(tgt, u)
            utgt[0].value = (value(usrc[fe_src]))
        #: Set values for initial states
        for x in self.states:
            pn = x + "_ic"
            p = getattr(tgt, pn)
            vs = getattr(self.lsmhe, x)
            for ks in p.keys():
                p[ks].value = value(vs[(fe_src, self.ncp_tmhe) + (ks,)])
        #: Solve
        test = self.solve_dyn(tgt, o_tee=True, stop_if_nopt=False, max_cpu_time=300,
                            jacobian_regularization_value=1e-04,
                            jacobian_regularization_exponent=2.,
                            halt_on_ampl_error=False,
                            output_file="init_mhe.txt")
        #: Load solution as a guess to lsmhe
        if test != 0:
            self.journalist("I", self._iteration_count, "init_step_mhe", "Failed prediction for next step")
        self.load_iguess_dyndyn(tgt, self.lsmhe, self.nfe_tmhe - 1)



        if patch_pred_y:  #: patch the measurement associated with the solution of the dummy model to the mhe
            self.journalist("I", self._iteration_count, "init_step_mhe", "Prediction for advanced-step.. Ready")
            self.patch_meas_mhe(tgt, noisy=True)
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
        t_prior = t_ij(self.lsmhe.t, 1, 0)
        if set_suffix:
            for key in self.x_noisy:
                var = getattr(self.lsmhe, key)
                for j in self.x_vars[key]:
                    var[(t_prior,) + j].set_suffix_value(self.lsmhe.dof_v, 1)

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
                    var[(self.nfe_tmhe - 1, self.ncp_tmhe) + j].set_suffix_value(self.lsmhe.dof_v, 1)

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
                if v[(1, 0) + j].lb:
                    if v[(1, 0) + j].value - v[(1, 0) + j].lb < 1e-08:
                        active_bound = True
                if v[(1, 0) + j].ub:
                    if v[(1, 0) + j].ub - v[(1, 0) + j].value < 1e-08:
                        active_bound = True
                if active_bound:
                    print("Active bound {:s}, {:d}, value {:f}".format(x, j[0], v[(1, 0) + j].value), file=sys.stderr)
                    v[(1, 0) + j].set_suffix_value(self.lsmhe.dof_v, 0)
                    self.xkN_nexcl.append(0)
                    k += 1
                else:
                    v[(1, 0) + j].set_suffix_value(self.lsmhe.dof_v, 1)
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
        self.lsmhe.obfun_mhe.set_value(self.lsmhe.Arrival_e_mhe + self.lsmhe.R_e_mhe + self.lsmhe.Q_e_mhe + self.lsmhe.U_e_mhe)

        if self.lsmhe.obfun_dum_mhe.active:
            self.lsmhe.obfun_dum_mhe.deactivate()
        if not self.lsmhe.obfun_mhe.active:
            self.lsmhe.obfun_mhe.activate()
        if not self.lsmhe.hyk_c_mhe.active:
            self.lsmhe.hyk_c_mhe.activate()

    def load_covariance_prior(self):
        """Computes the reduced-hessian (inverse of the prior-covariance)
        Reads the result_hessian.txt file that contains the covariance information"""
        self.journalist("I", self._iteration_count, "load_covariance_prior", "K_AUG w red_hess")
        self.k_aug.options["compute_inv"] = ""
        if hasattr(self.lsmhe, "f_timestamp"):
            self.lsmhe.f_timestamp.clear()
        else:
            self.lsmhe.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        self.create_rh_sfx()
        try:
            self.k_aug.solve(self.lsmhe, tee=True)
        except ApplicationError:
            self.journalist("E", self._iteration_count, "load_covariance_prior", "K_AUG failed; no covariance info was loaded")
            self.lsmhe.write_nl(name="failed_covariance.nl")
            return 1
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
                    if vj[(1, 0) + j].get_suffix_value(self.lsmhe.dof_v) == 0:
                        #: This state is at its bound, skip
                        continue
                    for k in self.x_vars[key_k]:
                        if vk[(1, 0) + k].get_suffix_value(self.lsmhe.dof_v) == 0:
                            #: This state is at its bound, skip
                            print("vj {:s} {:d} .sfx={:d}, vk {:s} {:d}.sfx={:d}"
                                  .format(key_j, j[0], vj[(1, 0) + j].get_suffix_value(self.lsmhe.dof_v),
                                          key_k, k[0], vk[(1, 0) + k].get_suffix_value(self.lsmhe.dof_v),))
                            continue
                        row = vj[(1, 0) + j].get_suffix_value(self.lsmhe.rh_name)
                        col = vk[(1, 0) + k].get_suffix_value(self.lsmhe.rh_name)
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
                z0dest[z0] = value(var[(1, 0,) + j])

    def prior_phase(self):
        """Encapsulates all the prior-state related issues, like collection, covariance computation and update"""
        # Prior-Covariance stuff
        self.check_active_bound_noisy()
        self.load_covariance_prior()
        self.set_state_covariance()
        self.regen_objective_fun()
        # Update prior-state
        self.set_prior_state_from_prior_mhe()
        pass

    def update_noise_meas(self, cov_dict):
        self.journalist("I", self._iteration_count, "introduce_noise_meas", "Noise introduction")
        # f = open("m0.txt", "w")
        # f1 = open("m1.txt", "w")
        for y in self.y:
            # vy = getattr(mod,  y)
            # vy.display(ostream=f)
            for j in self.y_vars[y]:
                # vv = value(vy[(1, self.ncp_t) + j])
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
        self.journalist("I", self._iteration_count, "print_r_mhe", "Results at" + os.getcwd())
        self.journalist("I", self._iteration_count, "print_r_mhe", "Results suffix " + self.res_file_suf)
        for x in self.x_noisy:
            elist = []
            rlist = []
            xe = getattr(self.lsmhe, x)
            xr = getattr(self.PlantSample, x)
            for j in self.x_vars[x]:
                elist.append(value(xe[(self.nfe_tmhe - 1, self.ncp_tmhe) + j]))
                rlist.append(value(xr[(0, self.ncp_t) + j]))
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
            yr = getattr(self.PlantSample, y)
            for j in self.y_vars[y]:
                elist.append(value(ye[(self.nfe_tmhe-1, self.ncp_tmhe) + j]))
                rlist.append(value(yr[(0, self.ncp_t) + j]))
                nlist.append(self.curr_m_noise[(y, j)])
                yklst.append(value(self.lsmhe.yk0_mhe[self.nfe_tmhe-1, self.yk_key[(y, j)]]))
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

        with open("res_mhe_uoffset_" + self.res_file_suf + ".txt", "a") as f:
            for u in self.u:
                cu = self.curr_u_offset[u]
                f.write(str(cu))
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

    def compute_y_offset(self, noisy=False, uoff_update=True):
        """Gets the offset of prediction and real measurement for asMHE"""
        mhe_y = getattr(self.lsmhe, "yk0_mhe")
        for y in self.y:
            plant_y = getattr(self.PlantSample, y)
            for j in self.y_vars[y]:
                k = self.yk_key[(y, j)]
                mhe_yval = value(mhe_y[self.nfe_tmhe-1, k])
                plant_yval = value(plant_y[(0, self.ncp_t) + j])
                y_noise = self.curr_m_noise[(y, j)] if noisy else 0.0
                self.curr_y_offset[(y, j)] = plant_yval - mhe_yval
        if uoff_update:
            for u in self.u:
                mhe_u = getattr(self.lsmhe, u)
                # pla_u = getattr(self.PlantSample, u)
                self.curr_u_offset[u] = self.curr_u[u] - value(mhe_u[self.nfe_tmhe-1])
                print(self.curr_u_offset[u])


    def sens_dot_mhe(self):
        """Updates suffixes, solves using the dot_driver"""
        self.journalist("I", self._iteration_count, "sens_dot_mhe", "Set-up")

        if hasattr(self.lsmhe, "npdp"):
            self.lsmhe.npdp.clear()
        else:
            self.lsmhe.npdp = Suffix(direction=Suffix.EXPORT)
        self.create_sens_suffix_mhe()
        for y in self.y:
            for j in self.y_vars[y]:
                k = self.yk_key[(y, j)]
                self.lsmhe.hyk_c_mhe[self.nfe_tmhe-1, k].set_suffix_value(self.lsmhe.npdp, self.curr_y_offset[(y, j)])

        #: Added this bit to account for the case when the last input does not match the one used
        #: For the prediction for the next measurement of the MHE problem.
        for u in self.u:
            con_w = getattr(self.lsmhe, "w_" + u + "c_mhe")
            con_w[self.nfe_tmhe-1].set_suffix_value(self.lsmhe.npdp, self.curr_u_offset[u])


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

        self.journalist("I", self._iteration_count, "sens_dot_mhe", self.lsmhe.name)

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
        self.journalist("I", self._iteration_count, "sens_k_aug_mhe", "k_aug sensitivity")
        self.lsmhe.ipopt_zL_in.update(self.lsmhe.ipopt_zL_out)
        self.lsmhe.ipopt_zU_in.update(self.lsmhe.ipopt_zU_out)
        self.journalist("I", self._iteration_count, "sens_k_aug_mhe", self.lsmhe.name)

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
            self.journalist("I", self._iteration_count, "update_state_mhe", "offset ready for asnmpcmhe")
            for x in self.states:
                xvar = getattr(self.lsmhe, x)
                x0 = getattr(self.olnmpc, x + "_ic")
                for j in self.state_vars[x]:
                    #: Compute the offset between current loaded value for the nmpc and mhe
                    self.curr_state_offset[(x, j)] = value(xvar[self.nfe_tmhe-1, self.ncp_tmhe, j]) - value(x0[j])
                    # print("state !", self.curr_state_offset[(x, j)])

        for x in self.states:
            xvar = getattr(self.lsmhe, x)
            for j in self.state_vars[x]:
                self.curr_estate[(x, j)] = value(xvar[self.nfe_tmhe-1, self.ncp_tmhe, j])

    def update_measurement(self):
        """Update the current dictionary from the plant"""

        cp = getattr(self.PlantSample, "cp_t")
        cpa = max(cp)  #: From the source
        for y in self.y:
            var = getattr(self.PlantSample, y)
            for j in self.y_vars[y]:
                self.curr_meas[(y, j)] = value(var[(0, cpa,) + j])


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
