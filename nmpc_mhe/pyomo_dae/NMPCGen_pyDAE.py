# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, Set, Constraint, Expression, Param, Suffix, TransformationFactory
from pyomo.core.base.numvalue import value
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from nmpc_mhe.pyomo_dae.DynGen_pyDAE import DynGen_DAE
from nmpc_mhe.aux.utils import t_ij
from nmpc_mhe.aux.utils import fe_compute, load_iguess, augment_model, augment_steady, aug_discretization, create_bounds
from nmpc_mhe.aux.utils import clone_the_model, get_lu_KKT, get_jacobian_k_aug, dlqr, abline, solve_bounded_line
import sys
import os
import time
# import control
import numpy as np
import matplotlib.pyplot as plt
from pyomo.dae import DerivativeVar
from copy import deepcopy
from scipy.sparse import csc_matrix

__author__ = "David Thierry @dthierry, Kuan-Han Lin @kuanhanl" #: March 2018, Jul 2020

"""This version does not necesarily have the same time horizon/discretization as the MHE"""


class NmpcGen_DAE(DynGen_DAE):
    def __init__(self, d_mod, hi_t, states, controls, **kwargs):
        DynGen_DAE.__init__(self, d_mod, hi_t, states, controls, **kwargs)
        self.int_file_nmpc_suf = int(time.time())+1

        self.ref_state = kwargs.pop("ref_state", None)
        self.u_bounds = kwargs.pop("u_bounds", None)

        # One can specify different discretization lenght
        self.nfe_tnmpc = kwargs.pop('nfe_tnmpc', self.nfe_t)  #: Specific number of finite elements
        self.ncp_tnmpc = kwargs.pop('ncp_tnmpc', self.ncp_t)  #: Specific number of collocation points

        # We need a list of tuples that contain the bounds of u
        self.olnmpc = object()
        self.curr_soi = {}  #: Values that we would like to keep track of
        self.curr_sp = {}  #: Values that we would like to keep track (from SteadyRef2)
        self.curr_off_soi = {}
        self.curr_ur = dict.fromkeys(self.u, 0.0)  #: Controls that we would like to keep track of(from SteadyRef2)
        if self.ref_state:
            for k in self.ref_state.keys():
                self.curr_soi[k] = 0.0
                self.curr_sp[k] = 0.0
        else:
            self.journalist('W', self._iteration_count, "Initializing NMPC", "No ref_state has been specified")
        if not self.u_bounds:
            self.journalist('W', self._iteration_count, "Initializing NMPC", "No bounds dictionary has been specified")

        self.soi_dict = {}  #: State-of-interest.
        self.sp_dict = {}  #: Set-point.
        self.u_dict = dict.fromkeys(self.u, [])
        f = open("timings_nmpc_kaug_sens.txt", "a")
        f.write('\n' + '-' * 30 + '\n')
        f.write(str(self.int_file_nmpc_suf))
        f.write('\n')
        f.close()

        f = open("timings_nmpc_dot.txt", "a")
        f.write('\n' + '-' * 30 + '\n')
        f.write(str(self.int_file_nmpc_suf))
        f.write('\n')
        f.close()

        # self.res_file_name = "res_nmpc_" + str(int(time.time())) + ".txt"
        
        #objects to save full profile of setpoint
        self.profile_state_target = {}
        self.profile_u_target = {}
        self.profile_target = False
        
        #objects for amsnmpc
        self.num_flatten_var = None #number of flatten variables
        self.amsnmpc_Ns = kwargs.pop('Ns_amsnmpc', None) #Ns for amsnmpc
        if self.amsnmpc_Ns is not None:
            self.Pred_amsnmpc = None #state predictor for amsnmpc
            self.record_suffix_u = {}
            self.record_suffix_x = {}
            self.u_within_Ns_store = {} #store u within Ns steps
            self.z_within_Ns_store = {} #store state within Ns steps
            self.u_within_Ns_recent = {} #store u within Ns steps
            self.z_within_Ns_recent = {} #store state within Ns steps
            self.u_for_pred = {} #u for state predictor
            for i in range(self.amsnmpc_Ns):
                self.u_within_Ns_store[i] = {}
                self.z_within_Ns_store[i] = {}
                self.u_within_Ns_recent[i] = {} #store u within Ns steps
                self.z_within_Ns_recent[i] = {} #store state within Ns steps
                self.u_for_pred[i] = {}
            self.amsnmpc_kkt = None #kkt matrix from k_aug
            self.amsnmpc_kkt_size = None #size of kkt matrix
            self.varinfo = None #data from varorder.txt
            self.coninfo = None #data from conorder.txt
            self.var_num = None #number of vars
            self.con_num = None #number of constraints
            #Because there is a step mismatch when we get new ds&dj and use ds&dj,
            #I use two objects to store them. KH.L
            self.amsnmpc_ds_int_store = None #store ds_int
            self.amsnmpc_dj_store = {} #store dj
            self.amsnmpc_ds_int_recent = None #ds_int is about to be used
            self.amsnmpc_dj_recent = {} #dj is about to be used
            self.u_mod = None #current u after update with sensitivity
            
        self.true_u_name = [] #ture name of controls
        self.der_var = [] #derivative variables
        self.diff_der_var = {} #connection between diff_var and der_var
        self.other_var_list = [] #other variable (alge_var)
        self.other_vars = {} 
        self.diff_equ = [] #differential equaitons
        self.alge_equ = [] #algebric equations
        self.diff_var_con = {} #connection between diff_var and its constraints(DE)
        self.tp_vars_suffix = dict()
        self.tp_cons_suffix = dict()
        self.n_controls = None #number of controls
        self.n_flatten_othervars = None #number of flatten other vars
        self.n_DE = None #number of DE
        self.n_AE = None #number of AE
        self.tp_state_ss = {} #steady states
        self.tp_u_ss = {} #steady controls
        self.tp_simulate = None #pyomo model to simulate many ics of x
        self.tp_model = None#pyomo model to get steady states and controls
        self.tp_cf = None #radius of terminal region
        self.tp_area = None #area of terminal region
        self.tp_P = None #cost-to-go P of LQR >> matrix in terminal cost
        self.tp_exist = False
        

    def create_nmpc(self, **kwargs):
        """
        Creates the nmpc model for the optimization.
        Args:
            **kwargs:
        """
        kwargs.pop("newnfe", self.nfe_tnmpc)
        kwargs.pop("newncp", self.ncp_tnmpc)
        self.journalist('W', self._iteration_count, "Initializing NMPC",
                        "With {:d} fe and {:d} cp".format(self.nfe_tnmpc, self.ncp_tnmpc))
        _tnmpc = self.hi_t * self.nfe_tnmpc
        self.olnmpc = clone_the_model(self.d_mod)
        self.olnmpc.name = "olnmpc (Open-Loop NMPC)"

        augment_model(self.olnmpc, self.nfe_tnmpc, self.ncp_tnmpc, new_timeset_bounds=(0, _tnmpc))
        aug_discretization(self.olnmpc, self.nfe_tnmpc, self.ncp_tnmpc)

        self.olnmpc.fe_t = Set(initialize=[i for i in range(0, self.nfe_tnmpc)])  #: Set for the NMPC stuff

        tfe_dic = dict()
        for t in self.olnmpc.t:
            if t == max(self.olnmpc.t):
                tfe_dic[t] = fe_compute(self.olnmpc.t, t-1)
            else:
                tfe_dic[t] = fe_compute(self.olnmpc.t, t)
        #: u vars and u constraints creation
        for u in self.u:  #: u only has one index
            cv = getattr(self.olnmpc, u)  #: Get the param
            t_u = [t_ij(self.olnmpc.t, i, 0) for i in range(0, self.olnmpc.nfe_t)]
            c_val = [value(cv[t_u[i]]) for i in self.olnmpc.fe_t]  #: Current value
            # self.u1_cdummy = Constraint(self.t, rule=lambda m, i: m.Tjinb[i] == self.u1[i])
            dumm_eq = getattr(self.olnmpc, u + '_cdummy')
            dexpr = dumm_eq[0].expr.args[0]
            control_var = getattr(self.olnmpc, dexpr.parent_component().name)
            if isinstance(control_var, Var): #: all good
                pass
            else:
                raise ValueError  #: Some exception here

            self.olnmpc.del_component(cv)  #: Delete the dummy_param
            self.olnmpc.del_component(dumm_eq)  #: Delete the dummy_constraint
            self.olnmpc.add_component(u, Var(self.olnmpc.fe_t, initialize=lambda m, i: c_val[i]))
            cv = getattr(self.olnmpc, u)  #: Get the new variable
            for k in cv.keys():
                cv[k].setlb(self.u_bounds[u][0])
                cv[k].setub(self.u_bounds[u][1])

            self.olnmpc.add_component(u + '_cdummy', Constraint(self.olnmpc.t))
            dumm_eq = getattr(self.olnmpc, u + '_cdummy')
            dumm_eq.rule = lambda m, i: cv[tfe_dic[i]] == control_var[i]
            dumm_eq.reconstruct()

        #: Dictionary of the states for a particular time point i
        self.xmpc_l = {}
        #: Dictionary of the position for a state in the dictionary
        self.xmpc_key = {}
        #:
        self.xmpc_l[0] = []
        #: First build the name dictionary
        k = 0
        for x in self.states:
            n_s = getattr(self.olnmpc, x)  #: State
            t = t_ij(self.olnmpc.t, 0, self.ncp_t)
            for j in self.state_vars[x]:
                self.xmpc_l[0].append(n_s[(t,) + j])
                self.xmpc_key[(x, j)] = k
                k += 1
        #: Iterate over the rest
        for t in range(1, self.nfe_tnmpc):
            time = t_ij(self.olnmpc.t, t, self.ncp_tnmpc)
            self.xmpc_l[t] = []
            for x in self.states:
                n_s = getattr(self.olnmpc, x)  #: State
                for j in self.state_vars[x]:
                    self.xmpc_l[t].append(n_s[(time,) + j])
        #: A set with the length of flattened states
        self.olnmpc.xmpcS_nmpc = Set(initialize=[i for i in range(0, len(self.xmpc_l[0]))])
        #: Create set of noisy_states
        self.olnmpc.xmpc_ref_nmpc = Param(self.olnmpc.xmpcS_nmpc, initialize=0.0, mutable=True)  #: Ref-state
        self.olnmpc.xmpc_ref_nmpc2 = Param(self.olnmpc.fe_t, self.olnmpc.xmpcS_nmpc, initialize=0.0, mutable=True)
        self.olnmpc.Q_nmpc = Param(self.olnmpc.xmpcS_nmpc, initialize=1, mutable=True)  #: Control-weight
        # (diagonal Matrices)

        self.olnmpc.Q_w_nmpc = Param(self.olnmpc.fe_t, initialize=1e-04, mutable=True)
        self.olnmpc.R_w_nmpc = Param(self.olnmpc.fe_t, initialize=1e+02, mutable=True)
        #: Build the xT*Q*x part
        self.olnmpc.xQ_expr_nmpc = Expression(expr=sum(
            sum(self.olnmpc.Q_w_nmpc[fe] *
                self.olnmpc.Q_nmpc[k] * (self.xmpc_l[fe][k] - self.olnmpc.xmpc_ref_nmpc[k])**2
                for k in self.olnmpc.xmpcS_nmpc)
                for fe in range(0, self.nfe_tnmpc)))

        self.olnmpc.xQ_expr_nmpc2 = Expression(expr=sum(
            sum(self.olnmpc.Q_w_nmpc[fe] *
                self.olnmpc.Q_nmpc[k] * (self.xmpc_l[fe][k] - self.olnmpc.xmpc_ref_nmpc2[fe, k])**2
                for k in self.olnmpc.xmpcS_nmpc)
                for fe in range(0, self.nfe_tnmpc)))

        #: Build the control list
        self.umpc_l = {}
        for t in range(0, self.nfe_tnmpc):
            self.umpc_l[t] = []
            for u in self.u:
                uvar = getattr(self.olnmpc, u)
                self.umpc_l[t].append(uvar[t])
        #: Create set of u
        self.olnmpc.umpcS_nmpc = Set(initialize=[i for i in range(0, len(self.umpc_l[0]))])
        #: ref u
        self.olnmpc.umpc_ref_nmpc = Param(self.olnmpc.umpcS_nmpc, initialize=0.0, mutable=True)
        self.olnmpc.umpc_ref_nmpc2 = Param(self.olnmpc.fe_t, self.olnmpc.umpcS_nmpc, initialize=0.0, mutable=True)
        self.olnmpc.R_nmpc = Param(self.olnmpc.umpcS_nmpc, initialize=1, mutable=True)  #: Control-weight
        #: Build the uT * R * u expression
        self.olnmpc.xR_expr_nmpc = Expression(expr=sum(
            sum(self.olnmpc.R_w_nmpc[fe] *
                self.olnmpc.R_nmpc[k] * (self.umpc_l[fe][k] - self.olnmpc.umpc_ref_nmpc[k]) ** 2 for k in
                self.olnmpc.umpcS_nmpc)
            for fe in range(0, self.nfe_tnmpc)))
        
        self.olnmpc.xR_expr_nmpc2 = Expression(expr=sum(
            sum(self.olnmpc.R_w_nmpc[fe] *
                self.olnmpc.R_nmpc[k] * (self.umpc_l[fe][k] - self.olnmpc.umpc_ref_nmpc2[fe, k]) ** 2 for k in
                self.olnmpc.umpcS_nmpc)
            for fe in range(0, self.nfe_tnmpc)))
        
        self.olnmpc.objfun_nmpc = Objective(expr=self.olnmpc.xQ_expr_nmpc + self.olnmpc.xR_expr_nmpc)
        self.olnmpc.objfun_nmpc2 = Objective(expr=self.olnmpc.xQ_expr_nmpc2 + self.olnmpc.xR_expr_nmpc2)
        self.olnmpc.objfun_nmpc2.deactivate()

        #for amsnmpc
        count = 0
        for x in self.states:
            for j in self.state_vars[x]:
                count += 1
        self.num_flatten_var = count
        # print(self.num_flatten_var)
        
    def initialize_olnmpc(self, ref, src_kind, **kwargs):
        # The reference is always a model
        # The source of the state might be different
        # The source might be a predicted-state from forward simulation
        """Initializes the olnmpc from a reference state, loads the state into the olnmpc
        Args
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model
            fe (int): Source fe
            src_kind (str): the kind of source
        Returns:
            """
        fe = kwargs.pop("fe", 0)
        self.journalist("I", self._iteration_count, "initialize_olnmpc", "Attempting to initialize olnmpc")
        self.journalist("I", self._iteration_count, "initialize_olnmpc", "src_kind=" + src_kind)
        # self.load_init_state_nmpc(src_kind="mod", ref=ref, fe=1, cp=self.ncp_t)

        if src_kind == "real":
            self.load_init_state_nmpc(src_kind="dict", state_dict="real")
        elif src_kind == "estimated":
            self.load_init_state_nmpc(src_kind="dict", state_dict="estimated")
        elif src_kind == "predicted":  #: just as-nmpc
            self.load_init_state_nmpc(src_kind="dict", state_dict="predicted")
        else:
            self.journalist("E", self._iteration_count, "initialize_olnmpc", "SRC not given")
            raise ValueError("Unexpected src_kind %s" % src_kind)

        dum = clone_the_model(self.d_mod) #(1, self.ncp_tnmpc, _t=self.hi_t)
        augment_model(dum, 1, self.ncp_tnmpc, new_timeset_bounds=(0, self.hi_t))
        aug_discretization(dum, 1, self.ncp_tnmpc)
        create_bounds(dum, bounds=self.var_bounds)
        #: Load current solution
        # self.load_iguess_single(ref, dum, 0, 0)
        load_iguess(ref, dum, 0, 0)

        # self.load_iguess_dyndyn(ref, dum, fe, fe_src="s")  #: This is supossed to work
        for u in self.u:  #: Initialize controls dummy model
            cv_dum = getattr(dum, u)
            cv_ref = getattr(ref, u)
            for i in cv_dum.keys():
                cv_dum[i].value = value(cv_ref[fe])
        #: Patching of finite elements
        k_notopt = 0
        for finite_elem in range(0, self.nfe_tnmpc):
            dum.name = "Dummy I " + str(finite_elem)
            if finite_elem == 0:
                if src_kind == "predicted":
                    self.load_init_state_gen(dum, src_kind="dict", state_dict="predicted")
                elif src_kind == "estimated":
                    self.load_init_state_gen(dum, src_kind="dict", state_dict="estimated")
                elif src_kind == "real":
                    self.load_init_state_gen(dum, src_kind="dict", state_dict="real")
                else:
                    self.journalist("E", self._iteration_count, "initialize_olnmpc", "SRC not given")
                    sys.exit()
            else:
                self.load_init_state_gen(dum, src_kind="mod", ref=dum, fe=0)

            tst = self.solve_dyn(dum,
                               o_tee=False,
                               tol=1e-04,
                               iter_max=1000,
                               max_cpu_time=60,
                               stop_if_nopt=False,
                               output_file="dummy_ip.log")
            if tst != 0:
                self.journalist("W", self._iteration_count, "initialize_olnmpc", "non-optimal dummy")
                tst1 = self.solve_dyn(dum,
                             o_tee=True,
                             tol=1e-03,
                             iter_max=1000,
                             stop_if_nopt=False,
                             jacobian_regularization_value=1e-04,
                             ma57_small_pivot_flag=1,
                             ma57_pre_alloc=5,
                             linear_scaling_on_demand="yes", ma57_pivtol=1e-12,
                             output_file="dummy_ip.log")
                if tst1 != 0:
                    # sys.exit()
                    print("Too bad :(", file=sys.stderr)
                k_notopt += 1
            #: Patch
            # self.load_iguess_dyndyn(dum, self.olnmpc, finite_elem)
            load_iguess(dum, self.olnmpc, 0, finite_elem)

            for u in self.u:
                cv_nmpc = getattr(self.olnmpc, u)  #: set controls for open-loop nmpc
                cv_dum = getattr(dum, u)
                # works only for fe_t index
                cv_nmpc[finite_elem].set_value(value(cv_dum[0]))
        self.journalist("I", self._iteration_count, "initialize_olnmpc", "Done, k_notopt " + str(k_notopt))

    def preparation_phase_nmpc(self, as_strategy=False, make_prediction=False, plant_state=False, ams_strategy=False):
        # type: (bool, bool, bool, bool) -> bool
        """Initialization and loading initial state of the NMPC problem.

        Args:
            as_strategy (bool): True if as-NMPC is activated.
            make_prediction (bool): True if as-NMPC is desired (prediction of state).
            plant_state (bool): Override options to use plant states.
            ams_strategy(bool): True is ams-NMPC is activated.

        Returns:

        """
        if as_strategy and ams_strategy:
            raise RuntimeError("as_nmpc and ams_nmpc cannot be activated together.")
        
        if ams_strategy:
            if plant_state:
                self.predictor_amsNMPC(src="real")
            else:
                self.predictor_amsNMPC(src="estimated")
            
            self.initialize_olnmpc(self.SteadyRef, "predicted")
            self.load_init_state_nmpc(src_kind="state_dict", state_dict="predicted")
            return
            
        if plant_state:
            #: use the plant state instead
            #: Not yet implemented
            self.initialize_olnmpc(self.PlantSample, "real")
            self.load_init_state_nmpc(src_kind="state_dict", state_dict="real")
            return
        if as_strategy:
            if make_prediction:
                self.update_state_predicted(src="estimated")
                self.initialize_olnmpc(self.PlantPred, "predicted")
                self.load_init_state_nmpc(src_kind="state_dict", state_dict="predicted")
            else:
                self.initialize_olnmpc(self.PlantSample, "estimated")
                self.load_init_state_nmpc(src_kind="state_dict", state_dict="estimated")
        else:
            #: Similar to as_strategy w/o prediction just to prevent ambiguity
            #: WHY IS THIS HERE
            self.initialize_olnmpc(self.PlantSample, "estimated")
            self.load_init_state_nmpc(src_kind="state_dict", state_dict="estimated")

    def load_init_state_nmpc(self, src_kind="dict", **kwargs):
        """Loads ref state for set-point
        Args:
            src_kind (str): the kind of source
            **kwargs: Arbitrary keyword arguments.
        Returns:
            None
        Keyword Args:
            src_kind (str) : if == mod use reference model, otw use the internal dictionary
            ref (pyomo.core.base.PyomoModel.ConcreteModel): The reference model (default PlantPred)
            fe (int): The required finite element
            cp (int): The required collocation point
        """
        # src_kind = kwargs.pop("src_kind", "mod")
        self.journalist("I", self._iteration_count, "load_init_state_nmpc", "Load State to nmpc src_kind=" + src_kind)
        ref = kwargs.pop("ref", None)
        fe = kwargs.pop("fe", self.nfe_tnmpc)
        cp = kwargs.pop("cp", self.ncp_tnmpc)
        if src_kind == "mod":
            if not ref:
                self.journalist("W", self._iteration_count, "load_init_state_nmpc", "No model was given")
                self.journalist("W", self._iteration_count, "load_init_state_nmpc", "No update on state performed")
                sys.exit()
            # for x in self.states:
            #     xic = getattr(self.olnmpc, x + "_ic")
            #     xvar = getattr(self.olnmpc, x)
            #     xsrc = getattr(ref, x)
            #     for j in self.state_vars[x]:
            #         xic[j].value = value(xsrc[(fe, cp) + j])
            #         xvar[(0, 0) + j].set_value(value(xsrc[(fe, cp) + j]))  #: Need fixing
            pass
        else:
            state_dict = kwargs.pop("state_dict", None)
            if state_dict == "real":  #: Load from the real state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_rstate[(x, j)]
                        xvar[(0,) + j].set_value(self.curr_rstate[(x, j)])
            elif state_dict == "estimated":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_estate[(x, j)]
                        xvar[(0,) + j].set_value(self.curr_estate[(x, j)])
            elif state_dict == "predicted":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_pstate[(x, j)]
                        xvar[(0,) + j].set_value(self.curr_pstate[(x, j)])
            else:
                self.journalist("W", self._iteration_count, "load_init_state_nmpc", "No dict w/state was specified")
                self.journalist("W", self._iteration_count, "load_init_state_nmpc", "No update on state performed")
                raise ValueError("Unexpected state_dict %s" % state_dict)

    def compute_QR_nmpc(self, src="plant", n=-1, **kwargs):
        """Using the current state & control targets, computes the Qk and Rk matrices (diagonal)
        Strategy: take the inverse of the absolute difference between reference and current state such that every
        offset in the objective is normalized or at least dimensionless
        Args:
            src (str): The source of the update (default mhe) (mhe or plant)
            n (int): The exponent of the weight"""

        define_by_user = kwargs.pop("define_by_user", False)
        
        if define_by_user:
            dbu_Q_nmpc = kwargs.pop("Q_nmpc", None)
            dbu_R_nmpc = kwargs.pop("R_nmpc", None)
            self.update_targets_nmpc()
            for x in self.states:
                for j in self.state_vars[x]:
                    k = self.xmpc_key[(x,j)]
                    if isinstance(dbu_Q_nmpc, dict):
                        self.olnmpc.Q_nmpc[k].value = dbu_Q_nmpc[(x,j)]
                    if not isinstance(dbu_Q_nmpc, dict):
                        self.olnmpc.Q_nmpc[k].value = dbu_Q_nmpc
                    self.olnmpc.xmpc_ref_nmpc[k].value = self.curr_state_target[(x, j)]
                        
            k = 0
            for u in self.u:
                if isinstance(dbu_R_nmpc, dict):
                    self.olnmpc.R_nmpc[k].value = dbu_R_nmpc[u]
                if not isinstance(dbu_R_nmpc, dict):
                    self.olnmpc.R_nmpc[k].value = dbu_R_nmpc
                self.olnmpc.umpc_ref_nmpc[k].value = self.curr_u_target[u]
                k += 1
        else:
            check_values = kwargs.pop("check_values", False)
            # if check_values: #I think it's a bug if we have this if statement. KH.L
            max_w_value = kwargs.pop("max_w_value", 1e+06)
            min_w_value = kwargs.pop("min_w_value", 0.0)
            self.update_targets_nmpc()
            if src == "mhe":
                for x in self.states:
                    for j in self.state_vars[x]:
                        k = self.xmpc_key[(x, j)]
                        temp = abs(self.curr_estate[(x, j)] - self.curr_state_target[(x, j)])
                        if temp > 1e-08:
                            self.olnmpc.Q_nmpc[k].value = temp**n
                        else:
                            self.olnmpc.Q_nmpc[k].value = max_w_value
                        self.olnmpc.xmpc_ref_nmpc[k].value = self.curr_state_target[(x, j)]
            elif src == "plant":
                for x in self.states:
                    for j in self.state_vars[x]:
                        k = self.xmpc_key[(x, j)]
                        # self.olnmpc.Q_nmpc[k].value = abs(self.curr_rstate[(x, j)] - self.curr_state_target[(x, j)])**n
                        temp = abs(self.curr_rstate[(x, j)] - self.curr_state_target[(x, j)])
                        if temp > 1e-08:
                            self.olnmpc.Q_nmpc[k].value = temp ** n
                        else:
                            self.olnmpc.Q_nmpc[k].value = max_w_value
                        self.olnmpc.xmpc_ref_nmpc[k].value = self.curr_state_target[(x, j)]
            k = 0
            for u in self.u:
                self.olnmpc.R_nmpc[k].value = abs(self.curr_u[u] - self.curr_u_target[u])**n
                self.olnmpc.umpc_ref_nmpc[k].value = self.curr_u_target[u]
                k += 1
            if check_values:
                for k in self.olnmpc.xmpcS_nmpc:
                    if value(self.olnmpc.Q_nmpc[k]) < min_w_value:
                        self.olnmpc.Q_nmpc[k].value = min_w_value
                    if value(self.olnmpc.Q_nmpc[k]) > max_w_value:
                        self.olnmpc.Q_nmpc[k].value = max_w_value
                k = 0
                for u in self.u:
                    if value(self.olnmpc.R_nmpc[k]) < min_w_value:
                        self.olnmpc.R_nmpc[k].value = min_w_value
                    if value(self.olnmpc.R_nmpc[k]) > max_w_value:
                        self.olnmpc.R_nmpc[k].value = max_w_value
                    k += 1
                              
        target_step = kwargs.pop("target_step", None)
        if self.profile_target:
            self.olnmpc.objfun_nmpc.deactivate()
            self.olnmpc.objfun_nmpc2.activate()
            
            if target_step is None:
                raise RuntimeError("Please give the target step to load the corresponding setpoint.")
                
            sorted_key = list(self.profile_state_target.keys())
            sorted_key.sort()
            
            for i in range(self.nfe_tnmpc):
                goal = i + target_step
                for s in sorted_key:
                    if goal >= s:
                        cor = s # goal is in group "cor", which is s
                for x in self.states:
                    for j in self.state_vars[x]:
                        k = self.xmpc_key[(x, j)]
                        self.olnmpc.xmpc_ref_nmpc2[i, k].value = self.profile_state_target[cor][(x,j)]
                k = 0
                for u in self.u:
                    self.olnmpc.umpc_ref_nmpc2[i, k].value = self.profile_u_target[cor][u]
                    k += 1
                    
    def new_weights_olnmpc(self, state_weight, control_weight):
        """Change the weights associated with the control objective function"""

        if isinstance(state_weight, dict):
            for fe in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[fe].value = state_weight[fe]
        else:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[fe].value = state_weight

        if isinstance(control_weight, dict):
            for fe in self.olnmpc.fe_t:
                self.olnmpc.R_w_nmpc[fe].value = control_weight[fe]
        else:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.R_w_nmpc[fe].value = control_weight

    def create_suffixes_nmpc(self):
        """Creates the required suffixes for the advanced-step olnmpc problem (reduced-sens)
        """
        if hasattr(self.olnmpc, "npdp"):
            pass
        else:
            self.olnmpc.npdp = Suffix(direction=Suffix.EXPORT)
        if hasattr(self.olnmpc, "dof_v"):
            pass
        else:
            self.olnmpc.dof_v = Suffix(direction=Suffix.EXPORT)

        for u in self.u:
            uv = getattr(self.olnmpc, u)
            uv[0].set_suffix_value(self.olnmpc.dof_v, 1)

    def sens_dot_nmpc(self):
        self.journalist("I", self._iteration_count, "sens_dot_nmpc", "Set-up")

        if hasattr(self.olnmpc, "npdp"):
            self.olnmpc.npdp.clear()
        else:
            self.olnmpc.npdp = Suffix(direction=Suffix.EXPORT)
        with open("npdp_con.txt", "a") as f:
            for x in self.states:
                con_name = x + "_icc"
                con_ = getattr(self.olnmpc, con_name)
                for j in self.state_vars[x]:
                    con_[j].set_suffix_value(self.olnmpc.npdp, self.curr_state_offset[(x, j)])

                con_.display(ostream=f)
        with open("npdp_vals.txt", "a") as f:
            self.olnmpc.npdp.display(ostream=f)
            f.close()

        if hasattr(self.olnmpc, "f_timestamp"):
            self.olnmpc.f_timestamp.clear()
        else:
            self.olnmpc.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        self.olnmpc.set_suffix_value(self.olnmpc.f_timestamp, self.int_file_nmpc_suf)

        self.olnmpc.f_timestamp.display(ostream=sys.stderr)

        self.journalist("I", self._iteration_count, "sens_dot_nmpc", self.olnmpc.name)

        results = self.dot_driver.solve(self.olnmpc, tee=True)
        self.olnmpc.solutions.load_from(results)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)

        ftiming = open("timings_dot_driver.txt", "r")
        s = ftiming.readline()
        ftiming.close()

        f = open("timings_nmpc_dot.txt", "a")
        f.write(str(s) + '\n')
        f.close()

        k = s.split()
        self._dot_timing = k[0]

    def sens_k_aug_nmpc(self):
        """Calls `k_aug` to compute the sensitivity matrix (reduced mode)

        """
        self.journalist("I", self._iteration_count, "sens_k_aug_nmpc", "k_aug sensitivity")
        self.olnmpc.ipopt_zL_in.update(self.olnmpc.ipopt_zL_out)
        self.olnmpc.ipopt_zU_in.update(self.olnmpc.ipopt_zU_out)
        self.journalist("I", self._iteration_count, "solve_k_aug_nmpc", self.olnmpc.name)

        if hasattr(self.olnmpc, "f_timestamp"):
            self.olnmpc.f_timestamp.clear()
        else:
            self.olnmpc.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                             datatype=Suffix.INT)

        self.olnmpc.set_suffix_value(self.olnmpc.f_timestamp, self.int_file_nmpc_suf)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)
        results = self.k_aug_sens.solve(self.olnmpc, tee=True, symbolic_solver_labels=False)
        self.olnmpc.solutions.load_from(results)
        #: Read the reported timings from `k_aug`
        ftimings = open("timings_k_aug.txt", "r")
        s = ftimings.readline()
        ftimings.close()

        f = open("timings_nmpc_kaug.txt", "a")
        f.write(str(s) + '\n')
        f.close()

        self._k_timing = s.split()

    def find_target_ss(self, ref_state=None, **kwargs):
        """Attempt to find a second steady state
        Args:
            ref_state (dict): Contains the reference state with value key "state", (j,): value
            kwargs (dict): Optional arguments
        Returns
            None"""

        if ref_state:
            self.ref_state = ref_state
        else:
            if not ref_state:
                self.journalist("W", self._iteration_count,
                                "find_target_ss", "No reference state was given, using default")
            if not self.ref_state:
                self.journalist("W", self._iteration_count,
                                "find_target_ss", "No default reference state was given, exit")
                sys.exit()

        weights_ref = dict.fromkeys(self.ref_state.keys())
        #: Default weights are computed by taking the inverse of the diffference between Ref and vsoi
        for i in self.ref_state.keys():
            v = getattr(self.SteadyRef, i[0])
            vkey = i[1]
            vss0 = value(v[(1,) + vkey])
            val = abs(self.ref_state[i] - vss0)
            if val < 1e-09:
                val = 1e+06
            else:
                val = 1/val
            weights_ref[i] = val

        weights = kwargs.pop("weights", weights_ref)

        self.journalist("I", self._iteration_count, "find_target_ss", "Attempting to find steady state")

        del self.SteadyRef2
        self.SteadyRef2 = clone_the_model(self.d_mod) # (1, 1)
        self.SteadyRef2.name = "SteadyRef2 (reference)"
        augment_steady(self.SteadyRef2)
        create_bounds(self.SteadyRef2, bounds=self.var_bounds)

        for u in self.u:
            cv = getattr(self.SteadyRef2, u)  #: Get the param
            c_val = value(cv[1])  #: Current value
            dumm_eq = getattr(self.SteadyRef2, u + '_cdummy')

            dexpr = dumm_eq[1].expr.args[0]
            control_var = getattr(self.SteadyRef2, dexpr.parent_component().name)
            if isinstance(control_var, Var):  #: all good
                pass
            else:
                raise ValueError  #: Some exception here

            self.SteadyRef2.del_component(cv)  #: Delete the dummy_param
            self.SteadyRef2.del_component(dumm_eq)  #: Delete the dummy_constraint
            self.SteadyRef2.add_component(u, Var([1], initialize=lambda m, i: c_val))
            cv = getattr(self.SteadyRef2, u)  #: Get the new variable
            for k in cv.keys():
                cv[k].setlb(self.u_bounds[u][0])
                cv[k].setub(self.u_bounds[u][1])

            self.SteadyRef2.add_component(u + '_cdummy', Constraint([1]))
            dumm_eq = getattr(self.SteadyRef2, u + '_cdummy')
            dumm_eq.rule = lambda m, i: cv[i] == control_var[i]
            dumm_eq.reconstruct()

        for vs in self.SteadyRef.component_objects(Var, active=True):  #: Load_guess
            vt = getattr(self.SteadyRef2, vs.getname())
            for ks in vs.keys():
                vt[ks].set_value(value(vs[ks]))
        ofexp = 0
        for i in self.ref_state.keys():
            v = getattr(self.SteadyRef2, i[0])
            vkey = i[1]
            ofexp += weights[i] * (v[(1,) + vkey] - self.ref_state[i])**2
        self.SteadyRef2.obfun_SteadyRef2 = Objective(expr=ofexp, sense=minimize)
        tst = self.solve_dyn(self.SteadyRef2, iter_max=10000, stop_if_nopt=True, halt_on_ampl_error=False, **kwargs)
        if tst != 0:
            self.SteadyRef2.display(filename="failed_SteadyRef2.txt")
            self.SteadyRef2.write(filename="failed_SteadyRef2.nl",
                           format=ProblemFormat.nl,
                           io_options={"symbolic_solver_labels": True})
            # sys.exit(-1)
        self.journalist("I", self._iteration_count, "find_target_ss", "Target: solve done")
        for i in self.ref_state.keys():
            print(i)
            v = getattr(self.SteadyRef2, i[0])
            vkey = i[1]
            val = value(v[(1,) + vkey])
            print("target {:}".format(i[0]),
                  "\tkey {:}".format(i[1]),
                  "\tweight {:f}".format(weights[i]),
                  "\tvalue {:f}".format(val))
        for u in self.u:
            v = getattr(self.SteadyRef2, u)
            val = value(v[1])
            print("target {:}".format(u),
                  "\tvalue {:f}".format(val))
        self.update_targets_nmpc()

    def update_targets_nmpc(self):
        """Use the reference model to update  the current state and control targets dictionaries"""
        for x in self.states:
            xvar = getattr(self.SteadyRef2, x)
            for j in self.state_vars[x]:
                self.curr_state_target[(x, j)] = value(xvar[(1,) + j])
        for u in self.u:
            uvar = getattr(self.SteadyRef2, u)
            self.curr_u_target[u] = value(uvar[1])

    def change_setpoint(self, ref_state, **kwargs):
        """update the ref_state dictionary, and attempt to find a new reference state"""
        if ref_state:
            self.ref_state = ref_state
        else:
            if not ref_state:
                self.journalist("W", self._iteration_count, "change_setpoint", "No reference state was given, using default")
            if not self.ref_state:
                self.journalist("W", self._iteration_count, "change_setpoint", "No default reference state was given, exit")
                sys.exit()
        #: Create a dictionary whose keys are the same as the ref state
        weights_ref = dict.fromkeys(self.ref_state.keys())
        model_ref = self.PlantSample  #: I am not sure about this
        for i in self.ref_state.keys():
            v = getattr(model_ref, i[0])
            vkey = i[1]
            vss0 = value(v[(0,) + vkey])
            val = abs(self.ref_state[i] - vss0)
            if val < 1e-09:
                val = 1e+06
            else:
                val = 1/val
            weights_ref[i] = val

        #: If no weights are passed, use the reference that we have just calculated
        weights = kwargs.pop("weights", weights_ref)

        ofexp = 0.0
        for i in self.ref_state.keys():
            v = getattr(self.SteadyRef2, i[0])
            vkey = i[1]
            ofexp += weights[i] * (v[(1,) + vkey] - self.ref_state[i]) ** 2

        self.SteadyRef2.obfun_SteadyRef2.set_value(ofexp)
        self.solve_dyn(self.SteadyRef2, iter_max=500, stop_if_nopt=True, **kwargs)

        for i in self.ref_state.keys():
            v = getattr(self.SteadyRef2, i[0])
            vkey = i[1]
            val = value(v[(1,) + vkey])
            print("target {:}".format(i[0]), "key {:}".format(i[1]), "weight {:f}".format(weights[i]),
                  "value {:f}".format(val))
        self.update_targets_nmpc()

    def compute_offset_state(self, src_kind="estimated"):
        """Missing noisy"""
        if src_kind == "estimated":
            for x in self.states:
                for j in self.state_vars[x]:
                    self.curr_state_offset[(x, j)] = self.curr_estate[(x, j)] - self.curr_pstate[(x, j)]
        elif src_kind == "real":
            for x in self.states:
                for j in self.state_vars[x]:
                    self.curr_state_offset[(x, j)] = self.curr_rstate[(x, j)] - self.curr_pstate[(x, j)]

    def print_r_nmpc(self):
        """This updates the soi for some reason"""
        self.journalist("I", self._iteration_count, "print_r_nmpc", "Results at" + os.getcwd())
        self.journalist("I", self._iteration_count, "print_r_nmpc", "Results suffix " + self.res_file_suf)
        for k in self.ref_state.keys():
            self.soi_dict[k].append(self.curr_soi[k])
            self.sp_dict[k].append(self.curr_sp[k])
            print("Current values\t", self.ref_state[k], k)

        with open("res_nmpc_rs_" + self.res_file_suf + ".txt", "a") as f:
            for k in self.ref_state.keys():
                i = self.soi_dict[k]
                iv = str(i[-1])
                f.write(iv)
                f.write('\t')
            for k in self.ref_state.keys():
                i = self.sp_dict[k]
                iv = str(i[-1])
                f.write(iv)
                f.write('\t')
            for u in self.u:
                i = self.curr_u[u]
                iv = str(i)
                f.write(iv)
                f.write('\t')
            for u in self.u:
                i = self.curr_ur[u]
                iv = str(i)
                f.write(iv)
                f.write('\t')
            f.write('\n')
            f.close()

        with open("res_nmpc_offs_" + self.res_file_suf + ".txt", "a") as f:
            for x in self.states:
                for j in self.state_vars[x]:
                    i = self.curr_state_offset[(x, j)]
                    iv = str(i)
                    f.write(iv)
                    f.write('\t')
            f.write('\n')
            f.close()

    def update_soi_sp_nmpc(self):
        """States-of-interest and set-point update"""
        if bool(self.soi_dict):
            pass
        else:
            for k in self.ref_state.keys():
                self.soi_dict[k] = []

        if bool(self.sp_dict):
            pass
        else:
            for k in self.ref_state.keys():
                self.sp_dict[k] = []

        for k in self.ref_state.keys():
            vname = k[0]
            vkey = k[1]
            var = getattr(self.PlantSample, vname)
            #: Assuming the variable is indexed by time
            t = t_ij(self.PlantSample.t, 0, self.ncp_t)
            self.curr_soi[k] = value(var[(t, ) + vkey])
        for k in self.ref_state.keys():
            vname = k[0]
            vkey = k[1]
            var = getattr(self.SteadyRef2, vname)
            #: Assuming the variable is indexed by time
            self.curr_sp[k] = value(var[(1,) + vkey])
        self.journalist("I", self._iteration_count, "update_soi_sp_nmpc", "Current offsets + Values:")
        for k in self.ref_state.keys():
            #: Assuming the variable is indexed by time
            self.curr_off_soi[k] = 100 * abs(self.curr_soi[k] - self.curr_sp[k])/abs(self.curr_sp[k])
            print("\tCurrent offset \% \% \t", k, self.curr_off_soi[k], end="\t")
            print("\tCurrent value \% \% \t", self.curr_soi[k])

        for u in self.u:
            ur = getattr(self.SteadyRef2, u)
            self.curr_ur[u] = value(ur[1])

    def method_for_nmpc_simulation(self):
        pass
    
    def setup_sp_profile(self, ref_info):
        
        for i in ref_info.keys():
            self.profile_state_target[i] = {}
            self.profile_u_target[i] = {}
            ref = ref_info[i]
            self.change_setpoint(ref_state=ref, keepsolve=True, wantparams=True, tag="sp")
            self.profile_state_target[i] = self.curr_state_target.copy()
            self.profile_u_target[i] = self.curr_u_target.copy()
            
        self.profile_target = True
            
    def create_suffixes_amsnmpc(self):
        '''create suffixes for amsnmpc. Because we need to extend the KKT matrix, 
        dsdp mode is used. Therefore, create suffixes seperately from asnmpc.
        '''
        
        if hasattr(self.olnmpc, "dcdp"):
            self.olnmpc.dcdp.clear()
        else:
            self.olnmpc.dcdp = Suffix(direction=Suffix.EXPORT)
        if hasattr(self.olnmpc, "var_order"):
            self.olnmpc.var_order.clear()
        else:
            self.olnmpc.var_order = Suffix(direction=Suffix.EXPORT)
            
        #declare dummy constraints(i.e. ics) in dcdp
        count = 1
        for x in self.states:
            con_name = x + "_icc"
            con_ = getattr(self.olnmpc, con_name)
            for j in self.state_vars[x]:
                con_[j].set_suffix_value(self.olnmpc.dcdp, count)
                count += 1
        
        self.record_suffix_u = {}
        count_var = 1
        for i in range(self.amsnmpc_Ns):
            # t = t_ij(self.olnmpc.t, i, 0)
            self.record_suffix_u[i] = {}
            for u in self.u:
                uv = getattr(self.olnmpc, u)
                uv[i].set_suffix_value(self.olnmpc.var_order, count_var)
                self.record_suffix_u[i][u] = count_var
                count_var += 1
                
        self.record_suffix_x = {}
        for i in range(self.amsnmpc_Ns):
            self.record_suffix_x[i] = {}
            t = t_ij(self.olnmpc.t, i, 0)
            for x in self.states:
                xvar = getattr(self.olnmpc, x)
                for j in self.state_vars[x]:
                    xvar[(t,) + j].set_suffix_value(self.olnmpc.var_order, count_var)
                    self.record_suffix_x[i][(x,j)] = count_var
                    count_var += 1
                    
        # print(self.record_suffix_u)
        # print(self.record_suffix_x)
    
    def store_u_within_Ns(self):
        for i in range(self.amsnmpc_Ns):
            # t = t_ij(self.olnmpc.t, i, 0)
            for u in self.u:
                uvar = getattr(self.olnmpc, u)
                vu = value(uvar[i])
                self.u_within_Ns_store[i][u] = vu
        # print(self.u_within_N)
        
    def store_z_within_Ns(self):
        for i in range(self.amsnmpc_Ns):
            t = t_ij(self.olnmpc.t, i, 0)
            state = {}
            for x in self.states:
                xvar = getattr(self.olnmpc, x)
                for j in self.state_vars[x]:
                    state[(x,j)] = value(xvar[t,j])
                    self.z_within_Ns_store[i] = state
                    
    def get_var_con_info_kaug(self, stampname=""):
        newname_var = "varorder" + str(stampname) + ".txt"
        self.varinfo = np.genfromtxt(newname_var, dtype=int)
        newname_con = "conorder" + str(stampname) + ".txt"
        self.coninfo = np.genfromtxt(newname_con, dtype=int)
        
        self.var_num, = np.shape(self.varinfo)
        self.con_num, = np.shape(self.coninfo)
        
    def sens_k_aug_amsnmpc(self): 
        '''
        calls k_aug to compute the kkt matrix (dsdp mode)

        '''
        self.journalist("I", self._iteration_count, "sens_k_aug_amsnmpc", "k_aug sensitivity")
        self.olnmpc.ipopt_zL_in.update(self.olnmpc.ipopt_zL_out)
        self.olnmpc.ipopt_zU_in.update(self.olnmpc.ipopt_zU_out)
        
        self.journalist("I", self._iteration_count, "solve_k_aug_amsnmpc", self.olnmpc.name)
        self.k_aug_sens.options["dsdp_mode"] = ""
        results = self.k_aug_sens.solve(self.olnmpc, tee=True, symbolic_solver_labels=False)
        self.olnmpc.solutions.load_from(results)
        
        os.rename(r'kkt.in', r'kkt' + str(self.int_file_nmpc_suf) + '.in')
        os.rename(r'varorder.txt', r'varorder' + str(self.int_file_nmpc_suf) + '.txt')
        os.rename(r'conorder.txt', r'conorder' + str(self.int_file_nmpc_suf) + '.txt')
        
        self.amsnmpc_kkt, self.amsnmpc_kkt_size = get_lu_KKT(namestamp = str(self.int_file_nmpc_suf))
        self.get_var_con_info_kaug(stampname = str(self.int_file_nmpc_suf))
        
    def calculate_ds_int(self):
        '''
        kkt * ds = -dphidp * delP0, (final target is ds but only solve for ds_int in this function.)
        ds = inv(kkt) * (-dphidp) *delP0
           = ds_int *delP0, where ds_int = inv(kkt) * (-dphidp)
        '''
        dphidp = np.zeros([self.var_num + self.con_num, self.num_flatten_var])
        p_ind = {}
        # for i, j in enumerate(self.states):
        #     p_ind[j], = self.var_num + np.where(self.coninfo == i+1)[0]
        #     dphidp[p_ind[j], i] = -1
        count = 1
        for x in self.states:
            for j in self.state_vars[x]:
                p_ind[(x,j)],  = self.var_num + np.where(self.coninfo == count)[0]
                dphidp[p_ind[(x,j)], count-1] = -1. 
                count += 1
        ds_int = self.amsnmpc_kkt.solve(-dphidp)
        # ds = np.matmul(ds_int, delP0)
        return ds_int        
        
    def build_amsNMPC_E0(self):
        row_E0 = []
        p_ind = {}
        # nz = len(self.states)
        # for i, j in enumerate(self.states):
        #     p_ind[j], = self.var_num + np.where(self.coninfo == i+1)[0]
        #     row_E0.append(p_ind[j])
        count = 1
        for x in self.states:
            for j in self.state_vars[x]:
                p_ind[(x,j)], = self.var_num + np.where(self.coninfo == count)[0]
                row_E0.append(p_ind[(x,j)])
                count += 1
        col_E0 = np.array([j for j in range(self.num_flatten_var)])
        data_E0 = -1. * np.ones_like(col_E0)
        E0 = csc_matrix((data_E0, (row_E0, col_E0)), shape=(self.amsnmpc_kkt_size, self.num_flatten_var)).toarray()
        return E0
    
    def build_amsNMPC_Mj(self, j_update): #j_update is element < Ns
        if j_update == 0:
            raise RuntimeError("j_update = 0 corresponds to regular ds")
    
        current_record = self.record_suffix_x[j_update]
        row_Mj = []
        z_ind = {}
        # nz = len(self.states)
        nz = self.num_flatten_var
        for i in current_record.keys():
            z_ind[i], = np.where(self.varinfo == current_record[i])[0]
            row_Mj.append(z_ind[i])
        col_Mj = np.array([j for j in range(nz)])
        data_Mj = 1. * np.ones_like(col_Mj)
        Mj = csc_matrix((data_Mj, (row_Mj, col_Mj)), shape=(self.amsnmpc_kkt_size, nz)).toarray()
        return Mj
    
    def solve_dj_Schur(self, lu_KKT, E0, Mj):
        '''solve the extended sensitivity matrix for dj with Schur complement'''
        
        E0_T = E0.transpose()
        Mj_T = Mj.transpose()
        #S0
        KinvE0 = lu_KKT.solve(E0)
        S0 = -np.matmul(E0_T, KinvE0)
        #S0_bar, Sj
        KinvMj = lu_KKT.solve(Mj)
        S0_bar = -np.matmul(E0_T, KinvMj)
        S0_bar_T = S0_bar.transpose()
        S0invS0bar = np.linalg.solve(S0, S0_bar)
        Sj = -np.matmul(Mj_T, KinvMj) - np.matmul(S0_bar_T, S0invS0bar)
        #backsolve dj
        dj3 = np.linalg.solve(Sj, np.eye(self.num_flatten_var))
        dj2 = np.linalg.solve(S0, -np.matmul(S0_bar, dj3))
        dj1 = lu_KKT.solve(-np.matmul(E0, dj2) - np.matmul(Mj, dj3))
        dj = np.concatenate((dj1, dj2, dj3), axis=0)
        return dj
        
    def get_sensitivity_info(self):
        # if status == "regular": #not for status == "prior"
        self.amsnmpc_ds_int_store = self.calculate_ds_int()
            
        E0 = self.build_amsNMPC_E0()
        for i in range(1, self.amsnmpc_Ns):
            Mj = self.build_amsNMPC_Mj(j_update = i)
            result_dj = self.solve_dj_Schur(self.amsnmpc_kkt, E0, Mj)
            self.amsnmpc_dj_store[i] = result_dj
            
    def setup_info_for_extended_sensitivity(self):
        '''
        setup some required info for extended sensitivity strategy
        1. save the first Ns u
        2. save the first Ns states
        3. solve the kkt matrix with k_aug
        4. calculate ds_int and dj

        '''
        self.store_u_within_Ns()
        self.store_z_within_Ns()
        self.sens_k_aug_amsnmpc()
        self.get_sensitivity_info()
            
    def load_ds_int_and_dj(self):
        '''
        load data from amsnmpc_ds_int_store & amsnmpc_dj_store
                  to   amsnmpc_ds_int_recent & amsnmpc_dj_recent

        '''
        self.amsnmpc_ds_int_recent = deepcopy(self.amsnmpc_ds_int_store)
        self.amsnmpc_dj_recent = deepcopy(self.amsnmpc_dj_store)
        self.u_within_Ns_recent = deepcopy(self.u_within_Ns_store)
        self.z_within_Ns_recent = deepcopy(self.z_within_Ns_store)
                    
    def sens_dot_amsnmpc(self, stage, src ="estimated"): #stage: 0 ~ Ns-1
        '''
        update control inputs with sensitivity
        
        '''
        if stage == 0: #be going to use results from diferent olnmpc problem, need to load data from store to recent
            self.load_ds_int_and_dj()
        
        if src == "estimated":
            true_state = self.curr_estate
        elif src == "real":
            true_state = self.curr_rstate
         
        #get delP of interest
        pred_state = self.z_within_Ns_recent[stage]
        delP_dict = {}
        for i in pred_state.keys():
            delP_dict[i] = true_state[i] - pred_state[i]     
        delP = []
        for i in delP_dict.keys():
            delP.append([delP_dict[i]])
        delP = np.array(delP)
        
        #get u want to update
        pred_u = self.u_within_Ns_recent[stage]
        pred_u_list = []
        for i in pred_u.keys():
            pred_u_list.append([pred_u[i]])
        pred_u_array = np.array(pred_u_list)
        
        current_suffix_u = self.record_suffix_u[stage]
        
        if stage == 0:
            ds_int = self.amsnmpc_ds_int_recent
            ds = np.matmul(ds_int, delP)
            
            # dv0 = []
            # for i in current_suffix_u.keys():
            #     suf = current_suffix_u[i]
            #     ind, = np.where(self.varinfo == suf)[0]
            #     dv0.append(ds[ind])
            # dv0 = np.array(dv0)
            # v0_mod = pred_u_array + dv0
            # self.u_mod = v0_mod
            
            dv = []
            for k in range(self.amsnmpc_Ns):
                suffix_u = self.record_suffix_u[k]
                for i in suffix_u.keys():
                    suf = suffix_u[i]
                    ind, = np.where(self.varinfo == suf)[0]
                    dv.append(ds[ind])
            dv = np.array(dv)
            
            pred_u_array_full = []
            for k in range(self.amsnmpc_Ns):
                val_dic = self.u_within_Ns_recent[k]
                for i in val_dic.keys():
                    pred_u_array_full.append([val_dic[i]]) 
            pred_u_array_full = np.array(pred_u_array_full)
            
            v_mod = pred_u_array_full + dv
            # print(pred_u_array_full)
            # print(dv)
            # print(v_mod)
            nu = len(self.u)
            self.u_mod = v_mod[:nu][:] #only need u in the first step
            
            count = 0
            for i in range(self.amsnmpc_Ns):
                for j in self.u:
                    self.u_for_pred[i][j] = v_mod[count][0]
                    count += 1    
        
        else:
            # E0 = self.build_amsNMPC_E0()
            # Mj = self.build_amsNMPC_Mj(j_update=stage)
            # dj = self.solve_dj_Schur(self.amsnmpc_kkt, E0, Mj)
            dj = self.amsnmpc_dj_recent[stage]
            ds = np.matmul(dj, delP)
            dvj = []
            # for i in current_suffix_u.keys():
            #     dvj.append([current_suffix_u[i]])
            for i in current_suffix_u.keys():
                suf = current_suffix_u[i]
                ind,  = np.where(self.varinfo == suf)[0]
                dvj.append(ds[ind])
            dvj = np.array(dvj)
            vj_mod = pred_u_array + dvj
            self.u_mod = vj_mod
            
    def create_predictor_amsNMPC(self):
        self.Pred_amsnmpc = clone_the_model(self.d_mod)  # (1, self.ncp_t, _t=self.hi_t)
        augment_model(self.Pred_amsnmpc, self.amsnmpc_Ns, self.ncp_t, new_timeset_bounds=(0, self.hi_t*self.amsnmpc_Ns))
        
        self.Pred_amsnmpc.name = "Dynamic Predictor for amsNMPC"
        aug_discretization(self.Pred_amsnmpc, nfe=self.amsnmpc_Ns, ncp=self.ncp_t)

    def predictor_amsNMPC(self, src="estimated"):
        """Predict the states for the next Nsth step for amsNMPC"""
        if self.Pred_amsnmpc:
            pass
        else:
            self.create_predictor_amsNMPC()
        for i in range(self.amsnmpc_Ns):
            load_iguess(self.olnmpc, self.Pred_amsnmpc, i, i) #better to use result after update but it's fine KH.L
        if src == "estimated":
            self.load_init_state_gen(self.Pred_amsnmpc, src_kind="dict", state_dict="estimated")  #: Load the initial state
        else:
            self.load_init_state_gen(self.Pred_amsnmpc, src_kind="dict", state_dict="real")  #: Load the initial state
            
        #inject inputs
        check = 0
        for i in self.u_for_pred.keys():
            if not self.u_for_pred[i]:
                check = 1
        if check == 0:
            u_inject = self.u_for_pred
        else:
            u_inject = self.u_within_Ns_recent
        for i in range(self.amsnmpc_Ns):
            for j in range(self.ncp_tnmpc+1):
                tij = t_ij(self.Pred_amsnmpc.t, i, j)
                for u in self.u:
                    pred_var = getattr(self.Pred_amsnmpc, u)
                    pred_var[tij].value = u_inject[i][u]
       
        stat = self.solve_dyn(self.Pred_amsnmpc, skip_update=True,
                              iter_max=250,
                              stop_if_nopt=True,
                              jacobian_regularization_value=1e-02,
                              linear_scaling_on_demand=True)
        for x in self.states:
            xvar = getattr(self.Pred_amsnmpc, x)
            t = t_ij(self.Pred_amsnmpc.t, self.amsnmpc_Ns-1, self.ncp_t) #end time
            for j in self.state_vars[x]:
                self.curr_pstate[(x, j)] = value(xvar[t, j]) 
            
    def update_u_amsnmpc(self):
        stat = 0
        fe = 0
        for i,u in enumerate(self.u):
            val = self.u_mod[i][0]
            # print(val)
            self.curr_u[u] = val
            uvar = getattr(self.olnmpc, u)
            if uvar[fe].lb is None:
                pass
            else:
                if val < uvar[fe].lb:
                    stat = 1
            if uvar[fe].ub is None:
                pass
            else:
                if val > uvar[fe].ub:
                    stat = 1
        return stat
        
    #terminal properties
    def tp_get_true_control_name(self):
        '''
        Get the true name of controls in the model and save it in 
        self.true_u_name(list).

        '''
        self.true_u_name = []
        for u in self.u:
            dumm_eq = getattr(self.d_mod, u+"_cdummy")
            dexpr = dumm_eq[0].expr.args[0]
            true_control_name = dexpr.parent_component().name
            self.true_u_name.append(true_control_name)
    
    def tp_get_differential_var(self):
        '''
        Get the derivative variables(e.g. dotCa) and save in the self.der_var.
        Also create a dict self.diff_der_var for connection 
        between der_var(key) and diff_var(value)

        '''
        self.der_var = []
        self.diff_der_var = {}
        for dv in self.d_mod.component_objects(DerivativeVar):
            self.der_var.append(dv.name)
            diffvar = dv.get_state_var()
            self.diff_der_var[dv.name] = diffvar.getname()
            
    def catagorize_equations_vars(self):
        '''
        self.other_var_list(list)
        self.ohter_vars(dic)
        self.diff_equ(list)
        self.alge_equ(list)
        self.diff_var_con(dic): connection between diff_var and its diff_equ

        '''
        self.other_var_list = []
        self.other_vars = {}
        for var in self.tp_model.component_objects(Var):
            name = var.getname()
            if name not in self.states + self.true_u_name + self.der_var:
                self.other_var_list.append(name)
                self.other_vars[name] = []
                #flatten other_var_list
                if var._implicit_subsets is None:
                    self.other_vars[name] = ((),)
                else:
                    remaining_set = var._implicit_subsets[1]
                    for j in range(2, len(var._implicit_subsets)):
                        remaining_set *= var._implicit_subsets[j]
                    for index in remaining_set:
                        if isinstance(index, tuple):
                            self.other_vars[name].append(index)
                        else:
                            self.other_vars[name].append((index,))     
        
        self.diff_equ = []
        self.alge_equ = []
        self.diff_var_con = {}
        for con in self.tp_model.component_objects(Constraint):
            conname = con.getname()                
            for i in list(con.keys()):
                conexpr_string = con[i].expr.to_string()
                flag = 0
                for dv in self.der_var:
                    if dv in conexpr_string:
                        self.diff_equ.append((conname,i))
                        set_index = self.diff_der_var[dv]
                        self.diff_var_con[set_index] = conname
                        flag = 1
                if flag == 0:
                    self.alge_equ.append((conname,i))          
    
    def tp_k_aug_suffix_locate_cons_vars(self):
        
        if hasattr(self.olnmpc, "dcdp"):
            self.tp_model.dcdp.clear()
        else:
            self.tp_model.dcdp = Suffix(direction=Suffix.EXPORT)  #: the dummy constraints
            
        if hasattr(self.olnmpc, "var_order"):
            self.tp_model.var_order.clear()
        else:
            self.tp_model.var_order = Suffix(direction=Suffix.EXPORT)  #: Important variables (primal)
        
        self.tp_vars_suffix = dict()
        self.tp_cons_suffix = dict()
        count_var = 0
        count_con = 0
        for x in self.states:
            xvar = getattr(self.tp_model, x)
            for j in self.state_vars[x]:
                count_var += 1
                xvar[(1,) + j].set_suffix_value(self.tp_model.var_order, count_var)
                self.tp_vars_suffix[(x, (1,) + j)] = count_var
            
            condotx = self.diff_var_con[x]
            conv = getattr(self.tp_model, condotx)
            for j in self.state_vars[x]: #remaining set of constraint must be the same as x KH.L
                index = (1,) + j
                count_con += 1 
                conv[index].set_suffix_value(self.tp_model.dcdp, count_con)
                self.tp_cons_suffix[(condotx, index)] = count_con
        
            # for k in self.diff_equ[condotx]:
            #     count_con += 1
            #     conv = getattr(self.tp_model, condotx)
            #     conv[k].set_suffix_value(self.tp_model.dcdp, count_con)
            #     self.tp_cons_suffix[(condotx)] = count_con
                
        for u in self.true_u_name:
            count_var += 1
            uvar = getattr(self.tp_model, u)
            only_key = list(uvar.keys())[1]
            uvar[only_key].set_suffix_value(self.tp_model.var_order, count_var)
            self.tp_vars_suffix[(u,)] = count_var
        for ov in self.other_var_list:
            ovar = getattr(self.tp_model, ov)
            for j in self.other_vars[ov]:
                count_var += 1
                ovar[(1,) + j].set_suffix_value(self.tp_model.var_order, count_var)
                self.tp_vars_suffix[(ov, (1,) + j)] = count_var
        
        # for con_list in [self.diff_equ, self.alge_equ]:
        for con_flag in self.alge_equ:
            count_con += 1
            name = con_flag[0]
            index = con_flag[1]
            conv = getattr(self.tp_model, name)
            conv[index].set_suffix_value(self.tp_model.dcdp, count_con)
            self.tp_cons_suffix[con_flag] = count_con
        
    def tp_rearrange_jac(self, jac):
        #change the column order first
        col_jac_info = np.genfromtxt("varorder.txt", dtype = float)
        ncol, = np.shape(col_jac_info)
        col_perm = np.zeros((ncol, ncol))
        for i, idx in enumerate(col_jac_info):
            col_perm[i, np.int(idx)-1] = 1
        newjac = np.dot(jac, col_perm)
        
        #change the row order next
        row_jac_info = np.genfromtxt("conorder.txt", dtype = float)
        nrow, = np.shape(row_jac_info)
        row_perm = np.zeros((nrow, nrow))
        for i, idx in enumerate(row_jac_info):
            row_perm[np.int(idx)-1, i] = 1
        reordered_jac = np.dot(row_perm, newjac)
        return reordered_jac
    
    def tp_number_vars_cons_info(self):
        '''
        Get info about number of 
        1. flatten sates (already in self.num_flatten_var)
        2. controls >> self.n_controls
        3. other flatten vars not states >> self.n_flatten_othervars
        4. flatten differential equations >> self.n_DE
        4. flatten algebric equations >> self.n_AE

        '''
        # self.num_flatten_var
        self.n_controls = len(self.u)
        count = 0
        for ov in self.other_var_list:
            for j in self.other_vars[ov]:
                count+=1
        self.n_flatten_othervars = count
        
        self.n_DE = len(self.diff_equ)
        self.n_AE = len(self.alge_equ)
        
    def tp_calculate_A(self, reordered_jac):
        dfdx = reordered_jac[0:self.n_DE, 0:self.num_flatten_var]
        dfdy = reordered_jac[0:self.n_DE, 
                             self.num_flatten_var+self.n_controls:self.num_flatten_var+self.n_controls+self.n_flatten_othervars]
        dgdx = reordered_jac[self.n_DE:self.n_DE+self.n_AE, 0:self.num_flatten_var]
        dgdy = reordered_jac[self.n_DE:self.n_DE+self.n_AE, 
                             self.num_flatten_var+self.n_controls:self.num_flatten_var+self.n_controls+self.n_flatten_othervars]
          
        a1 = np.matmul(dfdy, np.linalg.inv(dgdy))
        tp_A = dfdx - np.matmul(a1, dgdx) 
        tp_A = -tp_A
        return tp_A
    
    def tp_calculate_B(self, reordered_jac):
        dfdx = reordered_jac[0:self.n_DE,
                             self.num_flatten_var:self.num_flatten_var+self.n_controls]
        dfdy = reordered_jac[0:self.n_DE, 
                             self.num_flatten_var+self.n_controls:self.num_flatten_var+self.n_controls+self.n_flatten_othervars]
        dgdx = reordered_jac[self.n_DE:self.n_DE+self.n_AE,
                             self.num_flatten_var:self.num_flatten_var+self.n_controls]
        dgdy = reordered_jac[self.n_DE:self.n_DE+self.n_AE, 
                             self.num_flatten_var+self.n_controls:self.num_flatten_var+self.n_controls+self.n_flatten_othervars]
        
        b1 = np.matmul(dfdy, np.linalg.inv(dgdy))
        tp_B = dfdx - np.matmul(b1, dgdx) 
        tp_B = -tp_B
        return tp_B
        
    def tp_linearize_A_B(self):
        '''
        Linearize the DAE model with steady states and controls. 
        1. Solve for the steady states and controls
        2. Get the Jacobian matrix with k_aug
        3. Rearrange the Jacobian 
        4. Retrieve the needed info to calculate A and B
            (because the model is DAE not ODE, need to use the "derivative jacobian"
             to get A and B)

        Returns
        -------
        tp_A : numpy.ndarray(matrix)
        tp_B : numpy.ndarray(matrix)

        '''
        self.tp_get_true_control_name()
        self.tp_get_differential_var()
        
        self.tp_model = clone_the_model(self.SteadyRef2)
        obj_sim = 1.0
        self.tp_model.obfun_SteadyRef2.set_value(obj_sim)
        for i in self.tp_model.component_objects(Var):
            if isinstance(i, DerivativeVar):
                continue #DerivativeVar have no bounds
            i.setlb(None)
            i.setub(None)
        for u in self.u:
            uvar = getattr(self.tp_model, u)
            uvar.fix()
        self.solve_dyn(self.tp_model, iter_max=10, stop_if_nopt=True)
        
        self.tp_state_ss = {}
        for x in self.states:
            xvar = getattr(self.tp_model, x)
            for j in self.state_vars[x]: #j in setProduct KH.L
                self.tp_state_ss[(x, j)] = value(xvar[(1,) + j])
                if abs(self.tp_state_ss[(x, j)] - self.curr_state_target[(x, j)]) >= 1.0E-5:
                    raise RuntimeError("Error when calculating terminal properties: states")
       
        self.tp_u_ss = {}
        for u in self.u:
            uvar = getattr(self.tp_model, u)
            self.tp_u_ss[u] = value(uvar[1])
            if abs(self.tp_u_ss[u] - self.curr_u_target[u]) >= 1.0E-5:
                raise RuntimeError("Error when calculating terminal properties: controls")
        
        for u in self.u:
            dumm_eq = getattr(self.tp_model, u + "_cdummy")
            self.tp_model.del_component(dumm_eq)
            self.tp_model.del_component(u)
        
        self.catagorize_equations_vars()
        self.tp_k_aug_suffix_locate_cons_vars()
        
        self.solve_dyn(self.tp_model, iter_max=2, stop_if_nopt=True)
        
        self.tp_model.ipopt_zL_in.update(self.tp_model.ipopt_zL_out)  #: important!
        self.tp_model.ipopt_zU_in.update(self.tp_model.ipopt_zU_out)  #: important!
        
        self.k_aug_sens.options["dsdp_mode"] = ""
        results = self.k_aug_sens.solve(self.tp_model, tee=True, symbolic_solver_labels=False)
        self.tp_model.solutions.load_from(results)
        os.rename(r'jacobi_debug.in', r'jacobi_debug' + str(self.int_file_nmpc_suf) + '.in')
        
        jac, sizejac = get_jacobian_k_aug(self.int_file_nmpc_suf)
        reordered_jac = self.tp_rearrange_jac(jac)
        
        self.tp_number_vars_cons_info()
        tp_A = self.tp_calculate_A(reordered_jac)
        tp_B = self.tp_calculate_B(reordered_jac)
        
        return tp_A, tp_B
    
    def tp_build_Q_R(self):
        
        tp_Q_nmpc = np.zeros((self.num_flatten_var, self.num_flatten_var))
        Q_comp = getattr(self.olnmpc, "Q_nmpc")
        for x in self.states:
            for j in self.state_vars[x]:
                k = self.xmpc_key[(x, j)]
                tp_Q_nmpc[k,k] = value(Q_comp[k])
        
        tp_R_nmpc = np.zeros((self.n_controls, self.n_controls))
        R_comp = getattr(self.olnmpc, "R_nmpc")
        count = 0
        for u in self.u:
            tp_R_nmpc[count, count] = value(R_comp[count])
            count += 1
        
        return tp_Q_nmpc, tp_R_nmpc
    
    def tp_simulation_many_pts(self, simulate_points, state_norm, tp_Ad, tp_Bd, K, plot_figure):
        
        #initilaize with ss
        self.tp_simulate = clone_the_model(self.d_mod)
        augment_model(self.tp_simulate, 1, self.ncp_t, new_timeset_bounds=(0, self.hi_t))
        aug_discretization(self.tp_simulate, nfe=1, ncp=self.ncp_t)
        
        list_ln_pert_x = []
        list_ln_phi = []
        
        for pt in range(simulate_points):
            #icc
            norm_x = np.random.rand()*state_norm + 0.01
            random_x = np.zeros((self.num_flatten_var, 1))
            for i in range(self.num_flatten_var):
                random_x[i,0] = np.random.normal(0., 1.)
            norm_random_x = np.linalg.norm(random_x)
            pert_x = norm_x*(random_x/norm_random_x)
            
            n_pert_x = np.linalg.norm(pert_x, "fro")
            if not np.allclose([norm_x],[n_pert_x]):
                raise RuntimeError("assigned norm and calculated norm don't match")
            ln_n_pert_x = np.log(n_pert_x)
            
            uf = -K.dot(pert_x)
            
            count = 0
            for x in self.states:
                xvar = getattr(self.tp_simulate, x)
                x_ic = getattr(self.tp_simulate, x + "_ic")
                for j in self.state_vars[x]:
                    xvar[:,j] = self.tp_state_ss[(x,j)]
                    x_ic[j] = self.tp_state_ss[(x,j)] + pert_x[count,0]
                    count += 1
             
            count = 0
            for u in self.u:
                uvar = getattr(self.tp_simulate, u)
                uvar[:] = self.tp_u_ss[u] + uf[count,0] 
                count += 1
                
                
            self.solve_dyn(self.tp_simulate, iter_max=10, stop_if_nopt=True)
            
            phi_1 = np.zeros((self.num_flatten_var, 1))
            t = t_ij(self.tp_simulate.t, 0, self.ncp_t)
            count = 0
            for x in self.states:
                xvar = getattr(self.tp_simulate, x)
                for j in self.state_vars[x]:
                    phi_1[count,0] = value(xvar[(t,j)]) - self.tp_state_ss[(x,j)]
                    count += 1
            
            Ak = tp_Ad - tp_Bd.dot(K)
            phi_2 = Ak.dot(pert_x)
            
            phi = phi_1 - phi_2
            n_phi = np.linalg.norm(phi, "fro")
            ln_n_phi = np.log(n_phi)
            
            
            list_ln_pert_x.append(ln_n_pert_x)
            list_ln_phi.append(ln_n_phi)
        
        q, lnM = solve_bounded_line(list_ln_pert_x, list_ln_phi)
        
        if plot_figure:
            plt.plot(list_ln_pert_x, list_ln_phi, ".", label = "simulation points")
            # abline(2.2, 3.25, label = "from Devin's thesis")
            abline(q, lnM, label = "bounded line")
            plt.legend()
            plt.ylabel(r'$ln|\phi|$')
            plt.xlabel(r'$ln|x|$')
            plt.title("Simulation for bounding nonlinear part of DAE")
            plt.show()
        
        return q, lnM
        
            
    def add_terminal_property_nmpc(self, rhox, rhou, **kwargs):
        '''
        Calculate the terminal properties, cost and region according to chap 4. in Devin's thesis.
        The algorigm is as follows, 
        
        1. Get the steady states and controls
        2. Linearize the DAE model to get state space matrices A and B 
            with steady states and controls
        3. Discretize the contiunous A and B
        4. Get the cost matrices Q and R for NMPC
        5. Calculate cost matrices tp_Qt and tp_Rt for LQR
        6. Calculate cost-to-go P and gain K of the discrete-time LQR
        7. Simulate the DAE system for one sampling time with many ic of x 
            in the region of interest
        8. Calculate phi = f(x, u=-Kx) - (A-BK)x
        9. Plot ln|phi| vs. ln|x|, get the slope q and the intercept lnM to bound the ln|phi|
        10. Calculate the terminal area, cf, based on equ 4.16

        Parameters
        ----------
        rhox(float): Ratio between Q and tp_Qt.
        rhou(float): Ration between R and tp_Rt.
        simulate_points(float): Number of simulations.
        state_norm(float): Vecotr norm of the perturbed x.

        '''
        
        simulate_points = kwargs.pop("simulate_points", 1000)
        state_norm = kwargs.pop("state_norm", 0.005)
        plot_figure = kwargs.pop("plot_figure", True)
        safety_factor_on_M = kwargs.pop("safety_factor_on_M", False)
        
        #Step 1 ~ 2
        tp_Ac, tp_Bc = self.tp_linearize_A_B()
        tp_Cc = np.identity(self.num_flatten_var, dtype = float)
        tp_Dc = np.zeros((self.num_flatten_var, self.n_controls))
        
        #Step 3
        sysc = control.ss(tp_Ac, tp_Bc, tp_Cc, tp_Dc)
        sysd = control.sample_system(sysc, self.hi_t)
        tp_Ad = sysd.A
        tp_Bd = sysd.B
        tp_Cd = sysd.C
        tp_Dd = sysd.D
        
        #Step 4 ~ 5
        tp_Q_nmpc, tp_R_nmpc = self.tp_build_Q_R()
        tp_Qt = tp_Q_nmpc + rhox*tp_Q_nmpc
        tp_Rt = tp_R_nmpc + rhou*tp_R_nmpc
        
        #Step 6
        K, P, E = dlqr(tp_Ad, tp_Bd, tp_Qt, tp_Rt)
        
        #Step 7 ~ 9
        q, lnM = self.tp_simulation_many_pts(simulate_points, state_norm, tp_Ad, tp_Bd, K, plot_figure)
        
        if q <=1:
            print("Warning: q is smaller than 1 and could yield infinite terminal region!")
            print("Push q to 1.05")
            q = 1.05
        
        #Step 10
        M = np.exp(lnM)
        
        if safety_factor_on_M:
            M = M * 2
        
        Wt = tp_Qt + K.T.dot(tp_Rt).dot(K)
        dQ = tp_Qt - tp_Q_nmpc
        dR = tp_Rt - tp_R_nmpc
        dW = dQ + K.T.dot(dR).dot(K)
        Ak = tp_Ad - tp_Bd.dot(K)
        
        eigval_Wt, _ = np.linalg.eig(Wt)
        lwtmax = max(eigval_Wt)
        eigval_dW, _ = np.linalg.eig(dW)
        ldwmin = min(eigval_dW)
        _,s,_ = np.linalg.svd(Ak)
        sigma = max(s)
        if sigma<0. :
            print("Warning: The maximum singular value of AK is smaller than zero!!")
            print("Push it to 0.1")
            sigma = 0.1
        elif sigma>=1.:
            print("Warning: The maximum singular value of AK is greater than one!!")
            print("Push it to 0.9")
            sigma = 0.9

        lP = (lwtmax)/(1-sigma**2)
        
        cf = ( 
            ((-sigma*lP) + np.sqrt((sigma*lP)**2 + (ldwmin*lP))) / (lP*M) 
            )**(1/(q-1)) #cf is the radius

        area = np.pi * cf**2
        
        print("q=", q)
        print("lnM=", lnM)
        print("cf=", cf)
        print("area=", area)
        
        self.tp_cf = cf
        self.tp_area = area
        self.tp_P = P
        
        self.add_terminal_cost_region()
        
    def add_terminal_cost_region(self, **kwargs):
        
        term_pen_value = kwargs.pop("term_pen", 1000.)
        dict_tp_P = {}
        for i in range(self.num_flatten_var):
            for j in range(self.num_flatten_var):
                # if self.tp_P[i,j] < 1.0E3:
                #     dict_tp_P[(i,j)] = 1.0E3
                # else:
                dict_tp_P[(i,j)] = self.tp_P[i,j]
                
        def _terminal_region():
            expr_tr = 0
            t_end = t_ij(self.olnmpc.t, self.nfe_tnmpc-1, self.ncp_tnmpc)
            for x in self.states:
                xvar = getattr(self.olnmpc, x)
                for j in self.state_vars[x]:
                    expr_tr += (xvar[(t_end,) + j] - self.tp_state_ss[(x, j)])**2   
            return expr_tr
        
        if self.tp_exist:
            self.olnmpc.term_epi.value = 0.
            self.olnmpc.term_pen.value = term_pen_value
            self.olnmpc.term_cf.value = self.tp_cf
            self.olnmpc.term_P.store_values(dict_tp_P)
            
            self.olnmpc.del_component("terminal_region")
            self.olnmpc.terminal_region = Constraint(expr = _terminal_region() - self.olnmpc.term_epi <= self.olnmpc.term_cf**2)
        else:
            self.tp_exist = True
            self.olnmpc.term_epi = Var(initialize = 0., bounds = (0,None))
            self.olnmpc.term_pen = Param(initialize = term_pen_value, mutable = True)
            self.olnmpc.term_cf = Param(initialize = self.tp_cf, mutable = True)
            self.olnmpc.term_P = Param(self.olnmpc.xmpcS_nmpc, self.olnmpc.xmpcS_nmpc, initialize = dict_tp_P, mutable = True)
            
            stage_cost_x_expr  = sum(
                                    sum(self.olnmpc.Q_w_nmpc[fe] * 
                                        self.olnmpc.Q_nmpc[k] * (self.xmpc_l[fe][k] - self.olnmpc.xmpc_ref_nmpc[k])**2
                                        for k in self.olnmpc.xmpcS_nmpc)
                                    for fe in range(0, self.nfe_tnmpc-1)) #take the last element away: self.nfe_tnmpc >> self.nfe.tnmpc-1
            self.olnmpc.stage_cost_x = Expression(expr = stage_cost_x_expr)
            
            fe_end = self.nfe_tnmpc -1
            tc_expr = sum( (self.xmpc_l[fe_end][i] - self.olnmpc.xmpc_ref_nmpc[i])*
                          sum( (self.xmpc_l[fe_end][j] - self.olnmpc.xmpc_ref_nmpc[j]) * self.olnmpc.term_P[i,j] 
                              for j in self.olnmpc.xmpcS_nmpc)
                          for i in self.olnmpc.xmpcS_nmpc)
            self.olnmpc.terminal_cost_x = Expression(expr = tc_expr)
            
            self.olnmpc.terminal_region = Constraint(expr = _terminal_region() - self.olnmpc.term_epi <= self.olnmpc.term_cf**2)
        
        active_obj = []
        for obj in self.olnmpc.component_objects(Objective, active = True):
            active_obj.append(obj.getname())
        if len(active_obj) >= 2:
            raise RuntimeError("More than one objective functions are active in olnmpc!")
        
        obj_target = getattr(self.olnmpc, active_obj[0])
        obj_target.expr = self.olnmpc.stage_cost_x + self.olnmpc.xR_expr_nmpc + self.olnmpc.terminal_cost_x + self.olnmpc.term_pen * self.olnmpc.term_epi
        
    def delete_terminal_properties(self):
        
        self.tp_exist = False
        self.olnmpc.del_component("term_epi")
        self.olnmpc.del_component("term_pen")
        self.olnmpc.del_component("term_cf")
        self.olnmpc.del_component("term_P")
        self.olnmpc.del_component("terminal_region")
        
        obj_target = getattr(self.olnmpc, "objfun_nmpc")
        obj_target.expr = self.olnmpc.xQ_expr_nmpc + self.olnmpc.xR_expr_nmpc
        
        
            
            
    
    
        
            
            
            
        
        
            
        
        
        
                
            
        
        
    
        
        
                
    
                