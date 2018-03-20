# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, Set, Constraint, Expression, Param, Suffix
from pyomo.core.kernel.numvalue import value
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from nmpc_mhe.dync.DynGen_pyDAE import DynGen_DAE
import numpy as np
import sys, os, time
from six import iterkeys
__author__ = "David M Thierry @dthierry"

"""This version does not necesarily have the same time horizon/discretization as the MHE"""


class NmpcGen_DAE(DynGen_DAE):
    def __init__(self, d_mod, hi_t, states, controls, **kwargs):
        DynGen.__init__(self, d_mod, hi_t, states, controls, **kwargs)
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

        # self.res_file_name = "res_nmpc_" + str(int(time.time())) + ".txt"

    def create_nmpc(self, **kwargs):
        kwargs.pop("newnfe", self.nfe_tnmpc)
        kwargs.pop("newncp", self.ncp_tnmpc)
        self.journalist('W', self._iteration_count, "Initializing NMPC",
                        "With {:d} fe and {:d} cp".format(self.nfe_tnmpc, self.ncp_tnmpc))
        _tnmpc = self.hi_t * self.nfe_tnmpc
        self.olnmpc = self.d_mod(self.nfe_tnmpc, self.ncp_tnmpc, _t=_tnmpc)
        self.olnmpc.name = "olnmpc (Open-Loop NMPC)"
        self.olnmpc.create_bounds()

        for u in self.u:
            cv = getattr(self.olnmpc, u)  #: Get the param
            c_val = [value(cv[i]) for i in cv.keys()]  #: Current value
            self.olnmpc.del_component(cv)  #: Delete the param
            self.olnmpc.add_component(u, Var(self.olnmpc.fe_t, initialize=lambda m, i: c_val[i-1]))
            self.olnmpc.equalize_u(direction="r_to_u")
            cc = getattr(self.olnmpc, u + "_c")  #: Get the constraint
            ce = getattr(self.olnmpc, u + "_e")  #: Get the expression
            cv = getattr(self.olnmpc, u)  #: Get the new variable
            for k in cv.keys():
                cv[k].setlb(self.u_bounds[u][0])
                cv[k].setub(self.u_bounds[u][1])
            cc.clear()
            cc.rule = lambda m, i: cv[i] == ce[i]
            cc.reconstruct()
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
            for j in self.state_vars[x]:
                self.xmpc_l[0].append(n_s[(0, self.ncp_tnmpc) + j])
                self.xmpc_key[(x, j)] = k
                k += 1
        #: Iterate over the rest
        for t in range(1, self.nfe_tnmpc):
            self.xmpc_l[t] = []
            for x in self.states:
                n_s = getattr(self.olnmpc, x)  #: State
                for j in self.state_vars[x]:
                    self.xmpc_l[t].append(n_s[(t, self.ncp_tnmpc) + j])
        #: A set with the length of flattened states
        self.olnmpc.xmpcS_nmpc = Set(initialize=[i for i in range(0, len(self.xmpc_l[0]))])
        #: Create set of noisy_states
        self.olnmpc.xmpc_ref_nmpc = Param(self.olnmpc.xmpcS_nmpc, initialize=0.0, mutable=True)  #: Ref-state
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
        self.olnmpc.R_nmpc = Param(self.olnmpc.umpcS_nmpc, initialize=1, mutable=True)  #: Control-weight
        #: Build the uT * R * u expression
        self.olnmpc.xR_expr_nmpc = Expression(expr=sum(
            sum(self.olnmpc.R_w_nmpc[fe] *
                self.olnmpc.R_nmpc[k] * (self.umpc_l[fe][k] - self.olnmpc.umpc_ref_nmpc[k]) ** 2 for k in
                self.olnmpc.umpcS_nmpc)
            for fe in range(0, self.nfe_tnmpc)))
        self.olnmpc.objfun_nmpc = Objective(expr=self.olnmpc.xQ_expr_nmpc + self.olnmpc.xR_expr_nmpc)

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
        dum = self.d_mod(1, self.ncp_tnmpc, _t=self.hi_t)
        dum.create_bounds()
        #: Load current solution
        self.load_iguess_single(ref, dum, 0, 0)

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
            self.load_iguess_dyndyn(dum, self.olnmpc, finite_elem)


            for u in self.u:
                cv_nmpc = getattr(self.olnmpc, u)  #: set controls for open-loop nmpc
                cv_dum = getattr(dum, u)
                # works only for fe_t index
                cv_nmpc[finite_elem].set_value(value(cv_dum[0]))
        self.journalist("I", self._iteration_count, "initialize_olnmpc", "Done, k_notopt " + str(k_notopt))

    def preparation_phase_nmpc(self, as_strategy=False, make_prediction=False, plant_state=False):
        """By default we assume mhe is providing the state"""
        if plant_state:
            #: use the plant state instead
            #: Not yet implemented
            return None
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
            self.initialize_olnmpc(self.PlantSample, "estimated")
            self.load_init_state_nmpc(src_kind="state_dict", state_dict="estimated")

    def load_init_state_nmpc(self, src_kind, **kwargs):
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
            for x in self.states:
                xic = getattr(self.olnmpc, x + "_ic")
                xvar = getattr(self.olnmpc, x)
                xsrc = getattr(ref, x)
                for j in self.state_vars[x]:
                    xic[j].value = value(xsrc[(fe, cp) + j])
                    xvar[(0, 0) + j].set_value(value(xsrc[(fe, cp) + j]))
        else:
            state_dict = kwargs.pop("state_dict", None)
            if state_dict == "real":  #: Load from the real state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_rstate[(x, j)]
                        xvar[(0, 0) + j].set_value(self.curr_rstate[(x, j)])
            elif state_dict == "estimated":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        xic[j].value = self.curr_estate[(x, j)]
                        xvar[(0, 0) + j].set_value(self.curr_estate[(x, j)])
            elif state_dict == "predicted":  #: Load from the estimated state dict
                for x in self.states:
                    xic = getattr(self.olnmpc, x + "_ic")
                    xvar = getattr(self.olnmpc, x)
                    for j in self.state_vars[x]:
                        try:
                            xic[j].value = self.curr_pstate[(x, j)]
                        except KeyError:
                            print(self.curr_pstate)
                        xvar[(0, 0) + j].set_value(self.curr_pstate[(x, j)])
            else:
                self.journalist("W", self._iteration_count, "load_init_state_nmpc", "No dict w/state was specified")
                self.journalist("W", self._iteration_count, "load_init_state_nmpc", "No update on state performed")
                raise ValueError("Unexpeted state_dict %s" % state_dict)

    def compute_QR_nmpc(self, src="plant", n=-1, **kwargs):
        """Using the current state & control targets, computes the Qk and Rk matrices (diagonal)
        Strategy: take the inverse of the absolute difference between reference and current state such that every
        offset in the objective is normalized or at least dimensionless
        Args:
            src (str): The source of the update (default mhe) (mhe or plant)
            n (int): The exponent of the weight"""
        check_values = kwargs.pop("check_values", False)
        if check_values:
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

    def new_weights_olnmpc(self, state_weight, control_weight):
        """Change the weights associated with the control objective function"""
        # if type(state_weight) == float:
        #     for fe in self.olnmpc.fe_t:
        #         self.olnmpc.Q_w_nmpc[fe].value = state_weight
        #     self.olnmpc.Q_w_nmpc.display()
        #     print("float")
        #     sys.exit()
        if type(state_weight) == dict:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[fe].value = state_weight[fe]
            # self.olnmpc.Q_w_nmpc.display()
            # print("dict")
            # sys.exit()
        else:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.Q_w_nmpc[fe].value = state_weight


        # if type(control_weight) == float:
            # for fe in self.olnmpc.fe_t:
            #     self.olnmpc.R_w_nmpc[fe].value = control_weight
        if type(control_weight) == dict:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.R_w_nmpc[fe].value = control_weight[fe]
        else:
            for fe in self.olnmpc.fe_t:
                self.olnmpc.R_w_nmpc[fe].value = control_weight

        # with open("your_weight.txt", "w") as f:
        #     self.olnmpc.R_w_nmpc.pprint(ostream=f)
        #     f.close()

    def create_suffixes_nmpc(self):
        """Creates the required suffixes for the olnmpc problem"""
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

        for x in self.states:
            con_name = x + "_icc"
            con_ = getattr(self.olnmpc, con_name)
            for j in self.state_vars[x]:
                con_[j].set_suffix_value(self.olnmpc.npdp, self.curr_state_offset[(x, j)])

        if hasattr(self.olnmpc, "f_timestamp"):
            self.olnmpc.f_timestamp.clear()
        else:
            self.olnmpc.f_timestamp = Suffix(direction=Suffix.EXPORT,
                                            datatype=Suffix.INT)
        self.olnmpc.set_suffix_value(self.olnmpc.f_timestamp, self.int_file_nmpc_suf)

        self.olnmpc.f_timestamp.display(ostream=sys.stderr)

        self.journalist("I", self._iteration_count, "sens_dot_nmpc", self.olnmpc.name)

        results = self.dot_driver.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)

        ftiming = open("timings_dot_driver.txt", "r")
        s = ftiming.readline()
        ftiming.close()
        k = s.split()
        self._dot_timing = k[0]

    def sens_k_aug_nmpc(self):
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
        results = self.k_aug_sens.solve(self.olnmpc, tee=True, symbolic_solver_labels=True)
        self.olnmpc.solutions.load_from(results)
        self.olnmpc.f_timestamp.display(ostream=sys.stderr)
        ftimings = open("timings_k_aug.txt", "r")
        s = ftimings.readline()
        ftimings.close()
        self._k_timing = s.split()

    def stall_strategy(self, strategy, cmv=1e-04, **kwargs):  # Fix the damn stall strategy
        """Suggested three strategies: Change weights, change matrices, change linear algebra"""
        self._stall_iter += 1
        self.journalist("I", self._iteration_count, "stall_strategy", "Solver Stalled. " + str(self._stall_iter) + " Times")
        if strategy == "increase_weights":
            spf = 0
            ma57_as = "no"
            sw = self.olnmpc.s_w
            cw = self.olnmpc.c_w
            sw.value += sw.value
            cw.value += cw.value
            if sw.value > 1e+06 or cw.value > 1e+06:
                return 1
        elif strategy == "recompute_matrices":
            cmv += 1e+04 * 5
            self.load_qk(max_qval=cmv)
        elif strategy == "linear_algebra":
            spf = 1
            ma57_as = "yes"

        retval = self.solve_dyn(self.olnmpc, max_cpu_time=300,
                              small_pivot_flag=spf,
                              ma57_automatic_scaling=ma57_as,
                              want_stime=True,
                              rep_timing=True)
        if retval == 0:
            return 0
        else:
            if self._stall_iter > 10:
                self.journalist("I", self._iteration_count, "stall_strategy",
                                 "Max number of tries reached")
                sys.exit()
            self.stall_strategy("increase_weights")

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
        for i in self.ref_state.keys():
            v = getattr(self.SteadyRef, i[0])
            vkey = i[1]
            vss0 = value(v[(0, 1) + vkey])
            val = abs(self.ref_state[i] - vss0)
            if val < 1e-09:
                val = 1e+06
            else:
                val = 1/val
            weights_ref[i] = val

        weights = kwargs.pop("weights", weights_ref)

        self.journalist("I", self._iteration_count, "find_target_ss", "Attempting to find steady state")

        del self.SteadyRef2
        self.SteadyRef2 = self.d_mod(1, 1, steady=True)
        self.SteadyRef2.name = "SteadyRef2 (reference)"
        for u in self.u:
            cv = getattr(self.SteadyRef2, u)  #: Get the param
            c_val = [value(cv[i]) for i in cv.keys()]  #: Current value
            self.SteadyRef2.del_component(cv)  #: Delete the param
            self.SteadyRef2.add_component(u, Var(self.SteadyRef2.fe_t, initialize=lambda m, i: c_val[i-1]))
            self.SteadyRef2.equalize_u(direction="r_to_u")
            cc = getattr(self.SteadyRef2, u + "_c")  #: Get the constraint
            ce = getattr(self.SteadyRef2, u + "_e")  #: Get the expression
            cv = getattr(self.SteadyRef2, u)  #: Get the new variable
            for k in cv.keys():
                cv[k].setlb(self.u_bounds[u][0])
                cv[k].setub(self.u_bounds[u][1])
            cc.clear()
            cc.rule = lambda m, i: cv[i] == ce[i]
            cc.reconstruct()

        self.SteadyRef2.create_bounds()
        self.SteadyRef2.equalize_u(direction="r_to_u")

        for vs in self.SteadyRef.component_objects(Var, active=True):  #: Load_guess
            vt = getattr(self.SteadyRef2, vs.getname())
            for ks in vs.keys():
                vt[ks].set_value(value(vs[ks]))
        ofexp = 0
        for i in self.ref_state.keys():
            v = getattr(self.SteadyRef2, i[0])
            val = value((v[(0, 1) + vkey]))
            vkey = i[1]
            ofexp += weights[i] * (v[(0, 1) + vkey] - self.ref_state[i])**2
            # ofexp += -weights[i] * (v[(1, 1) + vkey])**2 #- self.ref_state[i])**2
        self.SteadyRef2.obfun_SteadyRef2 = Objective(expr=ofexp, sense=minimize)

        tst = self.solve_dyn(self.SteadyRef2, iter_max=10000, stop_if_nopt=True, halt_on_ampl_error=False, **kwargs)
        # self.SteadyRef2.write_nl(name="steady.nl")
        # self.SteadyRef2.write_nl()
        # self.SteadyRef2.snap_shot(filename="mom.py")
        # sys.exit()

        if tst != 0:
            self.SteadyRef2.display(filename="failed_SteadyRef2.txt")
            self.SteadyRef2.write(filename="failed_SteadyRef2.nl",
                           format=ProblemFormat.nl,
                           io_options={"symbolic_solver_labels": True})
            # sys.exit(-1)
        self.journalist("I", self._iteration_count, "find_target_ss", "Target: solve done")
        for i in self.ref_state.keys():
            v = getattr(self.SteadyRef2, i[0])
            vkey = i[1]
            val = value(v[(0, 1) + vkey])
            print("target {:}".format(i[0]), "key {:}".format(i[1]), "weight {:f}".format(weights[i]),
                  "value {:f}".format(val))
        for u in self.u:
            v = getattr(self.SteadyRef2, u)
            val = value(v[0])
            print("target {:}".format(u), " value {:f}".format(val))
        self.update_targets_nmpc()

    def update_targets_nmpc(self):
        """Use the reference model to update  the current state and control targets dictionaries"""
        for x in self.states:
            xvar = getattr(self.SteadyRef2, x)
            for j in self.state_vars[x]:
                self.curr_state_target[(x, j)] = value(xvar[0, 1, j])
        for u in self.u:
            uvar = getattr(self.SteadyRef2, u)
            self.curr_u_target[u] = value(uvar[0])

    def change_setpoint(self, ref_state, **kwargs):
        """Change the update the ref_state dictionary, and attempt to find a new reference state"""
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
            vss0 = value(v[(0, 1) + vkey])
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
            ofexp += weights[i] * (v[(0, 1) + vkey] - self.ref_state[i]) ** 2

        self.SteadyRef2.obfun_SteadyRef2.set_value(ofexp)
        self.solve_dyn(self.SteadyRef2, iter_max=500, stop_if_nopt=True, **kwargs)

        for i in self.ref_state.keys():
            v = getattr(self.SteadyRef2, i[0])
            vkey = i[1]
            val = value(v[(0, 1) + vkey])
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
        self.journalist("I", self._iteration_count, "print_r_nmpc", "Results at" + os.getcwd())
        self.journalist("I", self._iteration_count, "print_r_nmpc", "Results suffix " + self.res_file_suf)
        # print(self.soi_dict)
        for k in self.ref_state.keys():
            self.soi_dict[k].append(self.curr_soi[k])
            self.sp_dict[k].append(self.curr_sp[k])
            print("Current values\t", self.ref_state[k], k)

        # for u in self.u:
        #     self.u_dict[u].append(self.curr_u[u])
        #     print(self.curr_u[u])

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
        # with open("res_nmpc_u_" + self.res_file_suf + ".txt", "a") as f:
        #     for u in self.u:
        #         for i in range(0, len(self.u_dict[u])):
        #             iv = str(self.u_dict[u][i])
        #             f.write(iv)
        #             f.write('\t')
        #         f.write('\n')
        #     f.close()

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
            self.curr_soi[k] = value(var[(0, self.ncp_t) + vkey])
        for k in self.ref_state.keys():
            vname = k[0]
            vkey = k[1]
            var = getattr(self.SteadyRef2, vname)
            #: Assuming the variable is indexed by time
            self.curr_sp[k] = value(var[(0, 1) + vkey])
        self.journalist("I", self._iteration_count, "update_soi_sp_nmpc", "Current offsets + Values:")
        for k in self.ref_state.keys():
            #: Assuming the variable is indexed by time
            self.curr_off_soi[k] = 100 * abs(self.curr_soi[k] - self.curr_sp[k])/abs(self.curr_sp[k])
            print("\tCurrent offset \% \% \t", k, self.curr_off_soi[k], end="\t")
            print("\tCurrent value \% \% \t", self.curr_soi[k])


        for u in self.u:
            ur = getattr(self.SteadyRef2, u)
            self.curr_ur[u] = value(ur[0])

    def method_for_nmpc_simulation(self):
        pass