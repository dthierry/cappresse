# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from pyomo.dae import ContinuousSet, DerivativeVar
from pyomo.core.base import Suffix, ConcreteModel, Var, Suffix, Constraint, TransformationFactory, ConstraintList
from pyomo.dae import *
from pyomo.opt import ProblemFormat
from pyomo.core.kernel.numvalue import value
from os import getcwd, remove

__author__ = "David Thierry @dthierry"  #: March 2018


def t_ij(time_set, i, j):
    # type: (ContinuousSet, int, int) -> float
    """Return the corresponding time(continuous set) based on the i-th finite element and j-th collocation point

    Args:
        time_set (ContinuousSet): Parent Continuous set
        i (int): finite element
        j (int): collocation point

    Returns:
        float: Corresponding index of the ContinuousSet
    """
    h = time_set.get_finite_elements()[1] - time_set.get_finite_elements()[0]  #: This would work even for 1 fe
    tau = time_set.get_discretization_info()['tau_points']
    fe = time_set.get_finite_elements()[i]
    time = fe + tau[j] * h
    return round(time, 6)


def fe_cp(time_set, t):
    # type: (ContinuousSet, float) -> tuple
    """Return the corresponding fe and cp for a given time

    Args:
        time_set:
        t:
    """
    fe_l = time_set.get_lower_element_boundary(t)
    print("fe_l", fe_l)
    fe = None
    j = 0
    for i in time_set.get_finite_elements():
        if fe_l == i:
            fe = j
            break
        j += 1
    h = time_set.get_finite_elements()[1] - time_set.get_finite_elements()[0]
    tauh = [i * h for i in time_set.get_discretization_info()['tau_points']]
    j = 0  #: Watch out for LEGENDRE
    cp = None
    for i in tauh:
        if round(i + fe_l, 6) == t:
            cp = j
            break
        j += 1
    return (fe, cp)


def fe_compute(time_set, t):
    # type: (ContinuousSet, float) -> int
    """Return the corresponding fe given time
    Args:
        time_set:
        t:
    """
    fe_l = time_set.get_lower_element_boundary(t)

    fe = int()
    j = 0
    for i in time_set.get_finite_elements():
        if fe_l == i:
            fe = j
            break
        j += 1
    if t > 0 and t in time_set.get_finite_elements():
        fe += 1
    return fe


def augment_model(d_mod, nfe, ncp, new_timeset_bounds=None, given_name=None, skip_suffixes=False):
    """Attach Suffixes, and more to a base model

    Args:
        d_mod(ConcreteModel): Model of interest.
    """
    if hasattr(d_mod, "nfe") or hasattr(d_mod, "ncp"):
        print('Warning: redefining nfe and ncp')

    d_mod.nfe_t = nfe
    d_mod.ncp_t = ncp

    if hasattr(d_mod, 'is_steady'):
        #: keep it steady
        if d_mod.is_steady:
            pass
    else:
        #: steady is false by default
        d_mod.is_steady = False

    if skip_suffixes:
        pass
    else:
        d_mod.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
        d_mod.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
        d_mod.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
        d_mod.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
        d_mod.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)
    if not new_timeset_bounds is None:
        cs = None
        for s in d_mod.component_objects(ContinuousSet):
            cs = s
        if cs is None:
            raise RuntimeError("The model has no ContinuousSet")
        if not isinstance(new_timeset_bounds, tuple):
            raise RuntimeError("new_timeset_bounds should be tuple = (t0, tf)")
        cs._bounds = new_timeset_bounds
        cs.clear()
        cs.construct()

        for o in d_mod.component_objects([Var, DerivativeVar, Constraint]):
            #: This series of if conditions are in place to avoid some weird behaviour
            if o._implicit_subsets is None:
                if o.index_set() is cs:
                    pass
                else:
                    o.reconstruct()
                    continue
            else:
                if cs in o._implicit_subsets:
                    pass
                else:
                    o.reconstruct()
                    continue
            o.clear()
            o.construct()
            o.reconstruct()

    if isinstance(given_name, str):
        d_mod.name = given_name


def write_nl(d_mod, filename=None, labels=False):
    # type: (ConcreteModel, str) -> str
    """
    Write the nl file
    Args:
        d_mod (ConcreteModel): the model of interest.

    Returns:
        cwd (str): The current working directory.
    """
    if not filename:
        filename = d_mod.name + '.nl'
    d_mod.write(filename, format=ProblemFormat.nl, io_options={'symbolic_solver_labels': labels})
    cwd = getcwd()
    print("nl file {}".format(cwd + "/" + filename))
    return cwd


def reconcile_nvars_mequations(d_mod, keep_nl=False, **kwargs):
    # type: (ConcreteModel) -> tuple
    """
    Compute the actual number of variables and equations in a model by reading the relevant line at the nl file.
    Args:
        d_mod (ConcreteModel):  The model of interest

    Returns:
        tuple: The number of variables and the number of constraints.

    """
    fullpth = getcwd()
    fullpth += "/_reconcilied.nl"
    write_nl(d_mod, filename=fullpth, **kwargs)
    with open(fullpth, 'r') as nl:
        lines = nl.readlines()
        line = lines[1]
        newl = line.split()
        nvar = int(newl[0])
        meqn = int(newl[1])
        nl.close()
    if keep_nl:
        pass
    else:
        remove(fullpth)
    return (nvar, meqn)


def load_iguess(src, tgt, fe_src, fe_tgt):
    # type: (ConcreteModel, ConcreteModel, int, int) -> None
    """Loads the current values of the src model into the tgt model, i.e. src-->tgt.
    This will assume that the time set is always at the beginning.

    Args:
        src (ConcreteModel): Model with the source values.
        tgt (ConcreteModel): Model with the target variables.
        fe_src (int): Source finite element.
        fe_tgt (int): Target finite element.

    Returns:
        None:
    """
    uniform_mode = True
    steady = False
    if src.name == "unknown" or None:
        pass
    elif src.name == "SteadyRef":
        #: If we use a steay-state model we have to change the strategy
        print("Steady!")
        fe_src = 1
        steady = True
    fe0_src = getattr(src, "nfe_t")
    fe0_tgt = getattr(tgt, "nfe_t")
    print("fetgt", fe0_tgt, fe_tgt)
    if fe_src > fe0_src - 1:
        if steady:
            pass
        else:
            raise KeyError("Finite element beyond maximum: src")
    if fe_tgt > fe0_tgt - 1:
        raise KeyError("Finite element beyond maximum: tgt")

    cp_src = getattr(src, "ncp_t")
    cp_tgt = getattr(tgt, "ncp_t")
    #: Continuous time set
    tS_src = getattr(src, "t")
    tS_tgt = getattr(tgt, "t")

    if cp_src != cp_tgt:
        print("These variables do not have the same number of Collocation points (ncp_t)")
        # raise UnexpectedOption("These variables do not have the same number of Collocation points (ncp_t)")
        uniform_mode = False

    if steady:
        for vs in src.component_objects(Var, active=True):
            if vs._implicit_subsets is None:
                if vs.index_set() is tS_src:
                    vd = getattr(tgt, vs.getname())
                    for j in range(0, cp_tgt + 1):
                        t_tgt = t_ij(tS_tgt, fe_tgt, j)
                        vd[t_tgt].set_value(value(vs[1]))
                else:
                    continue
            else:
                if not tS_src in vs._implicit_subsets:
                    continue
                else:
                    vd = getattr(tgt, vs.getname())
                    remaining_set = vs._implicit_subsets[1]
                    for j in range(2, len(vs._implicit_subsets)):
                        remaining_set *= vs._implicit_subsets[j]
                    for index in remaining_set:
                        for j in range(0, cp_tgt + 1):
                            # t_src = 1
                            t_tgt = t_ij(tS_tgt, fe_tgt, j)
                            index = index if isinstance(index, tuple) else (index,)  #: Transform to tuple
                            vd[(t_tgt,) + index].set_value(value(vs[(1,) + index]))
    elif uniform_mode:
        for vs in src.component_objects(Var, active=True):
            if vs._implicit_subsets is None:
                if vs.index_set() is tS_src:
                    vd = getattr(tgt, vs.getname())
                    for j in range(0, cp_src + 1):
                        t_src = t_ij(tS_src, fe_src, j)
                        t_tgt = t_ij(tS_tgt, fe_tgt, j)
                        vd[t_tgt].set_value(value(vs[t_src]))
                else:
                    continue
            else:
                if not tS_src in vs._implicit_subsets:
                    continue
                else:
                    vd = getattr(tgt, vs.getname())
                    remaining_set = vs._implicit_subsets[1]
                    for j in range(2, len(vs._implicit_subsets)):
                        remaining_set *= vs._implicit_subsets[j]
                    for index in remaining_set:
                        for j in range(0, cp_src + 1):
                            t_src = t_ij(tS_src, fe_src, j)
                            t_tgt = t_ij(tS_tgt, fe_tgt, j)
                            index = index if isinstance(index, tuple) else (index,)  #: Transform to tuple
                            vd[(t_tgt,) + index].set_value(value(vs[(t_src,) + index]))
    else:
        for vs in src.component_objects(Var, active=True):
            if vs._implicit_subsets is None:
                if vs.index_set() is tS_src:
                    vd = getattr(tgt, vs.getname())
                    for j in range(0, cp_tgt + 1):
                        t_src = t_ij(tS_src, fe_src, cp_src)  #: only patch the last value
                        t_tgt = t_ij(tS_tgt, fe_tgt, j)
                        #: Better idea: interpolate
                        vd[t_tgt].set_value(value(vs[t_src]))
                else:
                    continue
            else:
                if not tS_src in vs._implicit_subsets:
                    continue
                else:
                    vd = getattr(tgt, vs.getname())
                    remaining_set = vs._implicit_subsets[1]
                    for j in range(2, len(vs._implicit_subsets)):
                        remaining_set *= vs._implicit_subsets[j]
                    for index in remaining_set:
                        for j in range(0, cp_tgt + 1):
                            t_src = t_ij(tS_src, fe_src, cp_src)  #: only patch the last value
                            t_tgt = t_ij(tS_tgt, fe_tgt, j)
                            index = index if isinstance(index, tuple) else (index,)  #: Transform to tuple
                            #: Better idea: interpolate
                            vd[(t_tgt,) + index].set_value(value(vs[(t_src,) + index]))


def augment_steady(dmod):
    cs = None
    for s in dmod.component_objects(ContinuousSet):
        cs = s
    if cs is None:
        raise RuntimeError("The model has no ContinuousSet")
    #: set new bounds on the time set
    augment_model(dmod, 1, 1, new_timeset_bounds=(0,1))
    dv_list = []
    for dv in dmod.component_objects(DerivativeVar):
        dv_list.append(dv.name)  #: We have the differential variables

    #: Search for collocation equations
    dae = TransformationFactory('dae.collocation')
    dae.apply_to(dmod, nfe=1, ncp=1)

    #: Deactivate collocation constraint
    for dv in dv_list:
        col_con = getattr(dmod, dv + "_disc_eq")
        col_con.deactivate()
        #: Check whether we need icc cons
        if hasattr(dmod, dv + "_icc"):
            icc_con = getattr(dmod, dv + "_icc")
            icc_con.deactivate()
    dmod.add_component("dvs_steady", ConstraintList())
    clist = getattr(dmod, "dvs_steady")
    for dv in dv_list:
        dvar = getattr(dmod, dv)
        for key in dvar.keys():
            clist.add(dvar[key] == 0)
    if hasattr(dmod, 'name'):
        pass
    if hasattr(dmod, 'is_steady'):
        dmod.is_steady = True



def aug_discretization(d_mod, nfe, ncp):
    collocation = TransformationFactory("dae.collocation")
    collocation.apply_to(d_mod, nfe=nfe, ncp=ncp, scheme="LAGRANGE-RADAU")



def create_bounds(d_mod, bounds=None, clear=False, pre_clear_check=True):
    if pre_clear_check:
        for i in d_mod.component_data_objects(Var):
            i.setlb(None)
            i.setub(None)
    if bounds is None:
        return
    elif isinstance(bounds, dict):
        print("Model: {} Bounds activated".format(d_mod.name))
        for var_name in bounds.keys():
            try:
                var = getattr(d_mod, var_name)
            except AttributeError:
                print("The variable {} is not part of the model.".format(var_name))
                raise RuntimeError("Error in the bounds dictionary")
            if not isinstance(bounds[var_name], tuple):
                raise RuntimeError("The value for {} key is not tuple; all values must be tuples (None, None)")
            for i in var.keys():
                if clear:
                    var[i].setlb(None)
                    var[i].setub(None)
                else:
                    var[i].setlb(bounds[var_name][0])
                    var[i].setub(bounds[var_name][1])
    else:
        raise RuntimeWarning("bounds is of type {} and it should be of type dict, no bounds declared".format(type(bounds)))


def clone_the_model(d_mod):
    src_id = id(d_mod)
    new_mod = d_mod.clone()
    nm_id = id(new_mod)
    assert(src_id != nm_id)
    print("New model at {}".format(nm_id))
    return new_mod
