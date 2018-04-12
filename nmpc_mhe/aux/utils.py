# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from pyomo.dae import ContinuousSet
from pyomo.core.base import Suffix, ConcreteModel, Var, Suffix
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


def augment_model(d_mod):
    """Attach Suffixes, and more to a base model

    Args:
        d_mod(ConcreteModel): Model of interest.
    """
    d_mod.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    d_mod.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    d_mod.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    d_mod.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    d_mod.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)


def write_nl(d_mod, filename=None):
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
    d_mod.write(filename, format=ProblemFormat.nl)
    cwd = getcwd()
    print("nl file {}".format(cwd + "/" + filename))
    return cwd


def reconcile_nvars_mequations(d_mod):
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
    write_nl(d_mod, filename=fullpth)
    with open(fullpth, 'r') as nl:
        lines = nl.readlines()
        line = lines[1]
        newl = line.split()
        nvar = int(newl[0])
        meqn = int(newl[1])
        nl.close()
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
            if not vs._implicit_subsets:
                continue
            if tS_src not in vs._implicit_subsets:
                continue
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
            if not vs._implicit_subsets:
                continue
            if tS_src not in vs._implicit_subsets:
                continue
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
                    # print(vd.getname(), t_tgt, value(vd[(t_tgt,) + index]))
    else:
        for vs in src.component_objects(Var, active=True):
            if not vs._implicit_subsets:
                continue
            if not vs._implicit_subsets:
                continue
            if tS_src not in vs._implicit_subsets:
                continue
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
