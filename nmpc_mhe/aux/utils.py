# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from pyomo.dae import ContinuousSet
from pyomo.core.base import Suffix, ConcreteModel
from pyomo.opt import ProblemFormat
from os import getcwd, remove
__author__ = "David Thierry @dthierry" #: March 2018


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
    """Attach Suffixes, and more to a base model"""
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
        d_mod (ConcreteModel): the model of interest

    Returns:
        object:
    """
    if not filename:
        filename = d_mod.name + '.nl'
    d_mod.write(filename, format=ProblemFormat.nl)
    cwd = getcwd()
    print("nl file {}".format(cwd + "/" + filename))
    return cwd


def reconcile_nvars_mequations(d_mod):
    # type: (ConcreteModel) -> tuple
    fullpth = getcwd()
    fullpth += "/_reconcilied.nl"
    write_nl(d_mod, filename=fullpth)
    with open(fullpth, 'r') as nl:
        lines = nl.readlines()
        line = lines[1]
        newl = line.split()
        nvar = int(newl[0])
        meqn = int(newl[1])
        print(newl)
        nl.close()
    remove(fullpth)
    print(nvar, meqn)
    return (nvar, meqn)


