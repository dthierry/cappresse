# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division
from pyomo.dae import ContinuousSet
from pyomo.core.base import Suffix

__author__ = "David M Thierry"  #: @march-2018


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
    """Return the corresponding fe and cp for a given time"""
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


def augment_model(d_mod):
    """Attach Suffixes, and more to a base model"""
    d_mod.dual = Suffix(direction=Suffix.IMPORT_EXPORT)
    d_mod.ipopt_zL_out = Suffix(direction=Suffix.IMPORT)
    d_mod.ipopt_zU_out = Suffix(direction=Suffix.IMPORT)
    d_mod.ipopt_zL_in = Suffix(direction=Suffix.EXPORT)
    d_mod.ipopt_zU_in = Suffix(direction=Suffix.EXPORT)

