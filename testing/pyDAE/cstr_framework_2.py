#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from sample_mods.cstr_rodrigo.cstr_c import cstr_rodrigo_dae
from nmpc_mhe.pyomo_dae.DynGen_pyDAE import DynGen_DAE, load_iguess

__author__ = "David Thierry @dthierry" #: March 2018

def main():
    states = ["Ca", "T", "Tj"]
    controls = []
    model = cstr_rodrigo_dae(1, 1)
    e = DynGen_DAE(model, 2, states, controls)
    return e


if __name__ == '__main__':
    e = main()
    e.create_dyn()
    e.solve_dyn(e.dyn)
    e.get_state_vars()
    print(e.state_vars)
