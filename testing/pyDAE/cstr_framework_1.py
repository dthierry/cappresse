#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from sample_mods.cstr_rodrigo.cstr_c import cstr_rodrigo_dae
from nmpc_mhe.pyomo_dae.DynGen_pyDAE import DynGen_DAE


def main():
    states = ["Ca", "T", "Tj"]
    controls = []
    e = DynGen_DAE(cstr_rodrigo_dae, 2, states, controls,
               k_aug_executable="/home/dav0/k_aug/src/k_aug/k_aug",
               dot_driver_executable="/home/dav0/k_aug/src/k_aug/dot_driver/dot_driver")
    print(type(e.PlantSample))
    e.PlantSample.pprint()
    return e


if __name__ == '__main__':
    e = main()
    e.create_dyn()
    e.solve_dyn(e.dyn)
    e.get_state_vars()


