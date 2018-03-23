#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from sample_mods.cstr_rodrigo.cstr_c_nmpc import cstr_rodrigo_dae
from nmpc_mhe.pyomo_dae.NMPCGen_pyDAE import NmpcGen_DAE
from nmpc_mhe.aux.utils import reconcile_nvars_mequations

__author__ = "David Thierry @dthierry" #: March 2018

def main():
    states = ["Ca", "T", "Tj"]
    controls = ["u1"]
    u_bounds = {"u1": (0, 1000)}
    e = NmpcGen_DAE(cstr_rodrigo_dae, 2, states, controls, u_bounds=u_bounds,
               k_aug_executable="/home/dav0/k_aug/src/k_aug/k_aug",
               dot_driver_executable="/home/dav0/k_aug/src/k_aug/dot_driver/dot_driver")
    return e


if __name__ == '__main__':
    e = main()
    e.get_state_vars()
    e.create_nmpc()
    # e.olnmpc.pprint()
    reconcile_nvars_mequations(e.olnmpc)
