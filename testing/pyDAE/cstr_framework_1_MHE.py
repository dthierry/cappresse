#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
from pyomo.environ import *
from sample_mods.cstr_rodrigo.cstr_c_nmpc import cstr_rodrigo_dae
from nmpc_mhe.pyomo_dae.MHEGen_pyDAE import MheGen_DAE
from nmpc_mhe.aux.utils import load_iguess
from nmpc_mhe.aux.utils import reconcile_nvars_mequations
import matplotlib.pyplot as plt

__author__ = "David Thierry @dthierry" #: March 2018

def main():
    states = ["Ca", "T", "Tj"]
    measurements = ['T']
    controls = ["u1"]
    u_bounds = {"u1": (0, 1000)}
    ref_state = {("Ca", (0,)): 0.010}
    e = MheGen_DAE(cstr_rodrigo_dae, 2, states, controls, states, measurements, u_bounds=u_bounds, ref_state=ref_state,
                   override_solver_check=True)
    return e


if __name__ == '__main__':
    e = main()
    e.get_state_vars()
    e.load_iguess_steady()
    load_iguess(e.SteadyRef, e.PlantSample, 0, 0)
    e.lsmhe.pprint()