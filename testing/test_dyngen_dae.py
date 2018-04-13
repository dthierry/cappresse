# !/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from pyomo.environ import *
from sample_mods.cstr_rodrigo.cstr_c_nmpc import cstr_rodrigo_dae
from nmpc_mhe.pyomo_dae.DynGen_pyDAE import DynGen_DAE
import unittest, os

__author__ = "David M Thierry @dthierry"  #: April 2018
__copyright__ = "Copyright (C) 2018 David Thierry"

class TestDyngenMethods(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dyngen = DynGen_DAE(cstr_rodrigo_dae, 1, ['Ca', 'T', 'Tj'], ['u1'])

    @classmethod
    def tearDownClass(cls):
        os.remove("res_dyn_label_" + cls.dyngen.res_file_suf + ".txt")
        os.remove("ipopt.opt")

    def test_create_dyn(self):
        self.dyngen.create_dyn()
        self.assertIsInstance(self.dyngen.dyn, ConcreteModel)


    def test_get_state_vars(self):
        self.dyngen.get_state_vars()
        for i in self.dyngen.states:
            self.assertIn(i, self.dyngen.state_vars)


if __name__ == '__main__':
    unittest.main()
