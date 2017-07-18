#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

from pyomo.core.base import Var, Objective, minimize, value, Set, Constraint, Expression, Param, Suffix
from pyomo.opt import SolverFactory, ProblemFormat, SolverStatus, TerminationCondition
from nmpc_mhe.dync.DynGen import DynGen
import numpy as np
import sys

__author__ = "David M Thierry @dthierry"
"""Not yet."""


class MheGen(DynGen):
    def __init__(self):
        DynGen.__init__(self)


