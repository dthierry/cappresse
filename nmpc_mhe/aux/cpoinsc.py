#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import math as mt

"""written by David M Thierry
roots of orthogonal polynomials
given k collocation points, alpha and beta -> return collocation points of orthogonal polynomials
requires numpy"""

__author__ = 'David M Thierry @dthierry'


def collptsgen(kord, alp, bet):
    b = []
    a = []
    if alp == 1 and bet == 0:
        kord -= 1
        if kord == 0:
            resx = []
            resx.append(1.0)
            return resx
    # reduce order for Radau
    for j in range(0, kord):
        if j == 0 and alp == 0 and bet == 0:
            atemp = 0
            btemp = 0
        elif j == 0 and alp == 1 and bet == 0:
            atemp = ((bet ** 2) - (alp ** 2)) \
                    / ((2 * j + alp + bet) * (2 * j + alp + bet + 2))
            btemp = 0
        else:
            atemp = ((bet**2) - (alp**2))\
                    / ((2*j + alp + bet)*(2*j + alp + bet + 2))
            btemp = (4 * j * (j + alp) * (j + bet) * (j + alp + bet)) \
                    / (
                    (2 * j + alp + bet - 1) * ((2 * j + alp + bet) ** 2) *
                    (2 * j + alp + bet+1))
        a.append(atemp)
        b.append(btemp)

    # print (b, 'a')

    A = np.zeros(shape=(kord, kord))
    # print A
    for j in range(0, kord):
        A[j, j] = a[j]
        if j > 0:
            A[j, j-1] = mt.sqrt(b[j])
        if j < kord-1:
            A[j, j+1] = mt.sqrt(b[j+1])

    # print A

    w,v = np.linalg.eigh(A)
    wactual = w

    for i in range(0, len(w)):
        if abs(w[i]) < 1e-05:
            wactual[i] = 0.0
        # else:
        #     wactual[i] = 0.0

    colpt = []
    colpt = (wactual + 1)/2.

    resx = []
    for i in range(0, kord):
        resx.append(colpt[i])

    # print w
    # print wactual, 'wactual'
    # print colpt

    if alp == 1 and bet == 0:
        resx.append(1.)

    return resx

# results = collcp(self, 2, 0, 0)




# test = Colp()
# print collptsgen(2, 1, 0)

