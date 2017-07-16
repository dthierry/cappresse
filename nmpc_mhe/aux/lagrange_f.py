#!/usr/bin/env python

from cpoinsc import collptsgen

"""
Lagrange interpolating polynomials by David M Thierry
contains lgr, lgry, lgrdot, lgrydot
10/11/2016
"""

__author__ = 'David M Thierry'


def lgr(j, tau, kord, alp, bet):
    tauk = collptsgen(kord, alp, bet)
    tauk.reverse()
    tauk.append(0.)
    tauk.reverse()
    out = 1
    for k in range(0, kord + 1):
        if j != k:
            out *= (tau - tauk[k]) / (tauk[j] - tauk[k])
    return out


def lgry(j, tau, kord, alp, bet):
    tauk = collptsgen(kord, alp, bet)
    tauk.reverse()
    tauk.append(0.)
    tauk.reverse()
    out = 1
    # for legendre [0, K-1]
    if j == 0:
        return 0
    else:
        for k in range(1, kord + 1):
            if j != k:
                out *= (tau - tauk[k]) / (tauk[j] - tauk[k])
        return out


def lgrdot(j, tau, kord, alp, bet):
    tauk = collptsgen(kord, alp, bet)
    tauk.reverse()
    tauk.append(0.)
    tauk.reverse()
    out1 = 1
    for k in range(0, kord + 1):
        if k != j:
            out1 *= 1 / (tauk[j] - tauk[k])
    out2 = 1
    out3 = 0
    for m in range(0, kord + 1):
        if m != j:
            out2 = 1  # initialize multiplication
            for n in range(0, kord + 1):
                if n != m and n != j:
                    out2 *= tau - tauk[n]
                    # elif n == j:
                    # print ("we've got a problem here")
            out3 += out2
    out = out3 * out1

    return out


def lgrydot(j, tau, kord, alp, bet):
    tauk = collptsgen(kord, alp, bet)
    tauk.reverse()
    tauk.append(0.)
    tauk.reverse()
    out1 = 1
    for k in range(1, kord + 1):
        if k != j:
            out1 *= 1 / (tauk[j] - tauk[k])
    out2 = 1
    out3 = 0
    for m in range(1, kord + 1):
        if m != j:
            out2 = 1  # initialize multiplication
            for n in range(1, kord + 1):
                if n != m and n != j:
                    out2 *= tau - tauk[n]
                    # elif n == j:
                    # print ("we've got a problem here")
            out3 += out2
    out = out3 * out1

    return out

