#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def main():
    kind = "other"
    stamp = "1562947126"
    fformat = "pdf"
    print(kind)
    l = {}
    for i in range(1, 43):
        l[i] = i
    if kind == "sp":
        with open("../res_nmpc_rs_1562623415.txt", "r") as f:
            print("Je ferme les yeux.\n\n")
            lines = f.readlines()
            shape = (len(lines), len(lines[0].split("\t")) - 1)
            r0 = np.zeros(shape)
            j = 0
            for line in lines:
                vals = line.split("\t")
                new_l = []
                for i in vals:
                    if isinstance(i, str):
                        if i != '\n':
                            new_l.append(float(i))

                r0[j] = new_l
                j += 1

            plt.figure("1")
            fig = plt.plot(r0[:, :4])
            plt.savefig("myfig_1.pdf")
            plt.close("1")

            plt.figure("2")
            fig = plt.plot(r0[:, 4:])
            plt.savefig("myfig_2.pdf")
            plt.close("2")
    else:
        with open("../res_dyn_" + stamp + ".txt", "r") as f:
            print("dyn\n\n")
            lines = f.readlines()
            shape = (len(lines), len(lines[0].split("\t")) - 1)
            r0 = np.zeros(shape)
            j = 0
            for line in lines:
                vals = line.split("\t")
                new_l = []
                for i in vals:
                    if isinstance(i, str):
                        if i != '\n':
                            new_l.append(float(i))

                r0[j] = new_l
                j += 1

            plt.figure("1")
            fig = plt.plot(r0[:, 43:])
            plt.savefig("myfig_M." + fformat)
            plt.close("1")
            plt.figure("2")
            fig = plt.plot(r0[:, :42])
            plt.savefig("myfig_x." + fformat)
            plt.close("2")
        with open("../res_parfois_" + stamp + ".txt", "r") as f:
            print("parfois\n\n")
            lines = f.readlines()
            shape = (len(lines), len(lines[0].split("\t")) - 1)
            r0 = np.zeros(shape)
            j = 0
            for line in lines:
                vals = line.split("\t")
                new_l = []
                for i in vals:
                    if isinstance(i, str):
                        if i != '\n':
                            new_l.append(float(i))

                r0[j] = new_l
                j += 1

            plt.figure("3")
            for i in range(1, 42):
                fig = plt.plot(r0[:, i], label=i, linewidth=0.5)

            #plt.legend(loc='center')
            plt.title("$\\beta$")
            #plt.ylim((0, 2))
            plt.savefig("myfig_beta." + fformat)
            plt.close("3")

            plt.figure("4")
            fig = plt.plot(r0[:, 43:84])
            plt.savefig("myfig_T." + fformat)
            plt.close("4")

            plt.figure("5")
            fig = plt.plot(r0[:, 85:])
            plt.savefig("myfig_y." + fformat)
            plt.close("5")

if __name__ == '__main__':
    main()
