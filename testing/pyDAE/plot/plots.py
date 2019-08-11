#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def main():
    sp = True
    #stamp = "1562949020"
    stamp = "1565465970"
    stamp = "1565558305"
    stamp = "1565559962"
    stamp = "1565563121" #!
    stamp = "1565564933"

    fformat = "pdf"
    l = {}
    for i in range(1, 43):
        l[i] = i
    if sp:
        print("Set-point track")
        with open("../res_nmpc_rs_" + stamp + ".txt", "r") as f:
            print("SPT.\n\n")
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
            fig = plt.plot(r0[:, :2])
            plt.savefig("spt_1.pdf")
            plt.close("1")

            plt.figure("2")
            fig = plt.plot(r0[:, 2:])
            plt.savefig("spt_2.pdf")
            plt.close("2")

    print("")
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
        print("etc\n\n")
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
        plt.savefig("myfig_L." + fformat)
        plt.close("4")

        plt.figure("5")
        fig = plt.plot(r0[:, 85:])
        plt.savefig("myfig_V." + fformat)
        plt.close("5")

if __name__ == '__main__':
    main()
