# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
with open("../whatevs", "r") as f:
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

# with open("../res_nmpc_offs_1562120001.txt", "r") as f:
#     lines = f.readlines()
#     r1 = np.zeros(shape)
#     j = 0
#     for line in lines:
#         if j == shape[1]:
#             break
#         vals = line.split("\t")
#         new_l = []
#         for i in vals:
#             if isinstance(i, str):
#                 if i != '\n':
#                     new_l.append(float(i))
#         r1[j] = new_l
#         j += 1
    plt.figure("1")
    fig = plt.plot(r0[:, :4])
    plt.savefig("myfig_1.pdf")
    plt.close("1")

    plt.figure("2")
    fig = plt.plot(r0[:, 4:])
    plt.savefig("myfig_2.pdf")
    plt.close("2")

