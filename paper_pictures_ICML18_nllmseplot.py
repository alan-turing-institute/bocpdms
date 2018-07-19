#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 17:05:16 2018

@author: jeremiasknoblauch

Description: Plots MSE + NLL for GP-models vs. SSBVAR + MS
"""

import numpy as np
import matplotlib.pyplot as plt

dir_ = ("//Users//jeremiasknoblauch//Documents//OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//Paper//Presentation//MSE")


MSE_vals = [
        [0.553, 0.750, 2.62, 29.95], #ARGPCP
        [0.583, 0.689, 3.13, 30.17], #GPTSCP
        [0.585, 0.618, 3.17], #NSGP
        [0.55, 0.681, 1.74, 25.93] #BBVAR
        ]
NLL_vals = [
        [1.15, -0.604, 4.07, 39.5495], #ARGPCP
        [1.19, 1.17, 4.54, 39.44], #GPTSCP
        [1.15, -1.98, 4.19], #NSGP
        [1.13, 0.923, 3.57, 48.32] #BBVARa
        ]
MSE_95 = [
        [0.0962, 0.0315, 0.195, 0.5], #ARGPCP
        [0.0989, 0.0294, 0.241, 0.51], #GPTSCP
        [0.0988, 0.0242, 0.230], #NSGP
        [0.0948, 0.0245, 0.222, 0.906] #BVAR
        ]

NLL_95 = [
        [0.0555, 0.0385, 0.150, 0.22],
        [0.0548, 0.0183, 0.188, 0.22],
        [0.0655, 0.0561, 0.0212],
        [0.0684, 0.0231, 0.166, 0.964]
        ]

baseline = np.array([1, 1, 3, 30])



xlabsize, ylabsize, legendsize, ticksize = 15, 15, 13,12
linewidths = [3]*5
linestyles = ["-"]*5
linecolors = ["navy", "purple", "red", "orange"]
ax, fig = plt.subplots(1, figsize = (6,4))
handles, labels = fig.get_legend_handles_labels()
for i in [0,1,2,3]:
    if i == 2:
        dat = np.array(MSE_vals[i])/baseline[:-1]
        err = np.array(MSE_95[i])/baseline[:-1]
        x_ = [0,1,2]
    else:
        dat = np.array(MSE_vals[i])/baseline
        err = np.array(MSE_95[i])/baseline
        x_ = [0,1,2,3]
    handle = fig.errorbar(x=x_,y=dat,yerr = err,
                        linewidth = linewidths[i],
                       linestyle = linestyles[i],
                       color = linecolors[i],
                       #solid_capstyle='round',
                       marker = 'o',
                       ms=7,
                       capsize=5)
    handles.append(handle)
plt.xlabel("Data Set", size = xlabsize)
plt.ylabel("MSE/Variance", size = ylabsize)
labels = ['ARGPCP', 'GPTSCP', 'NSGP','SSBVAR']
plt.legend(handles, labels, prop = {'size':legendsize})
plt.xticks([0,1,2,3],["Nile", "Snow", "Bee", "30PF"])
plt.tick_params(labelsize = ticksize)


plt.savefig(dir_ + "//MSE.pdf",
            format = "pdf", dpi = 800)



xlabsize, ylabsize, legendsize, ticksize = 15, 15, 13,12
linewidths = [3]*5
linestyles = ["-"]*5
linecolors = ["navy", "purple", "red", "orange"]
ax, fig = plt.subplots(1, figsize = (6,4))
handles, labels = fig.get_legend_handles_labels()
for i in [0,1,2,3]:
    if i == 2:
        dat = np.array(NLL_vals[i])/baseline[:-1]
        err = np.array(NLL_95[i])/baseline[:-1]
        x_ = [0,1,2]
    else:
        dat = np.array(NLL_vals[i])/baseline
        err = np.array(NLL_95[i])/baseline
        x_ = [0,1,2,3]
    handle = fig.errorbar(x=x_,y=dat,yerr = err,
                        linewidth = linewidths[i],
                       linestyle = linestyles[i],
                       color = linecolors[i],
                       #solid_capstyle='round',
                       marker = 'o',
                       ms=7,
                       capsize=5)
    handles.append(handle)
plt.xlabel("Data Set", size = xlabsize)
plt.ylabel("NLL/Variance", size = ylabsize)
labels = ['ARGPCP', 'GPTSCP', 'NSGP','SSBVAR']
plt.legend(handles, labels, prop = {'size':legendsize})
plt.xticks([0,1,2,3],["Nile", "Snow", "Bee", "30PF"])
plt.tick_params(labelsize = ticksize)


plt.savefig(dir_ + "//NLL.pdf",
            format = "pdf", dpi = 800)