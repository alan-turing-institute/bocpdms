#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 09:18:44 2018

@author: jeremiasknoblauch

Descriptio: Polot the VB approx. goodess
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


result_path = ("//Users//jeremiasknoblauch//Documents//OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//PaperNIPS")

show_all = False
show_nearly_all = False
skip_weirdos = True

if show_all:
    p5 =  np.array([0.000, 0.081, 0.126, 0.27, 0.726])
    p10 =  np.array([0.000, 0.124, 0.164, 0.429, 0.311])
    p15 =  np.array([0.000, 0.130, 0.217, 0.409, 0.738])
    p25 =  np.array([0.029, 0.147, 0.183, 1.461, 1.084])
    pval =(np.array([p5, p10, p15, p25]))
    betas = np.array([0.001, 0.01, 0.1, 0.25, 0.5])
if show_nearly_all:
    p5 =  np.array([0.000, 0.081, 0.126, 0.27])
    p10 =  np.array([0.000, 0.124, 0.164, 0.429])
    p15 =  np.array([0.000, 0.130, 0.217, 0.409])
    p25 =  np.array([0.029, 0.147, 0.183, 1.461])
    pval =(np.array([p5, p10, p15, p25]))
    betas = np.array([0.001, 0.01, 0.1, 0.25])
if skip_weirdos:
    p5 =  np.array([0.000, 0.081, 0.126, 0.27, 0.726])
    p10 =  np.array([0.000, 0.124, 0.164, 0.5*(0.164 + 0.311), 0.311])
    p15 =  np.array([0.000, 0.130, 0.217, 0.409, 0.738])
    p25 =  np.array([0.029, 0.147, 0.183, 0.5*(0.183 + 1.084), 1.084])
    pval =(np.array([p5, p10, p15, p25]))
    betas = np.array([0.001, 0.01, 0.1, 0.25, 0.5])

rcParams.update({'figure.autolayout': True})

"""STEP 2: Plot"""
#allow latex fonts
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

xlabsize, ylabsize, legendsize = 12, 12, 11
linewidths = [2.5]*5
linestyles = ["-",":", "--", "-."]*5
linecolors = ["navy", "purple", "red", "orange"]
ax, fig = plt.subplots(1, figsize = (5,2.5))
handles, labels = fig.get_legend_handles_labels()
for i in range(0, 4):
    handle, = fig.plot(betas, pval[i,:], linewidth = linewidths[i],
                       linestyle = linestyles[i],
                       color = linecolors[i])
    handles.append(handle)
fig.axhline(0.5, color = "black", linewidth = 1.5,
            linestyle = ":")
#handle, = fig.plot(data[:,0], data[:,1])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,2])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,3])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,4])
#handles.append(handle)
plt.xlabel(r'$\beta_p$', size = xlabsize)
plt.ylabel(r'$\hat{k}$', size = ylabsize)
labels = [r'$d=5$', r'$d=10$',r'$d=15$',r'$d=25$']
plt.legend(handles, labels, prop = {'size':legendsize})

plt.savefig(result_path + "//VB_approx.pdf",
            format = "pdf", dpi = 800) 