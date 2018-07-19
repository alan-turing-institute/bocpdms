#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 14:11:07 2018

@author: jeremiasknoblauch

Description: Plots the theorem + the VB picture together
"""



import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import rcParams
import string
#from matplotlib import rc

dir_ = ("//Users//jeremiasknoblauch//Documents//OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//PaperNIPS//Thm1Illustration")
res_path  = ("//Users//jeremiasknoblauch//Documents//OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//PaperNIPS")

well_file = dir_ + "//illustrationData.csv"

"""STEP 1: Read in the data"""

raw_data = []
count = 0 
with open(well_file) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if count > 0: #skip header. Header is det | odds
            raw_data += row
        count = count +1

raw_data_float = []
for entry in raw_data:    
    raw_data_float.append(float(entry))
raw_data = raw_data_float

num_cols, num_rows = 2, int(len(raw_data)/2)
data = np.zeros((num_rows, num_cols))
count = 0
for entry, count in zip(raw_data, range(0, int(len(raw_data)))):
    ind_col = count % 2
    ind_row = int(count/2)
    data[ind_row, ind_col] = entry

#data = np.array(raw_data, dtype = float).reshape(num_cols, num_rows)

"""STEP 2: Plot"""
#allow latex fonts
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

#rcParams.update({'figure.autolayout': True})



xlabsize, ylabsize, legendsize = 15, 15, 10.5
width_ratio = [3, 7]
linewidths = [2, 4]*5
linestyles = ["-", ":"]*5
linecolors = ["navy", "orange", "purple", "red", "orange"]
fig, ax_array = plt.subplots(1,2, figsize = (10,3.1), sharey=False,
                             gridspec_kw = {'width_ratios':width_ratio})
ax = ax_array[0]
handles, labels = ax.get_legend_handles_labels()
#for i in range(1, 5):
handle, = ax.plot(data[:,0], data[:,1], linewidth = linewidths[0],
                   linestyle = linestyles[0],
                   color = linecolors[0])
handles.append(handle)
handle = ax.axhline(1.0, linewidth = linewidths[1], linestyle = linestyles[1],
                      color=linecolors[1])
handles.append(handle)
ax.set_xlabel(r'$|V|_{\min}$', size = xlabsize)
ax.set_ylabel("odds", size = ylabsize)
ax.set_yticks((np.arange(0, 3, step=0.5)))

ax.text(-0.125, 0.97, string.ascii_uppercase[0], 
            transform=ax.transAxes, 
            size=20, weight='bold')


"""STEP 3: Plot the second plot"""

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

#rcParams.update({'figure.autolayout': True})
#allow latex fonts
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

linewidths = [2.5]*5
linestyles = ["-",":", "--", "-."]*5
linecolors = ["navy", "purple", "red", "orange"]
#ax, fig = plt.subplots(1, figsize = (5,2.5))
ax = ax_array[1]
handles, labels = ax.get_legend_handles_labels()
for i in range(0, 4):
    handle, = ax.plot(betas, pval[i,:], linewidth = linewidths[i],
                       linestyle = linestyles[i],
                       color = linecolors[i])
    handles.append(handle)
ax.axhline(0.5, color = "black", linewidth = 1.5,
            linestyle = ":")
#handle, = fig.plot(data[:,0], data[:,1])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,2])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,3])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,4])
#handles.append(handle)
ax.set_xlabel(r'$\beta_p$', size = xlabsize)
ax.set_ylabel(r'$\hat{k}$', size = ylabsize)
labels = [r'$d=5$', r'$d=10$',r'$d=15$',r'$d=25$']
ax.legend(handles, labels, prop = {'size':legendsize})

ax.text(-0.05, 0.97, string.ascii_uppercase[1], 
            transform=ax.transAxes, 
            size=20, weight='bold')

fig.savefig(res_path + "//THM_+_VB.pdf",
                format = "pdf", dpi = 800)
