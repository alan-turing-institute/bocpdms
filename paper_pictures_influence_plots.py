#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 12 16:32:59 2018

@author: jeremiasknoblauch

Description: Picture for influence functions
"""

import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt
#from matplotlib import rc

dir_ = ("//Users//jeremiasknoblauch//Documents//OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//PaperNIPS//influencePlot")
well_file = dir_ + "//InfluencePlotData.csv"

"""STEP 1: Read in the data"""

raw_data = []
count = 0 
with open(well_file) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if count > 0: #skip header
            raw_data += row
        count = count +1

raw_data_float = []
for entry in raw_data:    
    raw_data_float.append(float(entry))
raw_data = raw_data_float

num_cols, num_rows = 5, int(len(raw_data)/5)
data = np.zeros((num_rows, num_cols))
count = 0
for entry, count in zip(raw_data, range(0, int(len(raw_data)))):
    ind_col = count % 5
    ind_row = int(count/5)
    data[ind_row, ind_col] = entry

#data = np.array(raw_data, dtype = float).reshape(num_cols, num_rows)

"""STEP 2: Plot"""
#allow latex fonts
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

xlabsize, ylabsize, legendsize = 12, 12, 11
linewidths = [4]*5
linestyles = ["-"]*5
linecolors = [0, "navy", "purple", "red", "orange"]
ax, fig = plt.subplots(1, figsize = (3.5,4.5))
handles, labels = fig.get_legend_handles_labels()
for i in range(1, 5):
    handle, = fig.plot(data[:,0], data[:,i], linewidth = linewidths[i],
                       linestyle = linestyles[i],
                       color = linecolors[i])
    handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,1])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,2])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,3])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,4])
#handles.append(handle)
plt.xlabel("Standard Deviations", size = xlabsize)
plt.ylabel("Influence", size = ylabsize)
labels = [r'$\beta=0.0$ (KLD)', r'$\beta=0.05$', r'$\beta=0.2$',r'$\beta=0.25$']
plt.legend(handles, labels, prop = {'size':legendsize})
#fig.plot(data[:,0], data[:,5])
#fig.plot(data[:,0], data[:,2])
#fig.plot(data[:,0], data[:,3])
#fig.plot(data[:,0], data[:,4])
#fig.plot(data[:,0], data[:,5])
