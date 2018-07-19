#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 11:54:02 2018

@author: jeremiasknoblauch

Description: Plot illustration of Thm
"""


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
from matplotlib import rcParams
#from matplotlib import rc

dir_ = ("//Users//jeremiasknoblauch//Documents//OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//PaperNIPS//Thm1Illustration")
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

rcParams.update({'figure.autolayout': True})

xlabsize, ylabsize, legendsize = 15, 15, 15
linewidths = [2, 4]*5
linestyles = ["-", ":"]*5
linecolors = ["navy", "orange", "purple", "red", "orange"]
ax, fig = plt.subplots(1, figsize = (5,3))
handles, labels = fig.get_legend_handles_labels()
#for i in range(1, 5):
handle, = fig.plot(data[:,0], data[:,1], linewidth = linewidths[0],
                   linestyle = linestyles[0],
                   color = linecolors[0])
handles.append(handle)
handle = fig.axhline(1.0, linewidth = linewidths[1], linestyle = linestyles[1],
                      color=linecolors[1])
handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,1])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,2])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,3])
#handles.append(handle)
#handle, = fig.plot(data[:,0], data[:,4])
#handles.append(handle)
fig.set_xlabel(r'$|V|_{\min}$', size = xlabsize)
fig.set_ylabel("odds", size = ylabsize)
#labels = ["hallo"] #[r'$\beta=0.0$ (KLD)']
#plt.legend(handles, labels, prop = {'size':legendsize})
#fig.plot(data[:,0], data[:,5])
#fig.plot(data[:,0], data[:,2])
#fig.plot(data[:,0], data[:,3])
#fig.plot(data[:,0], data[:,4])
#fig.plot(data[:,0], data[:,5])

plt.savefig(dir_ + "//Thm1Pic.pdf",
                format = "pdf", dpi = 800)  

