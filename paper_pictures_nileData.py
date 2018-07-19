#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:18:58 2018

@author: jeremiasknoblauch

Description: Get the Nile data pics
"""

import pickle
import numpy as np
from Evaluation_tool import EvaluationTool
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import csv
import datetime
import matplotlib
import datetime


#ensure that we have type 1 fonts (for ICML publishing guiedlines)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


baseline_working_directory = ("//Users//jeremiasknoblauch//Documents//OxWaSP"+
    "//BOCPDMS//Code//SpatialBOCD//Paper//NileData")
nile_file = ("//Users//jeremiasknoblauch//Documents//OxWaSP"+
    "//BOCPDMS//Code//SpatialBOCD//Data//nileData//nile.txt")
results_file = ("//Users//jeremiasknoblauch//Documents//OxWaSP"+
    "//BOCPDMS//Code//SpatialBOCD//Paper//NileData//results_nile.txt")


"""Read in raw data"""
"""STEP 1: Read in and convert to float"""
raw_data = []
count = 0 
with open(nile_file) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        raw_data += row

raw_data_float = []
for entry in raw_data:
    raw_data_float.append(float(entry))
raw_data = raw_data_float

"""STEP 2: put into right form"""
T = int(len(raw_data)/2)
S1, S2 = 1,1
data = np.array(raw_data).reshape(T,2)
dates = data[:,0]
river_height = data[:,1]
mean, variance = np.mean(river_height), np.var(river_height)
river_height = (river_height-mean)/np.sqrt(variance)

"""STEP 3: Get dates"""
all_dates = []
for i in range(622+2, 1285):
    all_dates.append(datetime.date(i, 1,1))

"""Read in results"""
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_results(results_file) 



"""get MAP CPs in range"""
segmentation = np.array(EvT.results[EvT.names.index("MAP CPs")][-2])
models = np.union1d([e[1] for e in segmentation],
                    [e[1] for e in segmentation]) 


"""Obtain the plot for RLD"""
start, stop = (2007 + 7/12), 2009
#start, stop = datetime.date(2007, 8, 1), datetime.date(2008, 12, 31)
height_ratio =[10,14]
custom_colors = ["blue", "purple"] #["green", "darkviolet", "orange", "purple", "turquoise"]
fig, ax_array = plt.subplots(2, figsize=(8,5), sharex = True, 
                             gridspec_kw = {'height_ratios':height_ratio})
plt.subplots_adjust(hspace = .35, left = None, bottom = None, right = None, top = None)

"""Get the date format"""
years = mdates.YearLocator()   # every year
yearsFmt = mdates.DateFormatter('%Y')

"""placement of y-labels"""
ylabel_coords = [-0.065, 0.5]

EvT.plot_raw_TS(river_height[2:].reshape(T-2,1), 
                indices = [0], 
                xlab = None, 
                show_MAP_CPs = True, 
                           time_range = np.linspace(1,
                             T-2, 
                             T-2, dtype=int), 
                           print_plt = False,
                       ylab = "River Height", ax = ax_array[0],
                       all_dates = np.linspace(622 + 1, 1284, 1284 - (622 + 1), dtype = int),#None, #all_dates, 
                       custom_colors_series = ["black"],
                       custom_colors_CPs = ["blue", "blue"],
                       custom_linestyles = ["solid"]*2,
                       ylab_fontsize = 14,
                       ylabel_coords = ylabel_coords,
                       set_ylims = (-2.75, 3.95)
                       )

EvT.plot_run_length_distr(
    buffer=0, 
    show_MAP_CPs = True, 
    mark_median = False, 
    mark_max = True, 
    upper_limit = 660, 
    print_colorbar = True, 
    colorbar_location= 'bottom',
    log_format = True, 
    aspect_ratio = 'auto', 
    C1=0,C2=1, 
    time_range = np.linspace(1,
                             T-2, 
                             T-2, dtype=int), 
    start = 622 + 2, stop = 1284, #start=start, stop = stop, 
    all_dates = None, #all_dates,
    event_time_list=[715 ],#datetime.date(715,1,1)], 
    label_list=["nilometer"], space_to_colorbar = 0.52,
    custom_colors = ["blue", "blue"], #["blue"]*len(event_time_list),
    custom_linestyles = ["solid"]*3,
    custom_linewidth = 3,
    arrow_colors= ["black"],
    number_fontsize = 14,
    arrow_length = 135,
    arrow_thickness = 3.0,
    xlab_fontsize =14,
    ylab_fontsize = 14, 
    arrows_setleft_indices = [0],
    arrows_setleft_by = [50],
    zero_distance = 0.0,
    ax = ax_array[1], figure = fig,
    no_transform = True,
    date_instructions_formatter = None, #yearsFmt,
    date_instructions_locator = None,
    ylabel_coords = ylabel_coords,
    xlab = "Year",
    arrow_distance = 25
    )
    
    
#EvT.plot_prediction_error(river_height.reshape(T,1,1), indices = [0], 
#                          show_var = True, up_to = 250,
#                          show_MAP_CPs = True)
    
#EvT.plot_model_posterior(indices = [0,1,2], log_format = False, up_to = 150,
#                         plot_type = "MAPVariance1_det", show_MAP_CPs = True)
    

fig.savefig(baseline_working_directory + "//nile_plot.pdf",
            format = "pdf", dpi = 800)
fig.savefig(baseline_working_directory + "//nile_plot.jpg",
            format = "jpg", dpi = 800)
