#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:06:33 2018

@author: jeremiasknoblauch

Description: Get plots for EU1880 data
"""

import pickle
import numpy as np
from Evaluation_tool import EvaluationTool
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import csv
import datetime
import matplotlib
import math

#ensure that we have type 1 fonts (for ICML publishing guiedlines)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


"""STEP 1: Get file names"""

data_file = ("///Users//jeremiasknoblauch//Documents////OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//Data//EuropeanTemperatureData//1880//1880_temperatures.csv")

nbhs_file = ("//Users//jeremiasknoblauch//Documents////OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//Data//EuropeanTemperatureData//1880//1880_pw_distances.csv")

results_file = ("///Users//jeremiasknoblauch//Documents////OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//Paper//EUTemperature1880//" + 
    "results_EUTemp1880_medium_range_models.txt")

storage_folder = ("//Users//jeremiasknoblauch//Documents////OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//Paper//EUTemperature1880//")


"""STEP 2: Read in the data"""


"""Get p.w. distances"""
""" Read in (as strings)"""
pw_distances = []
station_IDs = []
count = 0 
with open(nbhs_file) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if count > 0:
            pw_distances += row
        else:
            station_IDs += row
        count += 1
num_stations = int(np.sqrt(len(pw_distances)))

"""Convert into floats"""
pwd = []
stat_IDs = []
for entry in pw_distances:
    pwd += [float(entry)]
count2 = 0
for entry in station_IDs:
    stat_IDs += [float(entry)]
pw_distances = np.array(pwd, dtype=float).reshape(num_stations, num_stations)
indices = np.linspace(0,num_stations-1, num_stations,dtype=int)

"""STEP 2.1: Read in both station IDs and temperature values"""
data_raw = []
count = 0 
with open(data_file) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if count > 0:
            data_raw += row
        count += 1
dat = []
for entry in data_raw:
    dat += [float(entry)]
    
T = int(len(dat)/(num_stations*2))
num_years = int(T/12)
data = np.array(dat).reshape(T*num_stations, 2)
temperatures = data[:,1]
IDs = data[:,0]


"""STEP 2.2: Using the station IDs, transform the data into spatial domain"""
temperatures_spatial = np.zeros((T,num_stations))
for (id_,location) in zip(stat_IDs, range(0,num_stations)):
    temperatures_spatial[:,location] = temperatures[np.where(id_ == IDs)]


"""STEP 2.3: Fill in the missings using averages as follows:
             Average(station) + Average(month for this station)"""
             
"""STEP 2.3.1: Get what you need to fill in"""
selection = temperatures_spatial != -9999.0
station_means = np.zeros(num_stations)
for location in range(0, num_stations):
    station_means[location] = np.mean(temperatures_spatial[
            selection[:,location], location])
year_means = np.zeros((num_stations, num_years))
month_means = np.zeros((num_stations, 12))
temperatures_spatial_demeaned = temperatures_spatial - station_means

for location in range(0, num_stations):
    """get month means after overall station mean subtracted"""
    for month in range(0, 12):
        selection_missings = temperatures_spatial[:,location] != -9999.0
        selection_month = ([False]*month + [True] + [False]*(11-month))*num_years
        selection = np.all(np.array([selection_missings, selection_month]), 
                           axis=0)
        month_means[location, month] = np.mean(temperatures_spatial_demeaned[selection, 
                   location])
    """get year means after overall station mean subtracted"""
    for year in range(0, num_years):
        start, stop = year*12, (year+1)*12
        selection = temperatures_spatial[
                start:stop,location] != -9999.0
        year_means[location, year] = np.mean(temperatures_spatial_demeaned[
                start:stop,location][selection])     
        
#Note: Some of the year means will be missings, so they need to be 
#      filled in by nearest neighbours
"""STEP 2.3.2: Fill in year means by looking for next past/future value 
that is not nan"""
for location in range(0, num_stations):
    for year in range(0, num_years):
        if math.isnan(year_means[location,year]):
            #find year_means adjacent that are not none and fill in using them
            found1, found2 = False, False
            val1, val2 = 0,0
            
            """search on the LHS (future)"""
            year_ = year+1
            while year_ < num_years:
                if not math.isnan(year_means[location, year_]):
                    found1=True
                    val1 = year_means[location,year_]
                year_ = year_ +1
            """search on the RHS (past)"""
            year_ = year-1
            while year_ >= 0:
                if not math.isnan(year_means[location, year_]):
                    found2=True
                    val2 = year_means[location,year_]
                year_ = year_ -1
            """average (if possible)"""
            if not found1:
                year_means[location,year] = val2
            if not found2:
                year_means[location,year] = val1
            if found1 and found2:
                year_means[location,year] = 0.5*(val1+val2)

"""STEP 2.3.3: Exclude the stations with too many missings (make it dependent 
on the data set) and adapt the data sets and means by dropping relevant rows/
cols"""
exclusion_indices = [12,14,18,20]
kept_indices = np.setdiff1d(indices, exclusion_indices)   
    
num_stations = len(kept_indices)
temperatures_spatial = np.delete(temperatures_spatial, obj = exclusion_indices,
                                axis=1)
temperatures_spatial_demeaned = np.delete(temperatures_spatial_demeaned, 
                obj = exclusion_indices, axis=1)
station_means = np.delete(station_means, obj = exclusion_indices)
year_means = np.delete(year_means, obj = exclusion_indices, axis=0)
month_means = np.delete(month_means, obj=exclusion_indices, axis=0)




"""STEP 2.3.4: Fill in individual months by combining year + month deviation"""
"""STEP 2.3.5: Deseasonalize uisng year-means+ month means"""
for location in range(0, num_stations):
    for year in range(0, num_years):
        year_effect_controlling_for_months = (year_means[location, year] - 
                                np.mean(month_means[location,:]))
        for month in range(0,12):
            """Select ranges corr. to missings, months, years"""
            selection_missings = temperatures_spatial[:,location] == -9999.0
            selection_month = ([False]*month + [True] + [False]*(11-month))*num_years
            selection_year = [False]*12*year + [True]*12 + [False]*(num_years-year-1)*12
            selection = np.all(np.array([selection_missings, selection_month,
                                         selection_year]), axis=0)  
            selection2 = np.all(np.array([selection_month,
                                         selection_year]), axis=0)
            """Fill in and obtain normal and demeaned versions"""
            temperatures_spatial[selection,location] = (
                    station_means[location] + month_means[location, month] +
                    year_effect_controlling_for_months)
            
            temperatures_spatial_demeaned[selection2,location] = (
                    temperatures_spatial[selection2,location] - 
                    station_means[location] -
                    month_means[location, month])

"""Normalize"""
temperatures_spatial_demeaned = (temperatures_spatial_demeaned - 
                        np.mean(temperatures_spatial_demeaned, axis=0))/np.sqrt(
                                np.var(temperatures_spatial_demeaned, axis=0))


"""STEP 3: Read in the results"""
"""Read in results"""
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_results(results_file) 

"""STEP 4: Get your plots"""
segmentation = EvT.results[EvT.names.index("MAP CPs")][-2]
model_labels = EvT.results[EvT.names.index("model labels")]
num_models = len(np.union1d(model_labels, model_labels))
relevant_models = np.union1d([seg[1] for seg in segmentation],[seg[1] for seg in segmentation])
mods = [8,11,13,17,18]
all_models = np.linspace(0, len(model_labels)-1, len(model_labels), dtype = int)


""""STEP 5: Get annotations"""
#http://file.scirp.org/pdf/ACS_2013062615184222.pdf
#https://en.wikipedia.org/wiki/History_of_climate_change_science
#https://en.wikipedia.org/wiki/Climate_change_in_Europe
#https://en.wikipedia.org/wiki/Second_Industrial_Revolution
#https://en.wikipedia.org/wiki/Post%E2%80%93World_War_II_economic_expansion 

#    if one_model_posterior_plot:
##custom_colors = ['purple', 'orange', 'red', 'green']

#indices: 11: Kremsmuenster, AT 0
#         21: Zagreb, HR 1
#         27: Prague, CZ 2
#         28: GER 3
#         49: Jena, GER 4
#        169: Bologna, IT 5 
#        173: Milan, IT  6
#        264: Oksoy Fyord, NO 7
#        271: Armagh, GB 8
#        303: Vestervig, DK 9
#        304: Nordby, DK 10
#        349: Stornoway airport, GB 11
#        441: Galway, IR 12
#        1684: Gospic, HR 13 
#        1685: Osijek, HR 14
#        1686: Zavizan, HR 15 
#        4013: Offenbach, GER 16
#        4291: Kleinmachnow, GER 17 
#        4327: Bamberg, GER 18
#        4431: Muenchen, Ger 19
#        10901: Knin, HR 20

plot_3  = True
if plot_3:
    #paper: height_ratio, num_subplots = [4,4,4], 3
    height_ratio, num_subplots = [3,4,3], 3
else:
    height_ratio, num_subplots = [3,4],2

ylabel_coords = [-0.05, 0.5]

#paper: figsize = (8,5)
fig, ax_array = plt.subplots(num_subplots, sharex = True, 
                             gridspec_kw = {'height_ratios':height_ratio}, 
                             figsize=(12,5))
plt.subplots_adjust(hspace = .2, left = None, bottom = None, right = None, top = None)

indices = [3,2]
rescaled = temperatures_spatial_demeaned.copy()
rescaled[:, indices ] = rescaled[:,indices] + np.linspace(-1,len(indices)-1, len(indices))*2.5
fig_1 = EvT.plot_raw_TS(rescaled, 
                        indices = indices,
                        start_plot = 1880, stop_plot= 2010,
                        xlab= None, ylab = "°C",
                        ylab_fontsize = 17,
                        yticks_fontsize = 15,
                        ylabel_coords = ylabel_coords, 
                        custom_colors_series = ["black"]*6, #["black","lightgray",  "darkgray", "dimgray", "black"]*len(indices))
                        ax = ax_array[0])

period_time_list = [[1880, 1914], [1950, 1973], [1987, 2010]]
label_list = ["1","2","3"]


fig_2 = EvT.plot_model_posterior(indices=mods, #mods, #mods, #relevant_models, 
                                 plot_type = "MAP",
                                 y_axis_labels = [#"AR(1)", 
                                                 "M(5+)", "M(6)", 
                                                  "M(6+)",
                                                  "M(7)", "M(7+)"],#relevant_models],
                                 log_format=False, aspect = 'auto',
                                 show_MAP_CPs = False,
                                 start_plot = 1880, stop_plot = 2010,
                                 custom_colors = ["green"], #custom_colors_models, 
                                 ax = ax_array[1], xlab = None, ylab = None,
                                 period_time_list = period_time_list, 
                                 label_list =label_list, 
                                 number_offset = 0.75,
                                 number_fontsize = 20,
                                 period_line_thickness = 4.0,
                                 xlab_fontsize = 14, ylab_fontsize = 14,
                                 ylabel_coords = ylabel_coords,
                                 xticks_fontsize = 14, yticks_fontsize = 14,
                                 window_len = 12*8) #MAP variance 2 det: windowlen = 125
                                                    #MAP variance 1 trace: window_len = 12*8

if plot_3:
    fig_3 = EvT.plot_model_posterior(indices=mods, #mods, #mods, #relevant_models, 
                                 plot_type = "MAPVariance2_det", #"MAPVariance1_trace",
                                 y_axis_labels = [#"AR(1)", 
                                                 "M(5+)", "M(6)", 
                                                  "M(6+)",
                                                  "M(7)", "M(7+)"],#relevant_models],
                                 log_format=False, aspect = 'auto',
                                 show_MAP_CPs = False,
                                 start_plot = 1880, stop_plot = 2010,
                                 custom_colors = ["orange"], #custom_colors_models, 
                                 ax = ax_array[2], xlab = "Year", ylab = "log(SGV)", #trace",
                                 period_time_list = None, 
                                 label_list =None, 
                                 number_offset = 0.75,
                                 number_fontsize = 20,
                                 period_line_thickness = 7.0,
                                 xlab_fontsize = 14, ylab_fontsize = 14,
                                 xticks_fontsize = 14, yticks_fontsize = 14,
                                 ylabel_coords = ylabel_coords,
                                 window_len = int(12*6),
                                 SGV = True,
                                 log_det = True)
#    fig_4 = EvT.plot_model_posterior(indices=mods, #mods, #mods, #relevant_models, 
#                                 plot_type = "MAPVariance2_det",
#                                 y_axis_labels = [#"AR(1)", 
#                                                 "M(5+)", "M(6)", 
#                                                  "M(6+)",
#                                                  "M(7)", "M(7+)"],#relevant_models],
#                                 log_format=False, aspect = 'auto',
#                                 show_MAP_CPs = False,
#                                 start_plot = 1880, stop_plot = 2010,
#                                 custom_colors = ["orange"], #custom_colors_models, 
#                                 ax = ax_array[3], xlab = "Year", ylab = "SVG",
#                                 period_time_list = None, 
#                                 label_list =None, 
#                                 number_offset = 0.75,
#                                 number_fontsize = 20,
#                                 period_line_thickness = 7.0,
#                                 xlab_fontsize = 14, ylab_fontsize = 14,
#                                 xticks_fontsize = 14, yticks_fontsize = 14,
#                                 ylabel_coords = ylabel_coords,
#                                 window_len = int(12*6),
#                                 SGV = True,
#                                 log_det = True)

fig.savefig(storage_folder + "//EU1880_model_posterior_1.pdf", format = "pdf", dpi = 800)




