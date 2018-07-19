#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 14:25:06 2018

@author: jeremiasknoblauch

Description: Plot the Air Pollution Data for NIPS submission
"""


import csv
import numpy as np
import scipy
from Evaluation_tool import EvaluationTool
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime
import matplotlib
#import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)

plot_model_post_only = False
plot_model_post_and_rld_only = True

plot_top = False
plot_bottom = True


"""""STEP 1: DATA TRANSFOMRATIONS"""""
normalize = True
deseasonalize_2h = True
deseasonalize_day = True #only one of the two deseasonalizations should be chosen
shortened, shortened_to = False, 500
daily_avg = True
if daily_avg:
    deseasonalize_2h = False


data_dir = ("//Users//jeremiasknoblauch//Documents////OxWaSP//BOCPDMS//" + 
            "//Code//SpatialBOCD//Data//AirPollutionData")
cp_type = "CongestionChargeData"
dist_file_road = (data_dir + "//" + cp_type + "//" + 
                  "RoadDistanceMatrix_")
dist_file_euclid = (data_dir + "//" + cp_type + "//" + 
                   "EuclideanDistanceMatrix_")
res_path = ("/Users//jeremiasknoblauch//Documents////OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//PaperNIPS//AirPollution//")
results_file_DPD = (res_path + "results_DPD.txt")
results_file_KL = (res_path + "results_KL.txt")
frequency = "2h" #2h, daily (=every 15 min), 
mode = "bigger" #bigger, smaller (bigger contains more filled-in values)


if mode == "bigger":
    stationIDs = ["BT1", "BX1", "BX2", "CR2", "CR4", 
                  "EA1", "EA2", "EN1", "GR4", "GR5", 
                  "HG1", "HG2", "HI0", "HI1", "HR1", 
                  "HS2", "HV1", "HV3", "KC1", "KC2",
                  "LH2", "MY1", "RB3", "RB4", "TD0", 
                  "TH1", "TH2", "WA2", "WL1"]
    
elif mode == "smaller":
    stationIDs = ["BT1", "BX2", "CR2", "EA2", "EN1", "GR4",
                  "GR5", "HG1", "HG2", "HI0", "HR1", "HV1",
                  "HV3", "KC1", "LH2", "RB3", "TD0", "WA2"]
    
num_stations = len(stationIDs)

"""STEP 1: Read in distances"""

"""STEP 1.1: Read in road distances (as strings)"""
pw_distances_road = []
station_IDs = []
count = 0 
with open(dist_file_road + mode + ".csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        pw_distances_road += row


"""STEP 1.2: Read in euclidean distances (as strings)"""
pw_distances_euclid = []
station_IDs = []
count = 0 
with open(dist_file_euclid + mode + ".csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        pw_distances_euclid += row

"""STEP 1.3: Convert both distance lists to floats and matrices"""
pw_d_r, pw_d_e = [], []
for r,e in zip(pw_distances_road, pw_distances_euclid):
    pw_d_r.append(float(r))
    pw_d_e.append(float(e))
pw_distances_road = np.array(pw_d_r).reshape(num_stations, num_stations)
pw_distances_euclid = np.array(pw_d_e).reshape(num_stations, num_stations)


"""STEP 2: Convert distance matrices to nbhs"""
cutoffs = [0.0, 10.0, 20.0, 30.0, 40.0, 100.0]
num_nbhs = len(cutoffs) - 1

"""STEP 2.1: road distances"""
road_nbhs = []
for location in range(0, num_stations):
    location_nbh = []
    for i in range(0, num_nbhs):
        larger_than, smaller_than = cutoffs[i], cutoffs[i+1]
        indices = np.intersect1d( 
            np.where(pw_distances_road[location,:] > larger_than),
            np.where(pw_distances_road[location,:] < smaller_than)).tolist()
        location_nbh.append(indices.copy())
    road_nbhs.append(location_nbh.copy())
        
"""STEP 2.2: euclidean distances"""
euclid_nbhs =[]
for location in range(0, num_stations):
    location_nbh = []
    for i in range(0, num_nbhs):
        larger_than, smaller_than = cutoffs[i], cutoffs[i+1]
        indices = np.intersect1d( 
            np.where(pw_distances_euclid[location,:] > larger_than),
            np.where(pw_distances_euclid[location,:] < smaller_than)).tolist()
        location_nbh.append(indices.copy()) 
    euclid_nbhs.append(location_nbh.copy())


"""STEP 3: Read in station data for each station"""
station_data = []
for id_ in stationIDs:
    file_name = (data_dir + "//" + cp_type + "//" + 
                 id_ + "_081702-081703_" + frequency + ".txt")
    
    """STEP 3.1: Read in raw data"""
    #NOTE: Skip the header 
    data_raw = []
    count = 0 
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if count > 0:
                data_raw += row
            count += 1    
    
    """STEP 3.2: Convert to floats"""
    #NOTE: We have row names, so skip every second
    dat = []
    for entry in data_raw:
        dat += [float(entry)]
    
    
    """STEP 3.3: Append to station_data list"""
    station_data.append(dat.copy())


"""STEP 4: Format the station data into a matrix"""
T, S1, S2 = len(station_data[0]), num_stations, 1
data = np.zeros((T, num_stations))
for i in range(0, num_stations):
    data[:,i] = np.array(station_data[i])
intercept_priors = np.mean(data,axis=0)
hyperpar_opt = "caron"


"""STEP 5: Transformation if necessary"""
if shortened:
    T = shortened_to
    data = data[:T,:]
    
if daily_avg:
    """average 12 consecutive values until all have been processed"""
    new_data = np.zeros((int(T/12), num_stations))
    for station in range(0, num_stations):
        new_data[:, station] = np.mean(data[:,station].
                reshape(int(T/12), 12),axis=1)
    data= new_data
    T = data.shape[0]
    

if deseasonalize_day:
    if deseasonalize_2h:
        print("CAREFUL! You want to deseasonalize twice, so deseasonalizing " +
              "was aborted!")
    elif not daily_avg:
        mean_day = np.zeros((7, num_stations))
        #deseasonalize
        for station in range(0, num_stations):
            """get the daily average. Note that we have 12 obs/day for a year"""
            for day in range(0, 7):
                selection_week = [False]*day + [True]*12 + [False]*(6-day)
                selection = (selection_week * int(T/(7*12)) + 
                             selection_week[:(T-int(T/(7*12))*7*12)])
                mean_day[day, station] = np.mean(data[selection,station])
                data[selection,station] = (data[selection,station] - 
                    mean_day[day, station])
                
if deseasonalize_day and daily_avg:
    mean_day = np.zeros((7, num_stations))
    #deseasonalize
    for station in range(0, num_stations):
        """get the daily average. Note that we have 12 obs/day for a year"""
        #Also note that T will already have changed to the #days
        for day in range(0, 7):
            selection_week = [False]*day + [True] + [False]*(6-day)
            selection = (selection_week * int(T/7) + 
                         selection_week[:(T-int(T/7)*7)])
            mean_day[day, station] = np.mean(data[selection,station])
            data[selection,station] = (data[selection,station] - 
                mean_day[day, station])
    T = data.shape[0]
              
                
if deseasonalize_2h:
    if deseasonalize_day:
        print("CAREFUL! You want to deseasonalize twice, so deseasonalizing " +
              "was aborted!")
    else:
        mean_2h = np.zeros((12*7, num_stations))
        for station in range(0, num_stations):
            """get the average for each 2h-interval for each weekday"""
            for _2h in range(0, 12*7):
                selection_2h = [False]*_2h + [True] + [False]*(12*7-1-_2h)
                selection = (selection_2h * int(T/(7*12)) + 
                             selection_2h[:(T-int(T/(7*12))*7*12)])
                mean_2h[_2h, station] = np.mean(data[selection,station])
                data[selection,station] = (data[selection,station] - 
                    mean_2h[_2h, station])
    
if normalize:
    data = (data - np.mean(data, axis=0))/np.sqrt(np.var(data,axis=0))
    intercept_priors = np.mean(data,axis=0)
    

"""""STEP 2: READ RESULTS"""""
EvTKL, EvTDPD = EvaluationTool(), EvaluationTool()
EvTKL.build_EvaluationTool_via_results(results_file_KL) 
EvTDPD.build_EvaluationTool_via_results(results_file_DPD)    
    

"""Get dates"""
def perdelta(start, end, delta, date_list):
    curr = start
    while curr < end:
        #yield curr
        date_list.append(curr)
        curr += delta
  
all_dates = []      
#start_year, start_month, start_day, start_hour = 2002, 8, 17, 0
#start_datetime = datetime.datetime(year = 2002, month = 8, day = 17, hour = 0)
#stop_datetime = datetime.datetime(year=2003, month = 8, day = 18, hour = 0)
#perdelta(start_datetime, stop_datetime, datetime.timedelta(hours = 2), all_dates)
start_year, start_month, start_day, start_hour = 2002, 8, 17, 0
start_datetime = datetime.date(year = 2002, month = 8, day = 17)
stop_datetime = datetime.date(year=2003, month = 8, day = 18)
perdelta(start_datetime, stop_datetime, datetime.timedelta(days = 1), all_dates)    
    
    
    



"""""STEP 3: Plot"""""
index_selection = [0,5,9,13,17,21,30]

#location, color
true_CPs = [[datetime.date(year = 2003, month = 2, day = 17), "red", 4.0]]

#paper: height_ratio, num_subplots = [4,3,5],3
#height_ratio, num_subplots = [4,3,4],3


if plot_model_post_only:
    
    TS_indices = [0,1,4, -1, -8]
    num_TS= len(TS_indices)
    height_ratio = [2] * num_TS + [3]*2
    num_subplots = len(height_ratio)
    #paper: ylabel_coords = [-0.085, 0.5]
    ylabel_coords = [-0.03, 0.5]
    
    #paper: figsize = (8,5)
    fig, ax_array = plt.subplots(num_subplots, sharex = True, 
                                 gridspec_kw = {'height_ratios':height_ratio}, 
                                 figsize=(12,5))
    plt.subplots_adjust(hspace = .125, left = None, bottom = None, right = None, top = None)
    
    off = 0
    time_range = np.linspace(10,T-2, T-2-off,dtype = int)
    all_dates = all_dates[-len(time_range):]
    show_CPs = True
    
    for i in range(0, num_TS):
        fig_1 = EvTKL.plot_raw_TS(data[-len(time_range):,:].reshape(len(time_range), 29), 
                                indices=[TS_indices[i]],
                                all_dates = all_dates, 
                                ax = ax_array[i], 
                                time_range = time_range,
                                custom_colors_series = ["black"]*10,
                                ylab_fontsize = 8,
                                yticks_fontsize = 8,
                                ylab = "NOX",
                                xlab=None,
                                ylabel_coords = ylabel_coords,
                                true_CPs = true_CPs)
    
    all_KL_CPs = EvTKL.results[EvTKL.names.index("MAP CPs")][-2]
    mod = [17,21]
    mod = [9,17,21]
    EvTKL.plot_model_posterior(indices=mod, #mods, #mods, #relevant_models, 
                                 plot_type = "trace", #"MAPVariance1_trace",
                                 #y_axis_labels = [str(e) for e in all_models],#[#"AR(1)", 
                                                 #"M(5+)", "M(6)", 
                                                 # "M(6+)",
                                                 # "M(7)", "M(7+)"],#relevant_models],
                                 time_range = time_range,
                                 y_axis_labels = [],
                                 log_format=False, aspect = 'auto',
                                 show_MAP_CPs = show_CPs,
                                 #start_plot = 2002.75, stop_plot = 2003.75,
                                 custom_colors = ["green", "blue", "orange"], # ["orange"], #custom_colors_models, 
                                 ax = ax_array[-2] ,#ax_array[1], #None, #ax_array[1], 
                                 xlab = None, #ylab = None, #trace",period_time_list = None,  
                                 number_offset = 1.0, #datetime.timedelta(days = 1),#0.75,
                                 number_fontsize = 20,
                                 period_line_thickness = 7.0,
                                 xlab_fontsize = 10, 
                                 ylab_fontsize = 10,
                                 xticks_fontsize = 10, 
                                 yticks_fontsize = 10,
                                 ylabel_coords = ylabel_coords,
                                 #ylab = None, #"Model posterior max",
                                 #period_time_list = [
                                 #   [datetime.datetime(year = 2003, month = 2, day = 17, hour = 0), 
                                 #   datetime.datetime(year = 2003, month = 2, day = 18, hour = 0)]], 
                                 #label_list = [["1"]], 
                                 #window_len = int(12*7*1),
                                 period_time_list = None, #[[datetime.datetime(year = 2003, month = 2, day = 17, hour = 0), 
                                    #datetime.datetime(year = 2003, month = 2, day = 18, hour = 0)]],
                                 label_list = None, #[["1"]],
                                 SGV = True,
                                 log_det = True,
                                 all_dates = all_dates,
                                 true_CPs = true_CPs)
    
    EvTDPD.plot_model_posterior(indices=[0,1,2], #mods, #mods, #relevant_models, 
                                 plot_type = "trace", #"MAPVariance1_trace",
                                 #y_axis_labels = [str(e) for e in all_models],#[#"AR(1)", 
                                                 #"M(5+)", "M(6)", 
                                                 # "M(6+)",
                                                 # "M(7)", "M(7+)"],#relevant_models],
                                 time_range = time_range,
                                 y_axis_labels = [],
                                 log_format=False, aspect = 'auto',
                                 show_MAP_CPs = show_CPs,
                                 #start_plot = 2002.75, stop_plot = 2003.75,
                                 custom_colors = ["blue", "green", "orange"], # ["orange"], #custom_colors_models, 
                                 ax = ax_array[-1] ,#ax_array[1], #None, #ax_array[1], 
                                 xlab = None, #ylab = None, #trace",period_time_list = None,  
                                 number_offset = 1.0, #datetime.timedelta(days = 1),#0.75,
                                 number_fontsize = 20,
                                 period_line_thickness = 7.0,
                                 xlab_fontsize = 10, 
                                 ylab_fontsize = 10,
                                 xticks_fontsize = 10, 
                                 yticks_fontsize = 10,
                                 ylabel_coords = ylabel_coords,
                                 #ylab = None, #"Model posterior max",
                                 #period_time_list = [
                                 #   [datetime.datetime(year = 2003, month = 2, day = 17, hour = 0), 
                                 #   datetime.datetime(year = 2003, month = 2, day = 18, hour = 0)]], 
                                 #label_list = [["1"]], 
                                 #window_len = int(12*7*1),
                                 period_time_list = None, #[[datetime.datetime(year = 2003, month = 2, day = 17, hour = 0), 
                                    #datetime.datetime(year = 2003, month = 2, day = 18, hour = 0)]],
                                 label_list = None, #[["1"]],
                                 SGV = True,
                                 log_det = True,
                                 all_dates = all_dates,
                                 true_CPs = true_CPs)
    
    fig.savefig(res_path + "//model_posteriors_NIPS.pdf",
                    format = "pdf", dpi = 800)


if plot_model_post_and_rld_only:
    #get the subplots
    
    if plot_top and plot_bottom:
        height_ratio = [4,4,4,7]
        fig, ax_array = plt.subplots(4, sharex = True, 
                                     gridspec_kw = {'height_ratios':height_ratio}, 
                                     figsize=(12,7.5)) #12, 5
        plt.subplots_adjust(hspace = .125, left = None, bottom = None, right = None, top = None)
    if (plot_top and not plot_bottom):
        height_ratio = [4,4]
        fig, ax_array = plt.subplots(2, sharex = True, 
                                     gridspec_kw = {'height_ratios':height_ratio}, 
                                     figsize=(12,7.5)) #12, 5
        plt.subplots_adjust(hspace = .125, left = None, bottom = None, right = None, top = None)
    if (not plot_top and plot_bottom):
        height_ratio = [4,4]
        fig, ax_array = plt.subplots(2, sharex = True, 
                                     gridspec_kw = {'height_ratios':height_ratio}, 
                                     figsize=(12,7.5)) #12, 5
        plt.subplots_adjust(hspace = .125, left = None, bottom = None, right = None, top = None)
    
    start = 1
    off = 2
    time_range = np.linspace(start,T-2, T-2-off,dtype = int)
    all_dates = all_dates[-len(time_range):]
    
    KL_CP_color = "gray"
    DPD_CP_color = "darkblue"
    max_color_KL = "red"
    max_color_DPD = "blue"
    max_width = 2.5
    CP_linewidth_DPD = 4
    CP_linewidth_KL = 3
    CP_style_KL = (0,(1.5,0.75))#"-"
    CP_style_DPD = "solid" #(0,(10,4))
    CP_transparence_KL = 0.75
    show_CPs_in_rld_KL = False
    show_CPs_in_rld_DPD = False
    show_CPs = False
    
    ylabel_coords = [-0.04, 0.5]
    
    mod = [9,17,21]
    model_colors_KL = ["green", "blue", "orange"]
    model_colors_DPD = ["blue", "green", "orange"]
    model_linewidths = [2.5]*3
    model_linestyles_KL = [":", "-", "--"]
    model_linestyles_DPD = ["-", ":", "--"]
    
    congestion_linewidth = 5.0
    congestion_linecolor = "purple"
        
    #adapt linestyles & linewidths for model posteriors
    #plot the true CP through all panels! (Maybe draw over plot boundaries?)
    #find out how far we are off the mark in robust BOCPD

    if plot_top:
        EvTKL.plot_model_posterior(indices=mod, #mods, #mods, #relevant_models, 
                                     plot_type = "trace", #"MAPVariance1_trace",
                                     #y_axis_labels = [str(e) for e in all_models],#[#"AR(1)", 
                                                     #"M(5+)", "M(6)", 
                                                     # "M(6+)",
                                                     # "M(7)", "M(7+)"],#relevant_models],
                                     time_range = time_range,
                                     y_axis_labels = [],
                                     log_format=False, aspect = 'auto',
                                     show_MAP_CPs = show_CPs,
                                     #start_plot = 2002.75, stop_plot = 2003.75,
                                     custom_colors = model_colors_KL, 
                                     custom_linewidths = model_linewidths,
                                     custom_linestyles = model_linestyles_KL, 
                                     # ["orange"], #custom_colors_models, 
                                     ax = ax_array[0] ,#ax_array[1], #None, #ax_array[1], 
                                     xlab = None, #ylab = None, #trace",period_time_list = None,  
                                     number_offset = 1.0, #datetime.timedelta(days = 1),#0.75,
                                     number_fontsize = 20,
                                     period_line_thickness = 7.0,
                                     xlab_fontsize = 10, 
                                     ylab_fontsize = 12,
                                     xticks_fontsize = 10, 
                                     yticks_fontsize = 10,
                                     ylabel_coords = ylabel_coords,
                                     #ylab = None, #"Model posterior max",
                                     #period_time_list = [
                                     #   [datetime.datetime(year = 2003, month = 2, day = 17, hour = 0), 
                                     #   datetime.datetime(year = 2003, month = 2, day = 18, hour = 0)]], 
                                     #label_list = [["1"]], 
                                     #window_len = int(12*7*1),
                                     period_time_list = None, #[[datetime.datetime(year = 2003, month = 2, day = 17, hour = 0), 
                                        #datetime.datetime(year = 2003, month = 2, day = 18, hour = 0)]],
                                     label_list = None, #[["1"]],
                                     SGV = True,
                                     log_det = True,
                                     all_dates = all_dates)#,
                                     #true_CPs = true_CPs)
    
    if plot_bottom:
        if not plot_top:
            ax = ax_array[0]
        else:
            ax = ax_array[2]
            
        EvTDPD.plot_model_posterior(indices=[0,1,2], #mods, #mods, #relevant_models, 
                                     plot_type = "trace", #"MAPVariance1_trace",
                                     #y_axis_labels = [str(e) for e in all_models],#[#"AR(1)", 
                                                     #"M(5+)", "M(6)", 
                                                     # "M(6+)",
                                                     # "M(7)", "M(7+)"],#relevant_models],
                                     time_range = time_range,
                                     y_axis_labels = [],
                                     log_format=False, aspect = 'auto',
                                     show_MAP_CPs = show_CPs,
                                     #start_plot = 2002.75, stop_plot = 2003.75,
                                     custom_colors = model_colors_DPD, 
                                     custom_linewidths = model_linewidths,
                                     custom_linestyles = model_linestyles_DPD, 
                                     ax = ax,#ax_array[1], #None, #ax_array[1], 
                                     xlab = None, #ylab = None, #trace",period_time_list = None,  
                                     number_offset = 1.0, #datetime.timedelta(days = 1),#0.75,
                                     number_fontsize = 20,
                                     period_line_thickness = 7.0,
                                     xlab_fontsize = 10, 
                                     ylab_fontsize = 12,
                                     xticks_fontsize = 10, 
                                     yticks_fontsize = 10,
                                     ylabel_coords = ylabel_coords,
                                     #ylab = None, #"Model posterior max",
                                     #period_time_list = [
                                     #   [datetime.datetime(year = 2003, month = 2, day = 17, hour = 0), 
                                     #   datetime.datetime(year = 2003, month = 2, day = 18, hour = 0)]], 
                                     #label_list = [["1"]], 
                                     #window_len = int(12*7*1),
                                     period_time_list = None, #[[datetime.datetime(year = 2003, month = 2, day = 17, hour = 0), 
                                        #datetime.datetime(year = 2003, month = 2, day = 18, hour = 0)]],
                                     label_list = None, #[["1"]],
                                     SGV = True,
                                     log_det = True,
                                     all_dates = all_dates)#,
                                     #true_CPs = true_CPs)

    if plot_top:
        if not plot_bottom:
            ax = ax_array[1]
        else:
            ax = ax_array[1]
        EvTKL.plot_run_length_distr(buffer=0, 
                                    show_MAP_CPs = show_CPs_in_rld_KL, 
                                           mark_median = False, 
            mark_max = True, upper_limit = 70, print_colorbar = False, 
            colorbar_location= None,
            space_to_colorbar = 0.5,
            log_format = False, aspect_ratio = 'auto', 
            #C1=0,C2=700, 
            time_range = time_range, #np.linspace(1,
                          #           T-2, 
                          #           T-2, dtype=int), 
            #start = 1, stop = T, 
            all_dates = all_dates, 
            #event_time_list=[715 ],
            #label_list=["nilometer"], space_to_colorbar = 0.52,
            custom_colors = [KL_CP_color] * 30, 
            custom_linestyles = [CP_style_KL]*30,
            custom_linewidth = CP_linewidth_KL,
            CP_transparence = CP_transparence_KL,
            xlab_fontsize =12,
            ylab_fontsize = 12, 
            ax = ax_array[1], figure = fig,
            no_transform = True,
            date_instructions_formatter = None, 
            date_instructions_locator = None,
            ylabel_coords = ylabel_coords,
            xlab = "Time",
            ylab = "run length", 
            arrow_distance = 25,
            mark_max_linewidth = max_width,
            mark_max_color = max_color_KL)
    
    if plot_bottom:
        if not plot_top:
            ax = ax_array[1]
        else:
            ax= ax_array[3]
        EvTDPD.plot_run_length_distr(buffer=0, 
                                    show_MAP_CPs = show_CPs_in_rld_DPD, 
                                           mark_median = False, 
            mark_max = True, upper_limit = 200, print_colorbar = True, 
            colorbar_location= 'bottom',
            space_to_colorbar = 0.5,
            log_format = False, aspect_ratio = 'auto', 
            #C1=0,C2=700, 
            time_range = time_range, #np.linspace(1,
                          #           T-2, 
                          #           T-2, dtype=int), 
            #start = 1, stop = T, 
            all_dates = all_dates, 
            #event_time_list=[715 ],
            #label_list=["nilometer"], space_to_colorbar = 0.52,
            custom_colors = [KL_CP_color] * 30, 
            custom_linestyles = [CP_style_KL]*30,
            custom_linewidth = CP_linewidth_KL,
            xlab_fontsize =12,
            ylab_fontsize = 12, 
            ax = ax, 
            figure = fig,
            no_transform = True,
            date_instructions_formatter = None, 
            date_instructions_locator = None,
            ylabel_coords = ylabel_coords,
            xlab = "Time",
            ylab = "run length", 
            arrow_distance = 25,
            mark_max_linewidth = max_width,
            mark_max_color = max_color_KL)
    
    #for i in range(0,4):
    ax_array[0].axvline(all_dates[180], 
            color = congestion_linecolor, 
            linewidth = congestion_linewidth, 
            ymin=-0.2,
            ymax=0.98,clip_on=False)
    if (plot_bottom and not plot_top) or (not plot_bottom and plot_top):
        ax_array[1].axvline(all_dates[180], 
                color = congestion_linecolor, 
                linewidth = congestion_linewidth)
    else:
        ax_array[1].axvline(all_dates[180], 
                color = congestion_linecolor, 
                linewidth = congestion_linewidth, 
                ymin=-1.2,
                ymax=1,clip_on=False)
    if (plot_bottom and plot_top):
        ax_array[2].axvline(all_dates[180], 
                color = congestion_linecolor, 
                linewidth = congestion_linewidth, 
                ymin=-0.5,
                ymax=1,clip_on=False)
        ax_array[3].axvline(all_dates[180], 
                color = congestion_linecolor, 
                linewidth = congestion_linewidth)
    
    #plot MAP segmentation as crosses
    if plot_top:
        ax = ax_array[1]
        allCPs = EvTKL.results[6][-2]
        for CP in allCPs[1:]:
            CPloc = CP[0] - start -1
            ax.scatter([all_dates[CPloc]],[80], marker = 'x', 
                    color = "gray",
                    s = 100)
        
    if plot_bottom:
        if not plot_top:
            ax = ax_array[1]
        else:
            ax = ax_array[3]
        DPDCP = EvTDPD.results[6][-2]
        for CP in DPDCP[1:]:
            CPloc = CP[0] - start -1
            ax.scatter([all_dates[CPloc]],[220], marker = 'x', 
                    color = "gray",
                    s = 100)
    
#    fig.savefig(res_path + "//AP_model_posteriors_and_rl_NIPS.pdf",
#                    format = "pdf", dpi = 800)
    if plot_top and not plot_bottom:
        fig.savefig(res_path + "//AP_model_posteriors_and_rl_NIPS_TOPONLY.pdf",
            format = "pdf", dpi = 800)
    if plot_bottom and not plot_top:
        fig.savefig(res_path + "//AP_model_posteriors_and_rl_NIPS_BOTTOMONLY.pdf",
            format = "pdf", dpi = 800)



sd = 40/30

def abs_loss_lim(x, lim):
    x[np.where(x >= lim)] = lim
    return np.abs(x)

train = 50
until = -20
resids = (data[1:,:] - EvTKL.results[10].reshape(366,29)[:-1])[train:until]
SQresids  = np.nansum(np.power(resids,2),axis=1)
KLSQres = SQresids
ABSresids = np.nansum(np.abs(resids),axis=1)
ABSLresids = np.nansum(abs_loss_lim(resids, 2*sd),axis=1)
KLABSres = ABSresids
KLABSLres = ABSLresids
SQE = np.mean(SQresids)
ABS = np.mean(ABSresids)
ABSL = np.mean(ABSLresids)
SQESD = scipy.stats.sem(SQresids)*1.96
ABSSD = scipy.stats.sem(ABSresids)*1.96
print("MSE KL", SQE)
print("SE MSE KL", SQESD)
print("MAE KL", ABS)
print("SE MAE KL", ABSSD)
print("MAEL KL", ABSL)
print("summary MSE KL:", 
      np.mean(np.power((data[1:,:] - EvTKL.results[10].reshape(366,29)[:-1])[train:until],2)))
print("summary MAE KL:", 
      np.mean(np.abs(data[1:,:] - EvTKL.results[10].reshape(366,29)[:-1])[train:until]))

resids = (data[1:,:] - EvTDPD.results[10].reshape(366,29)[:-1])[train:until]
SQresids  = np.nansum(np.power(resids,2),axis=1)
DPDSQres = SQresids
ABSresids = np.nansum(np.abs(resids),axis=1)
ABSLresids = np.nansum(abs_loss_lim(resids, 2*sd),axis=1)
DPDABSLres = ABSLresids
DPDABSres = ABSresids
SQE = np.mean(SQresids[np.where(SQresids > 0)])
ABS = np.mean(ABSresids[np.where(SQresids > 0)])
ABSL = np.mean(ABSLresids)
SQESD = scipy.stats.sem(SQresids[np.where(SQresids > 0)])*1.96
ABSSD = scipy.stats.sem(ABSresids[np.where(SQresids > 0)])*1.96
print("MSE DPD", SQE)
print("SE MSE DPD", SQESD)
print("MAE DPD", ABS)
print("SE MAE DPD", ABSSD)
print("MAEL DPD", ABSL)
print("summary MSE DPD:", 
      np.mean(np.power((data[1:,:] - EvTDPD.results[10].reshape(366,29)[:-1])[train:until],2)))
print("summary MAE DPD:", 
      np.mean(np.abs(data[1:,:] - EvTDPD.results[10].reshape(366,29)[:-1])[train:until]))


#Take a look at difference in pred errors
#EvTKL.plot_prediction_error(data)
#EvTDPD.plot_prediction_error(data)

#Idea: Take a look at bounded loss! I.e., everything beyond 2 standard devs const. otherwise abs.




residsKL = (data - EvTKL.results[10].reshape(366,29))[train:until]
residsDPD = (data - EvTDPD.results[10].reshape(366,29))[train:until]
residsKLSQ = np.power(residsKL, 2)
residsDPDSQ = np.power(residsDPD, 2)
#plt.plot(DPDSQres-KLSQres)
#plt.plot(KLSQres)
#plt.plot((DPDSQres - KLSQres))
#plt.plot((DPDABSres - KLABSres))
#plt.plot(np.sum(np.abs(data), axis=1))
#plt.plot(residsDPDSQ[:,0])
#
#plt.plot(SQResids)
#plt.plot(SQResids)

#fig_4 = EvTDPD.plot_model_posterior(indices=mod, #mods, #mods, #relevant_models, 
#                             plot_type = "BF", #"MAPVariance1_trace",
#                             #y_axis_labels = [str(e) for e in all_models],#[#"AR(1)", 
#                                             #"M(5+)", "M(6)", 
#                                             # "M(6+)",
#                                             # "M(7)", "M(7+)"],#relevant_models],
#                                             time_range = time_range,
#                             log_format=True, aspect = 'auto',
#                             show_MAP_CPs = False,
#                             #start_plot = 2002.7, stop_plot = 2003.7,
#                             custom_colors = ["green"], #custom_colors_models, 
#                             ax = ax_array[2], xlab = None, ylab = "log(BF)", #trace",
#                             period_time_list = None, 
#                             label_list =None, 
#                             number_offset = 0.75,
#                             number_fontsize = 20,
#                             period_line_thickness = 7.0,
#                             xlab_fontsize = 14, ylab_fontsize = 14,
#                             xticks_fontsize = 14, yticks_fontsize = 14,
#                             ylabel_coords = ylabel_coords,
#                             window_len = int(12*7*2),
#                             SGV = False,
#                             log_det = True, 
#                             all_dates = all_dates,
#                             true_CPs = true_CPs)
    
    
    
    
    
    
    
    