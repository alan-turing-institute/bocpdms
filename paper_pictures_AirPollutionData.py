#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 17:03:07 2018

@author: jeremiasknoblauch

Description: Plots pics from Air Pollution Data London
"""

import csv
import numpy as np
from Evaluation_tool import EvaluationTool
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import datetime
import matplotlib

#ensure that we have type 1 fonts (for ICML publishing guiedlines)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

"""""STEP 1: DATA TRANSFOMRATIONS"""""
normalize = True
deseasonalize_2h = True
deseasonalize_day = True #only one of the two deseasonalizations should be chosen
shortened, shortened_to = False, 500
daily_avg = True
if daily_avg:
    deseasonalize_2h = False


data_dir = ("//Users//jeremiasknoblauch//Documents//OxWaSP//BOCPDMS//" + 
            "//Code//SpatialBOCD//Data//AirPollutionData")
cp_type = "CongestionChargeData"
dist_file_road = (data_dir + "//" + cp_type + "//" + 
                  "RoadDistanceMatrix_")
dist_file_euclid = (data_dir + "//" + cp_type + "//" + 
                   "EuclideanDistanceMatrix_")
results_file = ("/Users//jeremiasknoblauch//Documents////OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//Paper//AirPollutionData//" + 
    "results_daily.txt")
res_path = ("/Users//jeremiasknoblauch//Documents////OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//Paper//AirPollutionData//")
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
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_results(results_file) 

segmentation = EvT.results[EvT.names.index("MAP CPs")][-2]
model_labels = EvT.results[EvT.names.index("model labels")]
num_models = len(np.union1d(model_labels, model_labels))
relevant_models = np.union1d([seg[1] for seg in segmentation],[seg[1] for seg in segmentation])
#mods = [8,11,13,17,18]
all_models = [e for e in range(0, len(model_labels))] #np.linspace(0, len(model_labels)-1, len(model_labels), dtype = int)


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
#poster: height_ratio, num_subplots = [4,3,4],3
height_ratio, num_subplots = [4,3,5],3
#paper: ylabel_coords = [-0.085, 0.5]
#poster:  [-0.06, 0.5]
ylabel_coords = [-0.085, 0.5]

#paper: figsize = (8,5) #for poster: 12,5
fig, ax_array = plt.subplots(num_subplots, sharex = True, 
                             gridspec_kw = {'height_ratios':height_ratio}, 
                             figsize=(8,5))
plt.subplots_adjust(hspace = .2, left = None, bottom = None, right = None, top = None)

off = 5
time_range = np.linspace(10,T-2, T-2-off,dtype = int)
all_dates = all_dates[-len(time_range):]

fig_1 = EvT.plot_raw_TS(data[-len(time_range):,:].reshape(len(time_range), 29), all_dates = all_dates, ax = ax_array[0], 
                        time_range = time_range,
                        custom_colors_series = ["black"]*10,
                        ylab_fontsize = 14,
                        yticks_fontsize = 14,
                        ylab = "NOX",
                        xlab=None,
                        ylabel_coords = ylabel_coords,
                        true_CPs = true_CPs)

mod = [17,21]
EvT.plot_model_posterior(indices=mod, #mods, #mods, #relevant_models, 
                             plot_type = "trace", #"MAPVariance1_trace",
                             #y_axis_labels = [str(e) for e in all_models],#[#"AR(1)", 
                                             #"M(5+)", "M(6)", 
                                             # "M(6+)",
                                             # "M(7)", "M(7+)"],#relevant_models],
                             time_range = time_range,
                             y_axis_labels = [],
                             log_format=False, aspect = 'auto',
                             show_MAP_CPs = False,
                             #start_plot = 2002.75, stop_plot = 2003.75,
                             custom_colors = ["blue", "orange"], # ["orange"], #custom_colors_models, 
                             ax = ax_array[1] ,#ax_array[1], #None, #ax_array[1], 
                             xlab = None, #ylab = None, #trace",period_time_list = None,  
                             number_offset = 1.0, #datetime.timedelta(days = 1),#0.75,
                             number_fontsize = 20,
                             period_line_thickness = 7.0,
                             xlab_fontsize = 14, ylab_fontsize = 14,
                             xticks_fontsize = 14, yticks_fontsize = 14,
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

fig_4 = EvT.plot_model_posterior(indices=mod, #mods, #mods, #relevant_models, 
                             plot_type = "BF", #"MAPVariance1_trace",
                             #y_axis_labels = [str(e) for e in all_models],#[#"AR(1)", 
                                             #"M(5+)", "M(6)", 
                                             # "M(6+)",
                                             # "M(7)", "M(7+)"],#relevant_models],
                             time_range = time_range,
                             log_format=True, aspect = 'auto',
                             show_MAP_CPs = False,
                             #start_plot = 2002.7, stop_plot = 2003.7,
                             custom_colors = ["green"], #custom_colors_models, 
                             ax = ax_array[2], xlab = None, ylab = "log(BF)", #trace",
                             period_time_list = None, 
                             label_list =None, 
                             number_offset = 0.75,
                             number_fontsize = 20,
                             period_line_thickness = 7.0,
                             xlab_fontsize = 14, ylab_fontsize = 14,
                             xticks_fontsize = 14, yticks_fontsize = 14,
                             ylabel_coords = ylabel_coords,
                             window_len = int(12*7*2),
                             SGV = False,
                             log_det = True, 
                             all_dates = all_dates,
                             true_CPs = true_CPs
                             )

fig.savefig(res_path + "APData.pdf")