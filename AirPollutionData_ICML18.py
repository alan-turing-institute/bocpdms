#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 12:19:49 2018

@author: jeremiasknoblauch

Description: process london air pollution data around the CP
             at 17/02/2003 (= Introduction of congestion charge)
"""

import numpy as np
import os
import csv
import matplotlib.pyplot as plt

from BVAR_NIG import BVARNIG
from detector import Detector
from Evaluation_tool import EvaluationTool
from cp_probability_model import CpModel



run_detectors = True    #whether we want to run detector or just read data
normalize = True        #normalize station-wise
daily_avg = True        #use daily averages (vs 2h-averages)
deseasonalize_2h = False    #only useful for 2h-averages. Will deseasonalize
                            #for each weekly 2h-interval ( = 12 * 7 intervals)
if daily_avg:
    deseasonalize_2h = False    #if daily_avg is True, 2h-deseasonalizing makes
                                #no sense
deseasonalize_day = True #only one of the two deseasonalizations should be 
                        #chosen, and this one means that we only take weekday 
                        #averages
shortened, shortened_to = False, 500 #wheter to process only the first 
                                     #shortened_to observations and stop then

"""folder containing dates and data (with result folders being created at 
run-time if necessary)"""
baseline_working_directory = ("//Users//jeremiasknoblauch//Documents//OxWaSP"+
    "//BOCPDMS//Code//SpatialBOCD//Data//AirPollutionData")

"""subset of the Airpollution data analyzed"""
cp_type = "CongestionChargeData" #only option available, originally wanted to
                                    #look at another time frame but didn't
                                    
"""Distance matrices computed using symmetrized road distances (i.e., taking
d(i,j) = 0.5*[d_road(i,j) + d_road(j,i)] and euclidean distances"""
dist_file_road = (baseline_working_directory + "//" + cp_type + "//" + 
                  "RoadDistanceMatrix_")
dist_file_euclid = (baseline_working_directory + "//" + cp_type + "//" + 
                   "EuclideanDistanceMatrix_")

"""File prototype for 2h-averaged station data from 08/17/2002 - 08/17/2003"""
prototype = "_081702-081703_2h.txt"

"""Decide if you want to take the bigger or smaller set of stations for the
analysis"""
mode = "bigger" #bigger, smaller (bigger contains more filled-in values)


"""These indices are used for reading in the station data of each station"""
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


"""STEP 2: Convert distance matrices to nbhs. Cutoffs define the concentric
rings around the stations in the road-distance or euclidean space"""
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
    file_name = (baseline_working_directory + "//" + cp_type + "//" + 
                 id_ + prototype)
    
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


"""Deseasonalize based on week-day averages if we have 2h-frequency"""
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

"""Deseasonalize based on week-day averages if we have 24h-frequency"""                
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
              

"""Deseasonalize based on 2h averages if we have 24h-frequency"""                    
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

"""normalize the data"""    
if normalize:
    data = (data - np.mean(data, axis=0))/np.sqrt(np.var(data,axis=0))
    intercept_priors = np.mean(data,axis=0)
    
    
"""STEP 6: Select the priors"""
prior_mean_scale = 0.0
intensity_list = [180]
var_scale_list = [0.0005]
a_prior_list =  [100]
b_prior_list = [25]
AR_selections = [1] #[1,2,3,4,5]
res_seq_list = [
        [[0]],
        [[0,1]],
        [[0,1,2]],
        [[0]]*2,
        [[0,1]]*2,
        [[0]]*3,
        [[0,1]]*3,
        [[0]]*4,
        [[0,1]]*4,
        [[0]]*5,
        [[0,1]]*5
        ]

#REPORTED IN ICML SUBMISSION:
# a = 100, b = 25, bvp = 0.0005, intensity = 180, models: AR(1-5), 
# [[0]]*1-5, [[0,1]]*1-5, daily data, deseasonalized


"""STEP 7: Intercept grouping"""
grouping = np.zeros((S1*S2, S1*S2))
for i in range(0, S1*S2):
    grouping[i,i]=1
grouping = grouping.reshape((S1*S2, S1,S2))

    

"""STEP 8: Build models and run algo"""
for intensity in intensity_list:
        for var_scale in var_scale_list:
            for a in a_prior_list:
                for b in b_prior_list:
                    cp_model = CpModel(intensity)
                    
                    """Create models"""
                    all_models = []
                    
                    """STEP 8.2: build AR models"""
                    AR_models = []
                    for lag in AR_selections:
                        AR_models += [BVARNIG(
                                        prior_a = a,prior_b = b,
                                        S1 = S1,S2 = S2,
                                        prior_mean_scale = prior_mean_scale,
                                        prior_var_scale = var_scale,
                                        intercept_grouping = grouping,
                                        general_nbh_sequence = np.array([[[]]*lag]*S2*S2), 
                                        general_nbh_restriction_sequence = np.array([[[0]]*lag]*S2*S2),
                                        general_nbh_coupling = "weak coupling",
                                        hyperparameter_optimization = hyperpar_opt)]
                    all_models = all_models + AR_models                   
                    
                    """STEP 6.3: build model universe entries with nbhs"""                                       
                    VAR_models_weak = []
                    for res in res_seq_list:
                        VAR_models_weak += [BVARNIG(
                                        prior_a = a,prior_b = b,
                                        S1 = S1,S2 = S2,
                                        prior_mean_scale = prior_mean_scale,
                                        prior_var_scale = var_scale,
                                        intercept_grouping = grouping,
                                        general_nbh_sequence = euclid_nbhs,
                                        general_nbh_restriction_sequence = res,
                                        general_nbh_coupling = "weak coupling",
                                        hyperparameter_optimization = hyperpar_opt)]
                        VAR_models_weak += [BVARNIG(
                                        prior_a = a,prior_b = b,
                                        S1 = S1,S2 = S2,
                                        prior_mean_scale = prior_mean_scale,
                                        prior_var_scale = var_scale,
                                        intercept_grouping = grouping,
                                        general_nbh_sequence = road_nbhs,
                                        general_nbh_restriction_sequence = res,
                                        general_nbh_coupling = "weak coupling",
                                        hyperparameter_optimization = hyperpar_opt)]
                        if not normalize:
                            VAR_models_weak[-1].prior_mean_beta = np.append(
                                intercept_priors, np.array([0]*VAR_models_weak[-1].
                                                        num_endo_regressors))
                            VAR_models_weak[-2].prior_mean_beta = np.append(
                                intercept_priors, np.array([0]*VAR_models_weak[-1].
                                                        num_endo_regressors))
                    all_models = all_models + VAR_models_weak
                        
                    
                    
                    model_universe = np.array(all_models)#AR_models)
                    model_prior = np.array([1/len(model_universe)]*len(model_universe))
        
                    """Build and run detector"""
                    detector = Detector(data=data, model_universe=model_universe, 
                            model_prior = model_prior,
                            cp_model = cp_model, S1 = S1, S2 = S2, T = T, 
                            store_rl=True, store_mrl=True,
                            trim_type="keep_K", threshold = 75,
                            notifications = 50,
                            save_performance_indicators = True,
                            training_period = 250)
                    detector.run()
                    
                    """Store results + real CPs into EvaluationTool obj"""
                    EvT = EvaluationTool()
                    EvT.build_EvaluationTool_via_run_detector(detector)
                            
                    """store that EvT object onto hard drive"""
                    prior_spec_str = ("//" + cp_type + "//a=" + str(a) + 
                        "//b=" + str(b) )
                    detector_path = (baseline_working_directory  + 
                                prior_spec_str + "//daily_good_recreated")
                    if not os.path.exists(detector_path):
                        os.makedirs(detector_path)
                    
                    results_path = detector_path + "//results.txt" 
                    EvT.store_results_to_HD(results_path)
                    
                    fig = EvT.plot_predictions(
                            indices = [0], print_plt = True, 
                            legend = False, 
                            legend_labels = None, 
                            legend_position = None, 
                            time_range = None,
                            show_var = False, 
                            show_CPs = True)
                    plt.close(fig)
                    fig = EvT.plot_run_length_distr(
                        print_plt = True, 
                        time_range = None,
                        show_MAP_CPs = True, 
                        show_real_CPs = False,
                        mark_median = False, 
                        log_format = True,
                        CP_legend = False, 
                        buffer = 50)
                    plt.close(fig)
                    
                    print("MSE", np.mean(detector.MSE))
                    print("NLL", np.mean(detector.negative_log_likelihood))
                    print("a", a)
                    print("b", b)
                    print("intensity", intensity)
                    print("beta var prior", var_scale )





