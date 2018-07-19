#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 17:50:54 2018

@author: jeremiasknoblauch

Description: Plot pictures for demo/artificial data
"""


import pickle
import numpy as np
import scipy
from Evaluation_tool import EvaluationTool
from matplotlib import pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
import csv
import datetime
import string

#only needed if you want to generate demo data yourself
from cp_probability_model import CpModel 
from detector import Detector 
#import synthetic_simulations_prepare_data    
#import synthetic_simulations_prepare_models

result_path = ("//Users//jeremiasknoblauch//Documents//OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//PaperNIPS//syntheticNIPS")
#date_file = "portfolio_dates.csv"
results_file_KL= ("//KL_K=5_k=1_T=600_2CPs_results_arld=015_ap=02_nolearning"+
                  "_rld_dpd_int=100_shrink=100.txt")
results_file_DPD= ("//DPD_K=5_k=1_T=600_2CPs_results_arld=015_ap=02_" + 
                   "nolearning_rld_dpd_int=100_shrink=100.txt" )

singlePlot = True
doublePlot = False

"""#CREATE THE DATA THAT WAS USED IN SIMULATION#"""

"""STEP 1: Set up the simulation"""
normalize = True
offset = True
max_num_plotted_series = 5
K = 5 #number of series
k = 1   #number of contaminated series
T = 600
burn_in = 100
data = np.zeros((T,K,1))
AR_coefs = [np.ones(K) * (-0.5),
            np.ones(K) * 0.75, #,
            np.ones(K) * -0.7]
levels = [np.ones(K) * 0.3, 
          np.ones(K) * (-0.25), #,
          np.ones(K) * 0.3]
CP_loc = [200,400] 
contamination_df = 4
contamination_scale = np.sqrt(5)

"""STEP 2: Run the simulation with contamination for i<k"""
for cp in range(0, len(CP_loc) + 1):
        #Retrieve the correct number of obs in segment
    if cp == 0:
        T_ = CP_loc[0] + burn_in
        start = 0
        fin = CP_loc[0]
    elif cp==len(CP_loc):
        T_ = T - CP_loc[cp-1]
        start = CP_loc[cp-1]
        fin = T #DEBUG: probably wrong.
    else:
        T_ = CP_loc[cp] - CP_loc[cp-1]
        start = CP_loc[cp-1]
        fin = CP_loc[cp]
        
    #Generate AR(1)  
    for i in range(0,K):
        np.random.seed(i)
        next_AR1 = np.random.normal(0,1,size=T_) 
        for j in range(1, T_):
            next_AR1[j] = next_AR1[j-1]*AR_coefs[cp][i] + next_AR1[j]  + levels[cp][i]
            
        #if i < k, do contamination
        if i<k:
            np.random.seed(i*20)
            contam = contamination_scale*scipy.stats.t.rvs(contamination_df, size=T_)
            contam[np.where(contam <3)] = 0
            next_AR1 = (next_AR1 + contam)
        
        #if first segment, cut off the burn-in
        if cp == 0:
            next_AR1 = next_AR1[burn_in:]
        
        #add the next AR 1 stream into 'data'
        data[start:fin,i,0] = next_AR1

"""STEP 3: Set up analysis parameters"""
S1, S2 = K,1 #S1, S2 give you spatial dimensions
if normalize:
    data = (data - np.mean(data))/np.sqrt(np.var(data))
    
"""STEP 4: Offset"""
if offset:
    for i in range(0, min(K, max_num_plotted_series)):
        data[:,i,:] = data[:,i,:] + i*7
    
    
    

"""#CREATE THE PICTURES USING THE STORED RESULTS#"""

EvTKL = EvaluationTool()
EvTKL.build_EvaluationTool_via_results(result_path + "//" + results_file_KL) 

EvTDPD = EvaluationTool()
EvTDPD.build_EvaluationTool_via_results(result_path + "//" + results_file_DPD) 


"""STEP 1: Set up the plot configs"""

#mpl.rcParams.update(mpl.rcParamsDefault)
#height_ratio =[5,5,5,5,5]
#custom_colors = ["blue", "purple"] 
#fig, ax_array = plt.subplots(5, figsize=(5,5), sharex = True, 
#                             gridspec_kw = {'height_ratios':height_ratio})
#plt.subplots_adjust(hspace = .35, left = None, bottom = None,
#                    right = None, top = None)
if singlePlot:
    fig, ax = plt.subplots(1, figsize=(8,5)) 
    ylabel_coords = [-0.065, 0.5]
elif doublePlot:
    #height_ratio =[3,5]
    width_ratio = [3,5]
    fig, ax_array = plt.subplots(1,2, figsize=(10,3.5), #paper: (10,3.5)
                                 gridspec_kw = {'width_ratios':width_ratio})
    plt.subplots_adjust(hspace = .05, wspace = .175, left = None, bottom = None,
                        right = None, top = None)
    ylabel_coords = [-0.065, 0.5]


#INSERT THE ADDITIONAL CPs KL DECLARES AS VERTICAL LINES

#Plot of raw Time Series
if singlePlot:
    
    EvTKL.plot_raw_TS(data.reshape(T,S1,S2), indices = [0,1,2,3,4], 
            show_MAP_CPs = True, 
            time_range = np.linspace(1,T, T, dtype=int), 
            print_plt = False,
            ylab = "",
            xlab = "Time",
            ax = ax, #ax_array[0],
            #all_dates = np.linspace(622 + 1, 1284, 1284 - (622 + 1), dtype = int),
            custom_colors_series = ["black"]*5,
            custom_colors_CPs = ["red"]* 100,
            custom_linestyles = [":"]*100,
            custom_linewidth = 2,
            ylab_fontsize = 14,
            xlab_fontsize = 14,
            ylabel_coords = ylabel_coords,
            additional_CPs = EvTDPD.results[EvTDPD.names.index("MAP CPs")][-2],
            custom_colors_additional_CPs = ["blue"] * 100,
            custom_linewidth_additional_CPs = 5.0,
            custom_linestyles_additional_CPs = ["-"] * 10)
    
    
    fig.savefig(result_path + "5SeriesRes_solidline.pdf",
                format = "pdf", dpi = 800)
elif doublePlot:
    """Set up the influence plot first"""
    
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
    data_ = np.zeros((num_rows, num_cols))
    count = 0
    for entry, count in zip(raw_data, range(0, int(len(raw_data)))):
        ind_col = count % 5
        ind_row = int(count/5)
        data_[ind_row, ind_col] = entry

    """STEP 2: Plot"""
    xlabsize, ylabsize, legendsize = 12, 12, 11
    linewidths = [2]*5
    linestyles = [0,"-",":", "--", "-."]
    linecolors = [0, "navy", "purple", "red", "orange"]
    #ax, fig = plt.subplots(1, figsize = (3.5,4.5))
    ax = ax_array[0]
    
    handles, labels = ax.get_legend_handles_labels()
    
    for i in range(1, 5):
        handle, = ax.plot(data_[:,0], data_[:,i], linewidth = linewidths[i],
                           linestyle = linestyles[i],
                           color = linecolors[i])
        handles.append(handle)
    ax.set_xlabel("Standard Deviations", size = xlabsize)
    ax.set_ylabel("Influence", size = ylabsize)
    labels = ["KLD", r'$\beta=0.05$', r'$\beta=0.2$',r'$\beta=0.25$']
    ax.legend(handles, labels, prop = {'size':legendsize})
    ax.text(-0.11, 0.98, string.ascii_uppercase[0], transform=ax.transAxes, 
            size=20, weight='bold')
    
    """STEP 3: Second plot"""
    EvTKL.plot_raw_TS(data.reshape(T,S1*S2), indices = [0,1,2,3,4], 
            show_MAP_CPs = True, 
            time_range = np.linspace(1,T, T, dtype=int), 
            print_plt = False,
            ylab = "Value",
            xlab = "Time",
            ax = ax_array[1], #ax_array[0],
            #all_dates = np.linspace(622 + 1, 1284, 1284 - (622 + 1), dtype = int),
            custom_colors_series = ["black"]*5,
            custom_colors_CPs = ["red"]* 100,
            custom_linestyles = [":"]*100,
            custom_linewidth = 2,
            ylab_fontsize = ylabsize,
            xlab_fontsize = xlabsize,
            ylabel_coords = ylabel_coords,
            additional_CPs = EvTDPD.results[EvTDPD.names.index("MAP CPs")][-2],
            custom_colors_additional_CPs = ["blue"] * 100,
            custom_linewidth_additional_CPs = 2.5,
            custom_linestyles_additional_CPs = ["-"] * 10)
    ax_array[1].text(-0.07, 0.98, string.ascii_uppercase[1], 
            transform=ax_array[1].transAxes, 
            size=20, weight='bold')
    
    fig.savefig(result_path + "InfluenceAndAR5.pdf",
                format = "pdf", dpi = 800)
    fig.savefig(result_path + "InfluenceAndAR5.jpg",
            format = "jpg", dpi = 800)

