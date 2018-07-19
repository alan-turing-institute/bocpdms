#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 12:03:35 2018

@author: jeremiasknoblauch

Description: Produces paper pictures for the well log data (NIPS18)
"""

import pickle
import numpy as np
import scipy
from Evaluation_tool import EvaluationTool
from matplotlib import pyplot as plt
import csv
import datetime


data_directory = ("//Users//jeremiasknoblauch//Documents//OxWaSP"+
    "//BOCPDMS/Code//SpatialBOCD//Data//well log") 
well_file = data_directory + "//well.txt"

result_path = ("//Users//jeremiasknoblauch//Documents//OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//PaperNIPS//well log")
#date_file = "portfolio_dates.csv"
results_file_KL= ("//well_log_KL_int=100.txt")
results_file_DPD= ("//well_log_DPD.txt" )


plot1 = False
plot2 = False
plot3 = True #plot only well log without anything else

"""STEP 1: Read in raw data"""
raw_data = []
count = 0 
with open(well_file) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        raw_data += row

raw_data_float = []
for entry in raw_data:
    raw_data_float.append(float(entry))
raw_data = raw_data_float

T = int(len(raw_data))
S1, S2 = 1,1 #S1, S2 give you spatial dimensions
data = np.array(raw_data).reshape(T,1,1)

"""STEP 2: Read in the data to create your EvT objects"""
EvTKL = EvaluationTool()
EvTKL.build_EvaluationTool_via_results(result_path + "//" + results_file_KL) 

EvTDPD = EvaluationTool()
EvTDPD.build_EvaluationTool_via_results(result_path + "//" + results_file_DPD) 


if plot1:
    """STEP 3: Set up the figure properties and plot"""
    height_ratio =[8,10] #10,10
    custom_colors = ["blue", "purple"] 
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2, 2, 
                                 figsize=(12,5), 
                                 sharex = 'col', 
                                 sharey = 'row',# True, 
                                 gridspec_kw = {'height_ratios':height_ratio})
    ax_arrays = np.array([np.array([ax1, ax2]), np.array([ax3, ax4])])
    
    plt.subplots_adjust(hspace = .05, wspace = .05, left = None, bottom = None,
                        right = None, top = None)
    
    ylabel_coords = [-0.15, 0.5]
    
    #for (ax_array, EvT) in zip(ax_arrays, (EvTKL, EvTDPD)):
        #Plot of raw Time Series
    EvTKL.plot_raw_TS(data.reshape(T,1), indices = [0], 
            xlab = None, 
            show_MAP_CPs = True, 
            time_range = np.linspace(1,T, T, dtype=int), 
            print_plt = False,
            ylab = "Nuclear Response", 
            ax = ax_arrays[0][0],
            #all_dates = np.linspace(622 + 1, 1284, 1284 - (622 + 1), dtype = int),
            custom_colors_series = ["black"],
            custom_colors_CPs = ["blue", "blue"]* 100,
            custom_linestyles = [":"]*100,
            custom_linewidth = 1.5,
            ylab_fontsize = 12,
            ylabel_coords = ylabel_coords)
    
    EvTDPD.plot_raw_TS(data.reshape(T,1), indices = [0], 
            xlab = "Time", 
            show_MAP_CPs = True, 
            time_range = np.linspace(1,T, T, dtype=int), 
            print_plt = False,
            ylab = "", 
            ax = ax_arrays[0][1],
            #all_dates = np.linspace(622 + 1, 1284, 1284 - (622 + 1), dtype = int),
            custom_colors_series = ["black"],
            custom_colors_CPs = ["blue", "blue"]* 100,
            custom_linestyles = [":"]*100,
            custom_linewidth = 1.5,
            ylab_fontsize = 12,
            ylabel_coords = ylabel_coords)
                               
    #Run length distribution plot
    EvTKL.plot_run_length_distr(buffer=0, show_MAP_CPs = True, 
                                       mark_median = False, 
        mark_max = True, upper_limit = 1400, print_colorbar = True, 
        colorbar_location= 'bottom',
        space_to_colorbar = 0.5,
        log_format = False, aspect_ratio = 'auto', 
        #C1=0,C2=700, 
        time_range = np.linspace(1,
                                 T-2, 
                                 T-2, dtype=int), 
        start = 1, stop = T, 
        all_dates = None, 
        #event_time_list=[715 ],
        #label_list=["nilometer"], space_to_colorbar = 0.52,
        custom_colors = ["blue", "blue"] * 30, 
        custom_linestyles = [":"]*30,
        custom_linewidth = 1.5,
        #arrow_colors= ["black"],
        #number_fontsize = 14,
        #arrow_length = 135,
        #arrow_thickness = 3.0,
        xlab_fontsize =12,
        ylab_fontsize = 12, 
        #arrows_setleft_indices = [0],
        #arrows_setleft_by = [50],
        #zero_distance = 0.0,
        ax = ax_arrays[1][0], figure = fig,
        no_transform = True,
        date_instructions_formatter = None, 
        date_instructions_locator = None,
        ylabel_coords = ylabel_coords,
        xlab = "Time",
        #ylab="",
        arrow_distance = 25,
        mark_max_linewidth = 0.8)
        
    #Run length distribution plot
    EvTDPD.plot_run_length_distr(buffer=0, show_MAP_CPs = True, 
                                       mark_median = False, 
        mark_max = True, upper_limit = 1400, print_colorbar = True, 
        colorbar_location= 'bottom',
        space_to_colorbar = 0.5,
        log_format = False, aspect_ratio = 'auto', 
        #C1=0,C2=700, 
        time_range = np.linspace(1,
                                 T-2, 
                                 T-2, dtype=int), 
        start = 1, stop = T, 
        all_dates = None, 
        #event_time_list=[715 ],
        #label_list=["nilometer"], space_to_colorbar = 0.52,
        custom_colors = ["blue", "blue"] * 30, 
        custom_linestyles = [":"]*30,
        custom_linewidth = 1.5,
        #arrow_colors= ["black"],
        #number_fontsize = 14,
        #arrow_length = 135,
        #arrow_thickness = 3.0,
        xlab_fontsize =12,
        ylab_fontsize = 12, 
        #arrows_setleft_indices = [0],
        #arrows_setleft_by = [50],
        #zero_distance = 0.0,
        ax = ax_arrays[1][1], figure = fig,
        no_transform = True,
        date_instructions_formatter = None, 
        date_instructions_locator = None,
        ylabel_coords = ylabel_coords,
        xlab = "Time",
        ylab = "",
        arrow_distance = 25,
        mark_max_linewidth = 1.5)
        
    
    fig.savefig(result_path + "//KL_DPD_CPs_and_RL.pdf",
                format = "pdf", dpi = 800)    
    

#rld extraction + looking at how often it drops to 0
maxima = []
rld = EvTKL.results[EvTKL.names.index("all run length log distributions")]
for rld_ in rld:
    index = np.argmax(rld_)
    maxima.append(index)
print("number of 0 rl drops is", maxima.count(0))
print("there are 11 CPs, so falsely labelled ones are", maxima.count(0) - 11)




"""different plot: Overlay the CPs, put the RLD underneath one another.
     Use different colors for the additional CPs and the rld maximum"""

if plot2:
    """STEP 1: Get the different CPs"""
    CPsDPD = np.array([e[0] for e in EvTDPD.results[EvTDPD.names.index("MAP CPs")][-2]])
    CPsKL = np.array([e[0] for e in EvTKL.results[EvTKL.names.index("MAP CPs")][-2]])
    
    k  = 25
    additional_CPs = []
    for cp_kl in CPsKL:
        lower = CPsDPD - k < cp_kl
        upper = CPsDPD + k > cp_kl
        if (np.any(lower == upper)):
            print("KL cp fine at ", cp_kl)
        else:
            additional_CPs.append([cp_kl,0])
    
    height_ratio =[10,4,4] #10,10
    #custom_colors = ["blue", "purple"] 
    
    KL_CP_color = "crimson"
    DPD_CP_color = "darkblue"
    max_color_KL = "red"
    max_color_DPD = "blue"
    max_width = 3
    CP_linewidth_DPD = 3
    CP_linewidth_KL = 3
    CP_style_KL = (0,(1,2.25))#"-"
    CP_style_DPD = "solid" #(0,(10,4))
    CP_transparence_KL = 0.75
    CP_transparence_DPD = 0.5
    show_CPs_in_rld = False
    
    xlabsize, ylabsize, ticksize = 25, 25, 22
    
    fig, ax_array = plt.subplots(3,  
                                 figsize=(18,7), 
                                 sharex = True, 
                                 #sharey = 'row',# True, 
                                 gridspec_kw = {'height_ratios':height_ratio})
    #ax_arrays = np.array([np.array([ax1, ax2]), np.array([ax3, ax4])])
    
    plt.subplots_adjust(hspace = .05, #wspace = .05, 
                        left = None, bottom = None,
                        right = None, top = None)
    
    ylabel_coords = [-0.1, 0.5]
    
    
    EvTDPD.plot_raw_TS(data.reshape(T,S1*S2), indices = [0], xlab = None, 
            show_MAP_CPs = True, 
            time_range = np.linspace(1,T, T, dtype=int), 
            print_plt = False,
            ylab = "Response", 
            ax = ax_array[0], #ax_array[0],
            #all_dates = np.linspace(622 + 1, 1284, 1284 - (622 + 1), dtype = int),
            custom_colors_series = ["black"]*5,
            custom_colors_CPs = [DPD_CP_color]* 100,
            custom_linestyles = [CP_style_DPD]*100,
            custom_linewidth = CP_linewidth_DPD,
            custom_transparency = CP_transparence_DPD,
            ylab_fontsize = ylabsize,
            yticks_fontsize = ticksize,
            ylabel_coords = ylabel_coords,
            additional_CPs = additional_CPs,
            custom_colors_additional_CPs = [KL_CP_color] * 100,
            custom_linewidth_additional_CPs = CP_linewidth_KL,
            custom_linestyles_additional_CPs = [CP_style_KL] * 10,
            custom_transparency_additional_CPs = CP_transparence_KL)
    
    
    EvTDPD.plot_run_length_distr(buffer=0, show_MAP_CPs = show_CPs_in_rld, 
                                       mark_median = False, 
        mark_max = True, 
        upper_limit = 1300, 
        print_colorbar = False, 
        colorbar_location= None,
        xlab = "",
        ylab = "", 
        #space_to_colorbar = 0.5,
        log_format = False, aspect_ratio = 'auto', 
        #C1=0,C2=700, 
        time_range = np.linspace(1,
                                 T-2, 
                                 T-2, dtype=int), 
        start = 1, stop = T, 
        all_dates = None, 
        custom_colors = [DPD_CP_color] * 30, 
        custom_linestyles = [CP_style_DPD]*30,
        custom_linewidth = CP_linewidth_DPD,
        xlab_fontsize = xlabsize,
        ylab_fontsize = ylabsize, 
        xticks_fontsize = ticksize,
        yticks_fontsize = ticksize,
        ax = ax_array[1], figure = fig,
        no_transform = True,
        date_instructions_formatter = None, 
        date_instructions_locator = None,
        ylabel_coords = ylabel_coords,
        #xlab = "Time",
        #ylab = "",
        arrow_distance = 25,
        mark_max_linewidth = max_width,
        mark_max_color = max_color_DPD)
    
    
    EvTKL.plot_run_length_distr(buffer=0, show_MAP_CPs = show_CPs_in_rld, 
                                       mark_median = False, 
        mark_max = True, upper_limit = 1200, print_colorbar =  False, #True, 
        colorbar_location=  None, #'bottom',
        space_to_colorbar = 0.9,
        log_format = False, aspect_ratio = 'auto', 
        #C1=0,C2=700, 
        time_range = np.linspace(1,
                                 T-2, 
                                 T-2, dtype=int), 
        start = 1, stop = T, 
        all_dates = None, 
        #event_time_list=[715 ],
        #label_list=["nilometer"], space_to_colorbar = 0.52,
        custom_colors = [KL_CP_color] * 30, 
        custom_linestyles = [CP_style_KL]*30,
        custom_linewidth = CP_linewidth_KL,
        xlab_fontsize =xlabsize,
        ylab_fontsize = ylabsize, 
        xticks_fontsize = ticksize,
        yticks_fontsize = ticksize,
        ylabel_coords = [-0.1, 1.25],
        ax = ax_array[2], figure = fig,
        no_transform = True,
        date_instructions_formatter = None, 
        date_instructions_locator = None,
        #ylabel_coords = ylabel_coords,
        xlab = "Time",
        ylab = "run lengths", 
        arrow_distance = 25,
        mark_max_linewidth = max_width,
        mark_max_color = max_color_KL)
        
        
    fig.savefig(result_path + "//KL_DPD_CPs_one_plot.pdf",
                format = "pdf", dpi = 800)  
    fig.savefig(result_path + "//wellpic.jpg",
            format = "jpg", dpi = 800)  

if plot3:
    xlabsize, ylabsize, ticksize = 18, 18, 16
    
    fig, ax = plt.subplots(1,  
                                 figsize=(12,5)
                                 )
                                 #sharex = True, 
                                 #sharey = 'row',# True, 
                                 #gridspec_kw = {'height_ratios':height_ratio})
    #ax_arrays = np.array([np.array([ax1, ax2]), np.array([ax3, ax4])])
    
    plt.subplots_adjust(hspace = .05, #wspace = .05, 
                        left = None, bottom = None,
                        right = None, top = None)
    
    ylabel_coords = [-0.11, 0.5]
    
    
    EvTDPD.plot_raw_TS(data.reshape(T,S1*S2), indices = [0], 
            show_MAP_CPs = False, 
            time_range = np.linspace(1,T, T, dtype=int), 
            print_plt = False,
            ylab = "Nuclear Response", 
            xlab = "Time",
            ax = ax, #ax_array[0],
            #all_dates = np.linspace(622 + 1, 1284, 1284 - (622 + 1), dtype = int),
            custom_colors_series = ["black"]*5,
            ylab_fontsize = ylabsize,
            xlab_fontsize = xlabsize,
            yticks_fontsize = ticksize,
            xticks_fontsize = ticksize,
            ylabel_coords = ylabel_coords,
            additional_CPs = None)
    
    fig.savefig(result_path + "//well_only_data.pdf",
                format = "pdf", dpi = 800)  


def abs_loss_lim(x, lim):
    x[np.where(x >= lim)] = lim
    return np.abs(x)

sd =  np.sqrt(np.var(data))

train = 0
until = -2
resids = (data[1:] - EvTKL.results[10].reshape(T,1)[:-1])[train:until]
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
      np.mean(np.power((data[1:] - EvTKL.results[10].reshape(T,1)[:-1])[train:until],2)))
print("summary MAE KL:", 
      np.mean(np.abs((data[1:] - EvTKL.results[10].reshape(T,1)[:-1])[train:until])))

resids = (data - EvTDPD.results[10].reshape(T,1)[:-1])[train:until]
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
      np.mean(np.power(((data - EvTDPD.results[10].reshape(T,1)[:-1]))[train:until],2)))
print("summary MAE DPD:", 
      np.mean(np.abs(((data - EvTDPD.results[10].reshape(T,1)[:-1]))[train:until])))

