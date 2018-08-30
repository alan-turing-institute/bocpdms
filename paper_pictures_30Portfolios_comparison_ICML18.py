#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 20:17:54 2018

@author: jeremiasknoblauch

Description: Produce plots based on 30PF data (run: 'comparison')
"""

import numpy as np
from Evaluation_tool import EvaluationTool
from matplotlib import pyplot as plt
import csv
import datetime
import os
import matplotlib
from matplotlib import rcParams

# Ensure that we have type 1 fonts (for ICML publishing guidelines)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

baseline_working_directory = os.getcwd()
data_directory = os.path.join(baseline_working_directory, "Data", "30PF")
results_directory = os.path.join(baseline_working_directory, "Output", "30PF", "time_frame=comparison",
                                 "transform=True", "a=100", "b=0.001")
date_file = "portfolio_dates.csv"
results_file = "results_30portfolios.txt"

"""Read in the dates & results"""
myl = []
count = 0 
with open(os.path.join(data_directory, date_file)) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        #Don't read in column header
        if count > 0:
            myl += row
        count += 1
        if count % 2500 == 0:
            print(count)

"""Dates for the x-axis"""
dates = []
for e in myl:
    dates.append(int(e))

"""Build the EvaluationTool from the results you have saved"""                              
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_results(os.path.join(results_directory, results_file))

"""Using the dates, select your range and indices: select 
03/07/1975 -- 31/12/2008, i.e. find indices that correspond"""
start_test = dates.index(19740703)
start_algo = dates.index(19750703)
stop = dates.index(20081231)

"""time period for which we want RLD"""
start_comparison = dates.index(19980102)
stop_comparison = stop

all_dates = []
for d in dates[start_comparison:stop-2]:
    s = str(d)
    all_dates.append(datetime.date(year=int(s[0:4]), 
                                   month=int(s[4:6]), day=int(s[6:8])))

"""indices of that time period"""
start_range = start_comparison - start_test
stop_range = stop_comparison - start_test
time_range = np.linspace(start_range, stop_range, 
                         stop_range - start_range, dtype = int)

"""Create list of event times and labels"""
label_list, event_time_list= [],[]
number_labels = True

"""Color for the arrows"""
color_us = "orange"
color_they = "black"
arrow_color_list = []

"""helper function getting closest date in all_dates"""
def nearest(pivot):
    return min(all_dates, key=lambda x: abs(x - pivot))

"""Get the CP dates"""

label_list.append("Asia Crisis")
#date: last quarter 98
event_time_list.append(nearest(datetime.date(1998, 9, 12)))
arrow_color_list.append(color_they)

label_list.append("DotCom bubble bursts")
#date: April, when inflation reports came in
event_time_list.append(nearest(datetime.date(2000, 4, 12))) 
arrow_color_list.append(color_they)

label_list.append("OPEC cuts output by 4%, Japanese central bank starts QE")
#date: announcement of output cut
event_time_list.append(nearest(datetime.date(2001,3,19))) 
arrow_color_list.append(color_us)

label_list.append("9/11")
#date: airplane day
event_time_list.append(nearest(datetime.date(2001, 9,  9))) 
arrow_color_list.append(color_they)

label_list.append("Afghanistan war")
#date: Operation Enduring Freedom begins
event_time_list.append(nearest(datetime.date(2001, 10, 7))) 
arrow_color_list.append(color_us)

label_list.append("2002 Crash")
#date: Crash of 2002, 23/7 lowest Dow Jones value
event_time_list.append(nearest(datetime.date(2002, 7, 23))) 
arrow_color_list.append(color_us)

label_list.append("Bali bombing attack")
#date: day bomb went up
event_time_list.append(nearest(datetime.date(2002, 10,13))) 
arrow_color_list.append(color_us)

label_list.append("Iraq war")
#date: start of invasion
event_time_list.append(nearest(datetime.date(2003,3,20))) 
arrow_color_list.append(color_us)   

label_list.append("Bush cuts")
#date: day regulation was passed
event_time_list.append(nearest(datetime.date(2003, 5, 23))) 
arrow_color_list.append(color_they)  

label_list.append("US Election")
#date: election day
event_time_list.append(nearest(datetime.date(2005,11,8))) 
arrow_color_list.append(color_they)

label_list.append("Iran announces that it has successfully enriched Uranium")
#date: day of announcement
event_time_list.append(nearest(datetime.date(2006, 7, 5)))
arrow_color_list.append(color_us)

label_list.append("Northern Rock bank run")
#date: day of run
event_time_list.append(nearest(datetime.date(2007, 9, 14))) 
arrow_color_list.append(color_they)

label_list.append("Lehman collapse")
#date: day when bankruptcy was declared
event_time_list.append(nearest(datetime.date(2008, 9, 15)))
arrow_color_list.append(color_they)

"""Get the CP label number"""
if number_labels:
    label_list = []
    for i in range(0, len(event_time_list)):
        label_list.append(str(i+1))

"""Obtain the plot for RLD"""
fig, ax = plt.subplots(1, figsize=(20, 5))  # 20,5
# fig.subplot.top : 0.5

_, myfig = EvT.plot_run_length_distr(buffer=50,
                                     show_MAP_CPs=False,
                                     mark_median=False,
                                     mark_max=True,
                                     upper_limit=285,
                                     print_colorbar=True,  # True
                                     colorbar_location="bottom",  # 'bottom',
                                     log_format=True,
                                     aspect_ratio='auto',
                                     C1=700, C2=1,
                                     time_range=np.linspace(start_comparison - start_test,
                                                            stop_comparison - start_test - 2,
                                                            stop_comparison - start_comparison - 2, dtype=int),
                                     start=(1998 + 1 / 365), stop=2009,
                                     all_dates=all_dates,
                                     event_time_list=event_time_list,
                                     label_list=label_list,
                                     space_to_colorbar=0.75,  # 0.6
                                     arrow_colors=arrow_color_list,
                                     xlab="Year",  # "Year"
                                     ax=ax, figure=fig,
                                     number_fontsize=27,  # 14
                                     xlab_fontsize=27,  # 14
                                     ylab_fontsize=27,  # 14
                                     xticks_fontsize=23,  # 12
                                     yticks_fontsize=23,  # 12
                                     arrow_length=30,  # 32
                                     arrow_thickness=4.0,
                                     arrows_setleft_indices=[3],
                                     arrows_setleft_by=[datetime.timedelta(days=35)],
                                     zero_distance=datetime.timedelta(days=0)
                                     # ylabel_coords = [10, 150]  #void
                                     )
    
"""Save the plot as pdf"""    
fig.savefig(os.path.join(results_directory, "30portfolio_picture_comparison.pdf"),
            format="pdf", dpi=800)
