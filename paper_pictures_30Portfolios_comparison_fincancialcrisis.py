#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 13:34:20 2018

@author: jeremiasknoblauch

Description: Use 30 Portfolios to zoom in to financial crisis and look at MAP 
             segmentation
"""


import pickle
import numpy as np
from Evaluation_tool import EvaluationTool
from matplotlib import pyplot as plt
import csv
import datetime
import matplotlib
import os

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
        #DEBUG: Unclear if this is needed
        if count > 0:
            myl += row
        count += 1
        if count % 2500 == 0:
            print(count)
dates = []
for e in myl:
    dates.append(int(e))
                              
result_path = baseline_working_directory
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_results(os.path.join(results_directory, results_file))

"""Using the dates, select your range and indices: select 
03/07/1975 -- 31/12/2008, i.e. find indices that correspond"""
start_test = dates.index(19740703)
start_algo = dates.index(19750703)
stop = dates.index(20081231)

"""time period for which we want RLD"""
start_comparison = dates.index(20070801)#dates.index(19980102)
stop_comparison = stop#stop

all_dates = []
for d in dates[start_comparison:stop-2]:
    s = str(d)
    all_dates.append(datetime.date(year=int(s[0:4]), month=int(s[4:6]), day=int(s[6:8])))

"""indices of that time period"""
start_range = start_comparison - start_test
stop_range = stop_comparison - start_test
time_range = np.linspace(start_range, stop_range, 
                         stop_range - start_range, dtype = int)

"""Create list of event times and labels"""
label_list, event_time_list= [],[]
afghanistan = False
crash2002  = True
iraq = True
london = True
number_labels = True
arrow_color_list = []
color_us = "orange"
color_they = "black"

#if comparison:
#    label_list.append("Asia Crisis")
#    event_time_list.append((1998 + (8)/12)) #date: start of last quarter 98
#    arrow_color_list.append(color_they)
#    label_list.append("DotCom bubble bursts")
#    event_time_list.append((2000 + (3/12))) #date: April, when inflation reports came in
#    arrow_color_list.append(color_they)
#    label_list.append("9/11")
#    event_time_list.append((2001 + (8 + 9/31)/12)) #date: airplane day
#    arrow_color_list.append(color_they)
#    if afghanistan:
#        label_list.append("Afghanistan war")
#        event_time_list.append(2001 + (9 + 7/31)/12) #Operation Enduring Freedom begins
#        arrow_color_list.append(color_us)
#    if crash2002:
#        label_list.append("2002 Crash")
#        event_time_list.append(2002 + (6 + 23/31)/12) #Crash of 2002, 23/7 lowest Dow Jones value
#        arrow_color_list.append(color_us)
#    if iraq:
#        label_list.append("Iraq war")
#        event_time_list.append(2003 + (2+20/31)/12) #Date of invasion
#        arrow_color_list.append(color_us)
#        
#    label_list.append("Bush cuts")
#    event_time_list.append((2003 + ((4+23/31)/12))) #date the regulation was passed
#    arrow_color_list.append(color_they)  
#    if london:
#        label_list.append("London Underground attacks, Hurricanes Katrina & Irma")
#        event_time_list.append(2005 + (6 + 7/31)/12) #date bombing 'terror premium'
#        arrow_color_list.append(color_us)
#    label_list.append("US Election")
#    event_time_list.append(2005 + (10 + 8/31)/12) #election day
#    arrow_color_list.append(color_they)
#    label_list.append("Northern Rock bank run")
#    event_time_list.append(2007 + (8 + 14/31)/12) #day of run
#    arrow_color_list.append(color_they)
#    label_list.append("Lehman collapse")
#    event_time_list.append(2008 + (8 + 15/31)/12) #day of bankruptcy
#    arrow_color_list.append(color_they)

#taken from https://www.theguardian.com/business/2012/aug/07/credit-crunch-boom-bust-timeline
#,          http://www.cityam.com/269810/global-financial-crisis-10-years-timeline-global-events
#           http://www.telegraph.co.uk/finance/financialcrisis/8592990/Timeline-of-world-financial-crisis.html
#           https://lauder.wharton.upenn.edu/wp-content/uploads/2015/06/Chronology_Economic_Financial_Crisis.pdf


#    label_list.append("UBS reports 3.4bn$ losses")
#    event_time_list.append(2007 + (9)/12) #date of announcement
#    arrow_color_list.append("blue")
#    label_list.append("UBS announces 4.3bn$ losses")
#    event_time_list.append(2007 + (9)/12) #date of trend reversal
#    arrow_color_list.append("blue")

"""helper function getting closest date in all_dates"""
def nearest(pivot):
    return min(all_dates, key=lambda x: abs(x - pivot))

label_list.append("BNP Paribas freezes funds")
event_time_list.append(nearest(datetime.date(2007,8,9))) #day of freeze
arrow_color_list.append(color_us)
label_list.append("Fed cuts lending rate by 0.5% points")
event_time_list.append(nearest(datetime.date(2007,8,17))) #day of freeze
arrow_color_list.append(color_us)
label_list.append("IKB announces 1bn$ losses related to subprime market")
event_time_list.append(nearest(datetime.date(2007,9,3))) #day of freeze
arrow_color_list.append(color_us)
label_list.append("Northern Rock bank run")
event_time_list.append(nearest(datetime.date(2007,9,14))) #day of run
arrow_color_list.append(color_they)
label_list.append("Fed cuts interest rate by 0.5% points, BoE injects 10bn£ into market")
event_time_list.append(nearest(datetime.date(2007,9,19))) #day of run
arrow_color_list.append(color_us)
#label_list.append("UBS announces 3.4bn$, Citigroup 3.1bn$ loss related to subprime mortgage markets")
#event_time_list.append(nearest(datetime.date(2007, 10, 1))) #date of CEO resigning
#arrow_color_list.append(color_us)
#label_list.append("Meryll Lynch unveils 7.9bn$ exposure")
#event_time_list.append(nearest(datetime.date(2007, 10, 30)))#date of CEO resigning
#arrow_color_list.append(color_us)
#label_list.append("S&P downgrades bond insurers")
#event_time_list.append(nearest(datetime.date(2007, 12, 19))) #date of CEO resigning
#arrow_color_list.append(color_us)
#label_list.append("Credit crunch hits US and EU")
#event_time_list.append(nearest(datetime.date(2008,1, 9)))
#arrow_color_list.append(color_us)
label_list.append("Bush announces plans to help >1 million homeowners facing foreclosure")
event_time_list.append(nearest(datetime.date(2007,12,6)))
arrow_color_list.append(color_us)
label_list.append("Fed, ECB, BoE offer loans to banks")
event_time_list.append(nearest(datetime.date(2007,12,13)))
arrow_color_list.append(color_us)
label_list.append("Fed slashes federal funds rate to 3.5%")
event_time_list.append(nearest(datetime.date(2008, 1, 22)))
arrow_color_list.append(color_us)
#label_list.append("Bond insurer MBIA announces 2.3bn$ loss in the last 3 months")
#event_time_list.append(nearest(datetime.date(2008, 1, 31)))
#arrow_color_list.append(color_us)
label_list.append("G7 estimates subprime mortgage related losses at 400bn$ worldwide")
event_time_list.append(nearest(datetime.date(2008, 2, 10))) #day of privatization
arrow_color_list.append(color_us)
#label_list.append("Northern Rock nationalized")
#event_time_list.append(nearest(datetime.date(2008, 2, 17))) #day of privatization
#arrow_color_list.append(color_us)
#label_list.append("Biggest Fed intervention yet makes 200bn$ available in liquidity")
#event_time_list.append(nearest(datetime.date(2008,3,7))) #day of privatization
#arrow_color_list.append(color_us)
label_list.append("JP Morgan buys Bear Stearns, Fed intervention makes 200bn$ available in liquidity")
event_time_list.append(nearest(datetime.date(2008,3,16))) #day of purchase
arrow_color_list.append(color_us)
label_list.append("IMF estimates worldwide losses at >1 trillion $")
event_time_list.append(nearest(datetime.date(2008,4,8))) #day of purchase
arrow_color_list.append(color_us)
#label_list.append("RBS largest rights issue in UK history")
#event_time_list.append(nearest(datetime.date(2008, 4, 22))) #day of purchase
#arrow_color_list.append(color_us)
#label_list.append("UBS rights issue to recover 37bn$")
#event_time_list.append(nearest(datetime.date(2008,5,22))) #day of purchase
#arrow_color_list.append(color_us)
#label_list.append("FBI arrests 406 mortgage fraudsters")
#event_time_list.append(nearest(datetime.date(2008, 6, 19))) #day of purchase
#arrow_color_list.append(color_us)
#label_list.append("First Fannie Mae & Freddie Mac support")
#event_time_list.append(nearest(datetime.date(2008, 7, 14))) #day of bailout
#arrow_color_list.append(color_us)
label_list.append("HBOS' rights issue fails")
event_time_list.append(nearest(datetime.date(2008, 7, 21))) #day of bailout
arrow_color_list.append(color_us)
#label_list.append("HSBC announces substantial fall in profits")
#event_time_list.append(nearest(datetime.date(2008, 8, 4))) #day of bailout
#arrow_color_list.append(color_us)
label_list.append("ECB efforts for liquidity pumping 200bn€ into the market")
event_time_list.append(nearest(datetime.date(2008,8,9))) #day of bailout
arrow_color_list.append(color_us)
label_list.append("Fannie Mae & Freddie Mac bailout")
event_time_list.append(nearest(datetime.date(2008,9,7))) #day of bailout
arrow_color_list.append(color_us)
label_list.append("Lehman collapse")
event_time_list.append(nearest(datetime.date(2008,9,15))) #day of bankruptcy
arrow_color_list.append(color_they)
#    label_list.append("Lloyds saves HBOS")
#    event_time_list.append(2008 + (8 + 17/31)/12) #day of purchase
#    arrow_color_list.append(color_us)
#    label_list.append("TARP programme")
#    event_time_list.append(2008 + (9 + 3/31)/12) #day of ratification
#    arrow_color_list.append("blue")
#    label_list.append("FED, ECB, BoE cut interest rates")
#    event_time_list.append(2008 + (9 + 8/26)/12) #day of cut
#    arrow_color_list.append(color_us)
#    label_list.append("UK bails out Lloyds, RBS, HBOS")
#    event_time_list.append(2008 + (9 + 13/26)/12) #day of bail
#    arrow_color_list.append(color_us)
#label_list.append("Ireland officially enters recession")
#event_time_list.append(nearest(datetime.date(2008,9,25))) #day of ratification
#arrow_color_list.append(color_us)
label_list.append("Russia announces 500bn roubles to fight the crisis, BoE injects 10bn£ into market")
event_time_list.append(nearest(datetime.date(2008,9,19))) #day of ratification
arrow_color_list.append(color_us)
label_list.append("BENELUX bail out Fortis with 11.8bn$")
event_time_list.append(nearest(datetime.date(2008,9,28))) #day of ratification
arrow_color_list.append(color_us)
label_list.append("UK announces 500bn£ bank rescue package")
event_time_list.append(nearest(datetime.date(2008,10,8))) #day of bail
arrow_color_list.append(color_us)
#label_list.append("UK bails out Lloyds, RBS, HBOS; NL bails out ING")
#event_time_list.append(nearest(datetime.date(2008,10,19))) #day of bail
#arrow_color_list.append(color_us)
label_list.append("IMF loan over 16.4bn$ to Ukraine, BoE and ECB cut interest rates")
event_time_list.append(nearest(datetime.date(2008, 11, 6))) #day of bail
arrow_color_list.append(color_us)
label_list.append("G20 pledge for fiscal stimuli")
event_time_list.append(nearest(datetime.date(2008, 11, 17))) #
arrow_color_list.append(color_us)
#label_list.append("Fed starts Quantitative Easing")
#event_time_list.append(nearest(datetime.date(2008, 11, 25))) #
#arrow_color_list.append(color_us)
#    label_list.append("Ireland enters recession")
#    event_time_list.append(2008 + (11 + 11/26)/12) #
#    arrow_color_list.append(color_us)
#label_list.append("Citigroup bailout for total of 351bn$")
#event_time_list.append(nearest(datetime.date(2008, 11, 23))) #
#arrow_color_list.append(color_us)
#label_list.append("BoE cuts interest to lowest rate since '51," + 
#        "ECB lowers interest rates, France announces 26bn EUR stimulus package" + 
#        "Germany & Sweden announce stimulus package over 38bn and 8bn EUR")
#event_time_list.append(nearest(datetime.date(2008, 12, 4))) #
#arrow_color_list.append(color_us)
#label_list.append("Germany & Sweden announce stimulus package over 38bn EUR")
#event_time_list.append(nearest(datetime.date(2008, 12, 6))) #
#arrow_color_list.append(color_us)
label_list.append("Madoff's 50bn$ Ponzi scheme revealed, 14bn$ car rescue " + 
                  "package in US does not pass senate, South Korean central" +
                  " bank lower interest to record low")
event_time_list.append(nearest(datetime.date(2008, 12, 11)))
arrow_color_list.append(color_us)
label_list.append("Fed cuts interest rate to 0.25%, " + 
                  "Japanese central bank cuts interest rate to 0.3%")
event_time_list.append(nearest(datetime.date(2008, 12, 16)))
arrow_color_list.append(color_us)
#    label_list.append("Japanese central bank cuts interest rate to 0.3%")
#    event_time_list.append(2008 + (11 + 19/26)/12) #
#    arrow_color_list.append(color_us)


#Idea: Maybe research the 2008/2009 period and match MAP CPs with individual events

if number_labels:
    label_list = []
    for i in range(0, len(event_time_list)):
        label_list.append(str(i+1))
    #arrow_color_list = ["black"]*len(event_time_list)


"""get MAP CPs in range"""
segmentation = np.array(EvT.results[EvT.names.index("MAP CPs")][-2])
segmentation_selection = segmentation[np.logical_and(
        segmentation[:,0] >= start_range, segmentation[:,0] <= stop_range),:]
models = np.union1d([e[1] for e in segmentation_selection],
                    [e[1] for e in segmentation_selection]) 


"""Obtain the plot for RLD"""
#start, stop = (2007 + 7/12), 2009
#start, stop = datetime.date(2007, 8, 1), datetime.date(2008, 12, 31)
custom_colors = ["blue"] * 5 #["green", "darkviolet", "orange", "purple", "turquoise"]
#paper: figsize = (20,5) poster: figsize=(12,3)
fig, ax = plt.subplots(1, figsize = (20,5))
#ax = ax_[0]
_, myfig=EvT.plot_run_length_distr(buffer=0, show_MAP_CPs = True, mark_median = False, 
    mark_max = True, upper_limit = 115, enforce_upper_limit=True, print_colorbar = True, 
    colorbar_location= 'top',log_format = True, aspect_ratio = 'auto', 
    C1=700,C2=1, 
    time_range = np.linspace(start_comparison - start_test,
                             stop_comparison - start_test - 2, 
                             stop_comparison-start_comparison-2, dtype=int), 
    #start=start, stop = stop, 
    all_dates = all_dates,
    event_time_list=event_time_list, 
    label_list=label_list, space_to_colorbar = 0.9,
    custom_colors = custom_colors, #["blue"]*len(event_time_list),
    custom_linestyles = ["solid"]*len(event_time_list),
    custom_linewidth = 3,
    arrow_colors= arrow_color_list,
    number_fontsize = 18,
    arrow_length = 13, #paper: arrow-length = 13 poster arrow-length = 27
    arrow_thickness = 4.0,
    arrows_setleft_indices = [3,13,14,15, 20, 21],
    #paper: arrows_setleft_indices = [3,13,14,15, 20, 21],
    #poster: arrows_setleft_indices = [3,13,14,15,16,17,18, 20, 21],
    #poster: arrows_setleft_by = [datetime.timedelta(days = 3),
    #                    datetime.timedelta(days=15),datetime.timedelta(days=8), 
    #                     -datetime.timedelta(days = 1),
    #                     -datetime.timedelta(days=6),
    #                     -datetime.timedelta(days=9),
    #                     datetime.timedelta(days = 3),
    #                     datetime.timedelta(days=9), datetime.timedelta(days=1)],
    #paper: 
    arrows_setleft_by = [datetime.timedelta(days = 3),
                        datetime.timedelta(days=7),datetime.timedelta(days=4), 
                         -datetime.timedelta(days = 1),
                         datetime.timedelta(days=9), datetime.timedelta(days=1)],
    zero_distance = datetime.timedelta(days=0),
    xlab_fontsize =14,
    ylab_fontsize = 14, 
    xticks_fontsize = 12,
    yticks_fontsize = 12,
    ax = ax, figure = fig,
    xlab = "Year-Month")

fig.savefig(os.path.join(results_directory, "30portfolio_picture_financial_crisis.pdf"),
            format="pdf", dpi=800)

#start_day = datetime.date(2007, 8, 1)
#stad = matplotlib.dates.date2num( start_day )
#stop_day = datetime.date(2008, 12, 31)
#stod = matplotlib.dates.date2num( stop_day )
#for mod, ind in zip(models, range(0, len(models))):
#    EvT.plot_model_posterior(indices = [mod], time_range = np.linspace(
#            start_range, stop_range-2, stop_range-start_range-2, dtype=int),
#        log_format = True, show_MAP_CPs = False, 
#        custom_colors = [custom_colors[ind]],
#        start_axis = stad, stop_axis = stod)
