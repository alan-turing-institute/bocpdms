#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 18:47:25 2018

@author: jeremias

Description: Extract pictures
"""
import pickle
import numpy as np
from Evaluation_tool import EvaluationTool
from matplotlib import pyplot as plt

#only needed if you want to generate demo data yourself
from cp_probability_model import CpModel 
from detector import Detector 
import synthetic_simulations_prepare_data    
import synthetic_simulations_prepare_models


#ensure that we have type 1 fonts (for ICML publishing guiedlines)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


baseline_working_directory = ("//Users//jeremiasknoblauch//Documents////OxWaSP//BOCPDMS//Code//" + 
    "SpatialBOCD//Paper//demo")


"""####################################################"""
"""DATA SET 1: Synthetic data set: VAR_infinity, fct. 3"""
"""####################################################"""

process_new = False #only if you want to re-run everything

"""get the right size"""
T, S1, S2 = 500, 5, 5

"""only do this if you want to generate data yourself"""
if process_new:
    """Call the relevant builder fct and add to all simulators"""
    full_fct_name = "generate_VAR_infty_3"
    method_to_call = getattr(synthetic_simulations_prepare_data, 
                         full_fct_name)
    sim = method_to_call(S1=S1, S2=S2, T=T)
    data = sim.generate_data() 
else:
    """read in data"""
    data_path = baseline_working_directory
    data_file = open(data_path + "//data_demo.txt", 'rb')
    data = pickle.load(data_file)
    data_file.close()

"""normalize"""
data = ((data - np.mean(data, axis=0))/
            np.sqrt(np.var(data,axis=0)))

"""shorten to interior """
data = data[:,1:S1-1, 1:S2-1]
T = data.shape[0]
S1 = S1 - 2
S2 = S2 - 2

if process_new:
    """priors"""
    intensity, a, b, b_default = 1500.0, 15.0, 0.05, 0.05
    mult = 1.0
    lower_AR, upper_AR = 1.0, int(mult*pow(float(T)/np.log(T), 0.25) + 1.0)
    upper_VAR, lower_VAR = 1.0, int(mult*pow(float(T)/np.log(T), 1.0/6.0) + 1)
    cp_model = CpModel(intensity)
    hyperpar_opt = "online"
    prior_var_beta_scale = 10.0
    """models"""
    method_to_call = getattr(
         synthetic_simulations_prepare_models, 
             "generate_VAR0_nbh_models")
    VAR0_list = method_to_call(S1,S2,a,b_default, 
        prior_var_beta_scale, lower_AR, upper_AR,
        hyperpar_opt) 
    method_to_call = getattr(
         synthetic_simulations_prepare_models, 
             "generate_VAR4_nbh_models")      
    VAR4_list = method_to_call(S1,S2,a,b_default, 
        prior_var_beta_scale, lower_VAR, upper_VAR,
        hyperpar_opt) 
    method_to_call = getattr(
         synthetic_simulations_prepare_models, 
             "generate_VAR8_nbh_models")
    VAR8_list = method_to_call(S1,S2,a,b_default, 
        prior_var_beta_scale, lower_VAR, upper_VAR,
        hyperpar_opt)   
    model_universe = VAR0_list + VAR4_list + VAR8_list
    """detector"""
    model_universe = np.array(model_universe)
    model_prior = np.array(
             [1.0/int(len(model_universe))] * 
             int(len(model_universe)))
    detector = Detector(
        data, model_universe, model_prior, cp_model, 
        S1, S2, T, exo_data=None, num_exo_vars=None, 
        threshold=200,
        store_rl=True, store_mrl=True)#,
    detector.run()
                                
    """build evaluation tool"""                           
    EvT = EvaluationTool()
    EvT.build_EvaluationTool_via_run_detector(detector)
else:
    """Directly read results"""
    result_path = baseline_working_directory
    EvT = EvaluationTool()
    EvT.build_EvaluationTool_via_results(result_path + "//" + "results_demo.txt") 


"""Plot in panels: Raw data (offsetting the mean), 
                         CPs + RLD
                         1-step-ahead-prediction + variance
                         model posterior"""
custom_colors_models =  ['green', 'purple', 'orange', 'blue', 'darkgray']
custom_colors_series = ['black']*4
custom_linestyles =['solid']*5
offsets = np.linspace(1,S1*S2, S1*S2).reshape(S1, S2)
data_original = data.copy()
data = data + offsets*2.5
data = data.reshape(T, S1*S2)

""""Get some quantities needed for rest"""
segmentation = EvT.results[EvT.names.index("MAP CPs")][-2]
model_labels = EvT.results[EvT.names.index("model labels")]
num_models = len(np.union1d(model_labels, model_labels))
relevant_models = np.union1d([seg[1] for seg in segmentation],[seg[1] for seg in segmentation])
#exclude 7, as it just obscures the view
relevant_models = [1,2,5]
all_models = np.linspace(0, num_models-1, num_models, dtype=int)

"""detemine if you want separate model posterior plots"""
one_model_posterior_plot = True

if one_model_posterior_plot:
    #paper: height_ratio = [6,6,5,24]  poster: height_ratio = [5,5,5,12]
    height_ratio = [6,6,5,24] 
    num_subplots = 4
else:
    height_ratio = [8,8, 4,4,4, 18]
    num_subplots = 6

"""set ylabel position"""
#paper: ylabel_coords = [-0.09, 0.5] poster: ylabel_coords = [-0.045, 0.5]
ylabel_coords = [-0.09, 0.5]
yticks_fontsize = 10

#paper: figsize argument left out poster: figsize = (12,5)
fig, ax_array = plt.subplots(num_subplots, sharex = True, 
                             gridspec_kw = {'height_ratios':height_ratio})#,
                             #figsize = (12,5))
plt.subplots_adjust(hspace = .185, left = None, bottom = None, right = None, top = None)

"""plot raw data"""
fig_1 = EvT.plot_raw_TS(data, indices=[0,2,4], 
                        custom_colors_series = custom_colors_series,
                        ax = ax_array[0],xlab=None,
                        ylabel_coords = ylabel_coords) #ax_array[0])

"""Plot + save pics (4) prediction + error"""
fig_2 = EvT.plot_prediction_error(show_var= True, data = data_original, 
                                  indices = [0], 
                                  custom_colors = ['black', 'darkgray'],
                                  ax=ax_array[1], 
                                  time_range = np.linspace(1,T-2,T-2,dtype=int),
                                  aspect_ratio = 'auto', xlab = None, ylab = "PE",
                                  ylabel_coords = ylabel_coords)

"""Plot + save pictures: (3) model posteriors"""
"""Plot + save pictures: (2) Raw TS with MAP CPs"""
#backconverter = tick/C2-C1 (backconv+C1)*C2 = tick
C1, C2 = 100, 1
#raw_ticks = [pow(10, -18), pow(10, -15), pow(10, -13), pow(10, -11)]
colorbar_ticks = None #[C2*(r+C1) for r in raw_ticks]
fig_4 = EvT.plot_run_length_distr(mark_max=True,upper_limit=250,
                                  aspect_ratio='auto', C1=C1, C2=C2,
                                  time_range = np.linspace(1,T-2,T-2, dtype=int), 
                                  CP_legend = True, CP_legend_fontsize = 8,
                                  CP_custom_legend_labels = ["AR(2)", "AR(3)", "VAR4(2)"],
                                  additional_legend_labels = ["VAR8(1)"],
                                  additional_legend_colors = ["blue"],
                                  CP_exclude_indices = [7], #just obscures view
                                  custom_colors = custom_colors_models,
                                  custom_linestyles =custom_linestyles, 
                                  ax = ax_array[-1], figure = fig, space_to_colorbar = 0.275,
                                  orientation = 'horizontal',
                                  xlab=None, colorbar_location = "bottom",
                                  ylabel_coords = ylabel_coords,
                                  colorbar_ticks_num = 1)

if one_model_posterior_plot:
#custom_colors = ['purple', 'orange', 'red', 'green']
    fig_3 = EvT.plot_model_posterior(indices= [1,2,5,6],#relevant_models.append(30), 
                                     plot_type = "trace",
                                     y_axis_labels = ["mod1", "mod2", "mod3"],
                                     log_format=False, aspect = 'auto',
                                     show_MAP_CPs = False,
                                     custom_colors = custom_colors_models, 
                                     ax = ax_array[2], xlab = None,
                                     ylabel_coords = ylabel_coords)
    
#    if one_model_posterior_plot:
##custom_colors = ['purple', 'orange', 'red', 'green']
#    fig_3 = EvT.plot_model_posterior(indices=relevant_models, 
#                                     plot_type = "MAP",
#                                     y_axis_labels = ["mod1", "mod2", "mod3"],
#                                     log_format=False, aspect = 'auto',
#                                     show_MAP_CPs = False,
#                                     custom_colors = custom_colors_models, 
#                                     ax = None, xlab = None)
else: 
    for ind, mod in zip(range(0, len(relevant_models)), relevant_models):
#        if ind == 2:
#            xlab = "Time"
#        else:
        xlab = None
        if ind == 1:
            ylab = "P(m|y)"
        else:
            ylab = None
        fig_3 = EvT.plot_model_posterior(indices=[relevant_models[ind]], 
                 log_format=False, aspect = 'auto',
                 show_MAP_CPs = False,
                 custom_colors = [custom_colors_models[ind]], 
                 ax = ax_array[2+ind], xlab = xlab, ylab = ylab,
                 ylabel_coords = ylabel_coords) 
    

#plt.show()

fig.savefig(baseline_working_directory + "//demo_picture.pdf", 
            format = "pdf", orientation = "portrait",
    dpi = 800)
