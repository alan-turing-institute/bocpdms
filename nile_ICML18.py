#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:45:27 2018

@author: Jeremias Knoblauch, j.knoblauch@warwick.ac.uk

Description: Reads in the nile data and processes it as a demo
"""

"""System packages/modules"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import csv
import os
"""Modules of the BOCPDMS algorithm"""
from cp_probability_model import CpModel
from BVAR_NIG import BVARNIG
from detector import Detector
from Evaluation_tool import EvaluationTool


"""STEP 1: Set the current working directory."""
baseline_working_directory = os.getcwd()
nile_file = os.path.join(baseline_working_directory, "Data", "nile.txt")


"""STEP 2: Read in the nile data from nile.txt"""
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

"""STEP 3: Format the nile data so that it can be processed with a Detector
object and instantiations of ProbabilityModel subclasses"""
T = int(len(raw_data)/2)
S1, S2 = 1,1 #S1, S2 give you spatial dimensions, but Nile is univariate.
data = np.array(raw_data).reshape(T,2)
dates = data[:,0]
river_height = data[:,1]
mean, variance = np.mean(river_height), np.var(river_height)

"""STEP 4: Standardize in order to be able to compare with GP-approaches"""
river_height = (river_height-mean)/np.sqrt(variance)

"""STEP 5: Set up initial hyperparameters (will be optimized throughout 
the algorithm) and lag lengths"""
intensity = 100
cp_model = CpModel(intensity) #This is simply a constant hazard function
a, b = 1,1 #Inverse Gamma hyperparameters, will be optimized inside code
prior_mean_scale, prior_var_scale = 0, 0.075 #0.075 #0.075 #0.075 #0.0075 #Normal hyperparameters 
                                              #(not optimized yet)
                                              #seems sensitive! Fewer CPs for var = 1

"""STEP 6: Decide the minimum and maximum lag length of your AR models. In 
the paper, we suggest int(mult*pow(float(T)/np.log(T), 0.25) + 1) for mult = 1
as the maximum lag length, and 1 for the minimum lag length, but the algorithm
is relatively insensitive to choosing different pairs"""
upper_AR = 3
lower_AR = 1 

"""STEP 7: Set up the AR-models, run algorithm"""
AR_models = []
for lag in range(lower_AR, upper_AR+1):
    
    """Generate next model object"""
    AR_models += [BVARNIG(
                    prior_a = a,prior_b = b,
                    S1 = S1,S2 = S2,
                    prior_mean_scale = prior_mean_scale,
                    prior_var_scale = prior_var_scale,
                    intercept_grouping = None,
                    nbh_sequence = [0]*lag,
                    restriction_sequence = [0]*lag,
                    hyperparameter_optimization = "online")]
        
"""STEP 8: Put all model objects together, create model universe, model priors"""
model_universe = np.array(AR_models)
model_prior = np.array([1/len(model_universe)]*len(model_universe))

"""STEP 9: Build and run detector, i.e. the object responsible for executing 
BOCPDMS with multiple (previously specificed) models for the segments and a 
CP model specified by cp_model"""
detector = Detector(
        data=river_height, 
        model_universe=model_universe, 
        model_prior = model_prior,
        cp_model = cp_model, 
        S1 = S1, S2 = S2, T = T, 
        store_rl=True, store_mrl=True,
        trim_type="keep_K", threshold = 50,
        notifications = 50,
        save_performance_indicators = True,
        generalized_bayes_rld = "kullback_leibler", 
        alpha_param_learning = "individual", 
        alpha_param  = 0.01, 
        alpha_param_opt_t = 30,
        alpha_rld_learning=True,
        loss_der_rld_learning="squared_loss",
        loss_param_learning="squared_loss")
detector.run()

"""STEP 10: Store results into EvaluationTool object with plotting capability"""
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_run_detector(detector)
EvT.store_results_to_HD(os.path.join(baseline_working_directory, "Output", "results_nile.txt"))

"""STEP 11: Inspect convergence of the hyperparameters"""
for lag in range(0, upper_AR + 1 - lower_AR):
    plt.plot(np.linspace(1, len(detector.model_universe[lag].a_list),
                         len(detector.model_universe[lag].a_list)),
             np.array(detector.model_universe[lag].a_list))
    plt.plot(np.linspace(1, len(detector.model_universe[lag].b_list),
                         len(detector.model_universe[lag].b_list)),
             np.array(detector.model_universe[lag].b_list))
    plt.savefig(os.path.join(baseline_working_directory, "Output",
                             "ICML18_ExtraFigure_Nile_ab_lag" + str(detector.model_universe[lag].lag_length) + ".pdf"),
                format="pdf", dpi=800)
    plt.cla()

"""STEP 12: Also plot some performance indicators (will usually be printed 
to the console before the plots)"""
print("\nCPs are ", detector.CPs[-2])
print("\n***** Predictive MSE + NLL from Table 1 in ICML 2018 paper *****")
print("MSE is %.5g with 95%% error of %.5g" %
      (np.mean(detector.MSE), 1.96*scipy.stats.sem(detector.MSE)))
print("NLL is %.5g with 95%% error of %.5g" %
      (np.mean(detector.negative_log_likelihood), 1.96*scipy.stats.sem(detector.negative_log_likelihood)))
