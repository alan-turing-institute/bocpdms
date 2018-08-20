#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:01:35 2018

@author: jeremias knoblauch

Description: Process the bee data with the multiple-model BOCD algorithm
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from cp_probability_model import CpModel
from BVAR_NIG import BVARNIG
from detector import Detector
from Evaluation_tool import EvaluationTool

"""STEP 1: Regulate whether you want to analyze or plot"""
normalize = True #we normalize as Saatci et al. (2010) do 
                 #(see also the code of Turner (2012))

data_directory = os.path.join(os.getcwd(), "Data")
output_directory = os.path.join(os.getcwd(), "Output")

"""STEP 2: Plug in WHICH of the bee waggle sets you want to process"""
waggle_id = 1 #which of the 6 data sets we want. 
              #Turner (2012) & Saatci et al. (2010) analyze the first
coupling = "weak coupling"
file_name = os.path.join(data_directory, "bee_seq" + str(waggle_id) + ".csv")


"""STEP 3: We need to set up the potential neighbourhoods. It is reasonable 
            to work with three models: pure-AR, all-VAR,
            and AR for the angle but VAR for the coord  """

"""Idea is to further down multiply the restriction-parts by lag_length"""

"""STEP 3.1: PURE AR nbh """
AR_nbh_elem = [[[]],[[]],[[]]] 
AR_res_elem = [[0]]

"""STEP 3.2: PURE VAR nbh """
VAR_nbh_elem = [[[1],[2]], [[0],[2]],[[0],[1]]] 
VAR_res_elem = [[0,1]] 

"""STEP 3.3 Mixed nbh """
mixed_nbh_elem = [[[1]], [[0]], [[]]]
mixed_res_elem = [[0]]


"""STEP 4: Read in the data and get the #obs=T per TS
read in (as strings), skip first line (=header)"""
mylist = []
count = 0 
with open(file_name) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if count > 0:
            mylist += row
        count += 1
        
"""STEP 4.1: transform into floats and bring into numpy format"""
mylist2 = []
for entry in mylist:
    mylist2 += [float(entry)]
data_raw = np.array([mylist2]).reshape(int(len(mylist2)/4), 4)
true_CPs = data_raw[:,0]
true_CP_location = np.where(true_CPs>0)[0]+1 
true_CP_model_index = 0
data = data_raw[:,1:]
S1,S2,T = 3,1,np.size(true_CPs)
if normalize:
    data = (data - np.mean(data,axis=0))/(np.sqrt(np.var(data,axis=0)))


"""STEP 4.2: Read in the data and get the #obs=T per TS"""
#read in (as strings), skip first line (=header)
mylist = []
count = 0 
with open(file_name) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if count > 0:
            mylist += row
        count += 1


"""STEP 5.0 Set up the specifics of the model"""

"""STEP 5.1: Set a and b"""
prior_a = 1.0
prior_b = 1.0

"""STEP 5.2: Intercept grouping (here, intercept for every series indep.)"""
grouping = np.zeros((S1*S2, S1*S2))
for i in range(0, S1*S2):
    grouping[i,i]=1
grouping = grouping.reshape((S1*S2, S1,S2))

"""STEP 5.3: get some priors & other settings"""
prior_mean_scale = 0 
prior_var_scale = 0.01 
hyperpar_opt = "caron"
normalize = True
intensity = 50


"""STEP 5.4: Set up the intensity prior"""
cp_model = CpModel(intensity)

"""STEP 5.5: Get the span of reasonable lags given T and the intensity
              (Here, L = 0.85 * (T/log(T))^{1/4} for AR, and ^{1/6} for VAR"""
mult = 0.85
upper_AR = int(mult*pow(float(T)/np.log(T), 0.25) + 1)
lower_AR = 1
upper_VAR = int(mult*pow(float(T)/np.log(T), 1.0/6.0) + 1)
lower_VAR = 1



"""STEP 5.6: Set up the intensity prior"""
cp_model = CpModel(intensity)

"""STEP 5.7: build model universe"""
AR_models = []
for lag in range(lower_AR, upper_AR+1):
    AR_models += [BVARNIG(
                    prior_a = prior_a,prior_b = prior_b,
                    S1 = S1,S2 = S2,
                    prior_mean_scale = prior_mean_scale,
                    prior_var_scale = prior_var_scale,
                    intercept_grouping = grouping,
                    general_nbh_sequence = AR_nbh_elem,
                    general_nbh_restriction_sequence = AR_res_elem * lag,
                    general_nbh_coupling = "weak coupling",
                    hyperparameter_optimization = hyperpar_opt)]

VAR_models = []
for lag in range(lower_VAR, upper_VAR+1):
    VAR_models += [BVARNIG(
                    prior_a = prior_a,prior_b = prior_b,
                    S1 = S1,S2 = S2,
                    prior_mean_scale = prior_mean_scale,
                    prior_var_scale=prior_var_scale,
                    intercept_grouping = grouping,
                    general_nbh_sequence = VAR_nbh_elem,
                    general_nbh_restriction_sequence = VAR_res_elem * lag,
                    auto_prior_update=False,
                    general_nbh_coupling = "weak coupling",
                    hyperparameter_optimization = hyperpar_opt)]

mixed_models = []
for lag in range(lower_VAR, upper_VAR+1):
    mixed_models += [BVARNIG(
                    prior_a = prior_a,prior_b = prior_b,
                    S1 = S1,S2 = S2,
                    prior_mean_scale = prior_mean_scale,
                    prior_var_scale = prior_var_scale,
                    intercept_grouping = grouping,
                    general_nbh_sequence = mixed_nbh_elem,
                    general_nbh_restriction_sequence = mixed_res_elem * lag,
                    general_nbh_coupling = "weak coupling",
                    hyperparameter_optimization = hyperpar_opt)]

#put models together
model_universe = np.array(AR_models + VAR_models + mixed_models)

#uniform prior
model_prior = np.array([1/len(model_universe)]*len(model_universe))


"""STEP 6: Build and run detector"""
detector = Detector(data=data, model_universe=model_universe,
        model_prior = model_prior,
        cp_model = cp_model, S1 = S1, S2 = S2, T = T,
        store_rl=True, store_mrl=True,
        trim_type="keep_K", threshold = 200,
        notifications = 100,
        save_performance_indicators = True,
        training_period = 250)
detector.run()


"""STEP 7: give some results/pictures/summaries"""

"""Store results + real CPs into EvaluationTool obj"""
EvT = EvaluationTool()
EvT.add_true_CPs(true_CP_location=true_CP_location,
                 true_CP_model_index=true_CP_location,
                 true_CP_model_label=-1)
EvT.build_EvaluationTool_via_run_detector(detector)
EvT.store_results_to_HD(os.path.join(output_directory, "results_bee.txt"))
# EvT.build_EvaluationTool_via_results(os.path.join(output_directory, "results_bee.txt"))

print("convergence diagnostics for on-line hyperparameter opt:")
plt.plot(np.linspace(1,len(detector.model_universe[0].a_list),
                     len(detector.model_universe[0].a_list)),
         np.array(detector.model_universe[0].a_list))
plt.plot(np.linspace(1,len(detector.model_universe[0].b_list),
                     len(detector.model_universe[0].b_list)),
         np.array(detector.model_universe[0].b_list))

fig = EvT.plot_run_length_distr(
    time_range = np.linspace(1,
                             T-max(upper_VAR, upper_AR)-1,
                             T-max(upper_VAR, upper_AR)-1,
                             dtype=int),
    print_plt = True, 
    C1=10,C2=10, #scale the rld at plotting time to make it visibly discernible
    show_MAP_CPs = True, 
    show_real_CPs = False,
    mark_median = False, 
    log_format = True,
    CP_legend = False, 
    colorbar_location= 'bottom',
    space_to_colorbar = 0.52,
    buffer = 50)

plt.show()

"""Print results for paper"""

print("\n")
print("***** Predictive MSE + NLL from Table 1 in ICML 2018 paper *****")
print("MSE is %.5g with 95%% error of %.5g" %
      (np.sum(np.mean(detector.MSE, axis=0)), np.sum(1.96*stats.sem(detector.MSE))))
print("NLL is %.5g with 95%% error of %.5g" %
      (np.mean(detector.negative_log_likelihood), 1.96*stats.sem(detector.negative_log_likelihood)))
