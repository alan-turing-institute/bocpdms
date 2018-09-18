#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:01:35 2018

@author: jeremias knoblauch

Description: Process the bee data with the multiple-model BOCD algorithm
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

from cp_probability_model import CpModel
from BVAR_NIG import BVARNIG
from detector import Detector
from Evaluation_tool import EvaluationTool


def load_bee_data(data_dir, w_id, normalize_data):
    """Read in the bee waggle dataset and return the data (x and y coordinates and head angle), the numbers of spatial
    dimensions and timesteps, and the true changepoints.
    Full dataset: http://mldata.org/repository/data/viewslug/bee/
    Original paper: Learning and Inferring Motion Patterns using Parametric Segmental Switching Linear Dynamic Systems,
    Sang Min Oh, James M. Rehg, Tucker Balch, Frank Dellaert. International Journal of Computer Vision (IJCV)
    Special Issue on Learning for Vision, May 2008. Vol.77(1-3). Pages 103-124."""

    # Path to the data
    file_name = os.path.join(data_dir, "bee_seq" + str(w_id) + ".csv")

    # Read in the data as strings, skipping the header (first row)
    mylist = []
    count = 0
    with open(file_name) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if count > 0:
                mylist += row
            count += 1

    # Transform into floats and bring into numpy format
    mylist2 = []
    for entry in mylist:
        mylist2 += [float(entry)]
    data_raw = np.array([mylist2]).reshape(int(len(mylist2) / 4), 4)

    # True CPs are given in the first column
    true_CPs = data_raw[:, 0]
    true_CP_location = np.where(true_CPs > 0)[0] + 1
    true_CP_model_index = 0

    # x, y and head angle are given in the remaining columns
    data = data_raw[:, 1:]
    S1, S2, T = 3, 1, np.size(true_CPs)

    # We may want to normalize the data
    if normalize_data:
        data = (data - np.mean(data, axis=0)) / (np.sqrt(np.var(data, axis=0)))

    return data, S1, S2, T, true_CP_location, true_CP_model_index


if __name__ == "__main__":

    # Set the locations of the datasets and output
    data_directory = os.path.join(os.getcwd(), "Data")
    output_directory = os.path.join(os.getcwd(), "Output")

    """STEP 1: Read in and process command line arguments"""

    #  Set up the parser
    parser = argparse.ArgumentParser(
        description="Options for applying the BOCPDMS algorithm to the bee waggle dance dataset.")
    parser.add_argument("-f", "--figures", type=bool, default=False, help="Show figures")
    parser.add_argument("-a", "--prior_a", type=float, default=1.0, help="Initial value of a")
    parser.add_argument("-b", "--prior_b", type=float, default=1.0, help="Initial value of b")
    parser.add_argument("-ms", "--prior_mean_scale", type=float, default=0.0,
                        help="Mean scale used to calculate beta_0")
    parser.add_argument("-vs", "--prior_var_scale", type=float, default=0.01,
                        help="Variance scale used to calculate V_0")
    parser.add_argument("-i", "--intensity", type=float, default=50, help="Intensity")
    parser.add_argument("-lAR", "--lower_AR", type=int, default=1, help="Lower lag length for AR models")
    parser.add_argument("-uAR", "--upper_AR", type=int, default=None, help="Upper lag length for AR models")
    parser.add_argument("-lVAR", "--lower_VAR", type=int, default=1, help="Lower lag length for VAR models")
    parser.add_argument("-uVAR", "--upper_VAR", type=int, default=None, help="Upper lag length for VAR models")

    # Get settings and parameter values from the command line arguments
    args = parser.parse_args()
    show_figures = args.figures
    prior_a = args.prior_a
    prior_b = args.prior_b
    prior_mean_scale = args.prior_mean_scale
    prior_var_scale = args.prior_var_scale
    intensity = args.intensity
    lower_AR = args.lower_AR
    upper_AR = args.upper_AR    # If not provided, will be set based on length of time-series after loading data
    lower_VAR = args.lower_VAR
    upper_VAR = args.upper_VAR  # As for upper_AR, will be set later if not provided

    # Set the remaining options and parameter values
    normalize = True            # Normalize as Saatci et al. (2010) do (see also the code of Turner (2012))
    hyperpar_opt = "caron"
    coupling = "weak coupling"

    """STEP 2: Load the data"""

    waggle_id = 1               # Turner (2012) & Saatci et al. (2010) analyze the first of the six datasets
    data, S1, S2, T, true_CP_location, true_CP_model_index = load_bee_data(data_directory, waggle_id, normalize)

    """STEP 3: We need to set up the potential neighbourhoods. It is reasonable to work with three models:
    pure-AR, all-VAR, and AR for the angle but VAR for the coord. The idea is to further down multiply the
    restriction-parts by lag_length."""

    # Pure AR nbh
    AR_nbh_elem = [[[]], [[]], [[]]]
    AR_res_elem = [[0]]

    # Pure VAR nbh
    VAR_nbh_elem = [[[1], [2]], [[0], [2]], [[0], [1]]]
    VAR_res_elem = [[0, 1]]

    # Mixed nbh
    mixed_nbh_elem = [[[1]], [[0]], [[]]]
    mixed_res_elem = [[0]]

    """STEP 4: Set up the specifics of the model"""

    # Intercept grouping (here, intercept for every series indep.)
    grouping = np.zeros((S1 * S2, S1 * S2))
    for i in range(0, S1 * S2):
        grouping[i, i] = 1
    grouping = grouping.reshape((S1 * S2, S1, S2))

    # Set up the intensity prior
    cp_model = CpModel(intensity)

    # If upper_AR and upper_VAR are not already set from the command line arguments, get the span of reasonable lags
    # given T (here, L = 0.85 * (T/log(T))^{1/4} for AR, and ^{1/6} for VAR)
    mult = 0.85
    if upper_AR is None:
        upper_AR = int(mult * pow(float(T) / np.log(T), 0.25) + 1)
    if upper_VAR is None:
        upper_VAR = int(mult * pow(float(T) / np.log(T), 1.0 / 6.0) + 1)

    # Create all AR models
    AR_models = []
    for lag in range(lower_AR, upper_AR + 1):
        AR_models += [BVARNIG(
            prior_a=prior_a, prior_b=prior_b,
            S1=S1, S2=S2,
            prior_mean_scale=prior_mean_scale,
            prior_var_scale=prior_var_scale,
            intercept_grouping=grouping,
            general_nbh_sequence=AR_nbh_elem,
            general_nbh_restriction_sequence=AR_res_elem * lag,
            general_nbh_coupling=coupling,
            hyperparameter_optimization=hyperpar_opt)]

    # Create all VAR models
    VAR_models = []
    for lag in range(lower_VAR, upper_VAR + 1):
        VAR_models += [BVARNIG(
            prior_a=prior_a, prior_b=prior_b,
            S1=S1, S2=S2,
            prior_mean_scale=prior_mean_scale,
            prior_var_scale=prior_var_scale,
            intercept_grouping=grouping,
            general_nbh_sequence=VAR_nbh_elem,
            general_nbh_restriction_sequence=VAR_res_elem * lag,
            auto_prior_update=False,
            general_nbh_coupling=coupling,
            hyperparameter_optimization=hyperpar_opt)]

    # Create all mixed models
    mixed_models = []
    for lag in range(lower_VAR, upper_VAR + 1):
        mixed_models += [BVARNIG(
            prior_a=prior_a, prior_b=prior_b,
            S1=S1, S2=S2,
            prior_mean_scale=prior_mean_scale,
            prior_var_scale=prior_var_scale,
            intercept_grouping=grouping,
            general_nbh_sequence=mixed_nbh_elem,
            general_nbh_restriction_sequence=mixed_res_elem * lag,
            general_nbh_coupling=coupling,
            hyperparameter_optimization=hyperpar_opt)]

    # Put models together into the model universe
    model_universe = np.array(AR_models + VAR_models + mixed_models)

    # Use a uniform prior over these models
    model_prior = np.array([1 / len(model_universe)] * len(model_universe))

    """STEP 5: Build and run detector"""

    detector = Detector(data=data,
                        model_universe=model_universe,
                        model_prior=model_prior,
                        cp_model=cp_model,
                        S1=S1, S2=S2, T=T,
                        store_rl=True, store_mrl=True,
                        trim_type="keep_K", threshold=200,
                        notifications=100,
                        save_performance_indicators=True,
                        training_period=250)
    detector.run()

    """STEP 6: Give some results/pictures/summaries"""

    # Store results + real CPs into EvaluationTool object
    EvT = EvaluationTool()
    EvT.add_true_CPs(true_CP_location=true_CP_location,
                     true_CP_model_index=true_CP_location,
                     true_CP_model_label=-1)
    EvT.build_EvaluationTool_via_run_detector(detector)
    EvT.store_results_to_HD(os.path.join(output_directory, "results_bee.txt"))

    if show_figures:

        # Plot the values of a and b in the first model
        plt.plot(np.linspace(1, len(detector.model_universe[0].a_list), len(detector.model_universe[0].a_list)),
                 np.array(detector.model_universe[0].a_list))
        plt.plot(np.linspace(1, len(detector.model_universe[0].b_list), len(detector.model_universe[0].b_list)),
                 np.array(detector.model_universe[0].b_list))

        # Plot the run-length distribution
        fig = EvT.plot_run_length_distr(time_range=np.linspace(1, T - max(upper_VAR, upper_AR) - 1,
                                                               T - max(upper_VAR, upper_AR) - 1, dtype=int),
                                        print_plt=True,
                                        C1=10, C2=10,  # Scale the rld at plotting time to make it visibly discernible
                                        show_MAP_CPs=True,
                                        show_real_CPs=False,
                                        mark_median=False,
                                        log_format=True,
                                        CP_legend=False,
                                        colorbar_location='bottom',
                                        space_to_colorbar=0.52,
                                        buffer=50)

        # Display the two plots
        plt.show()

    # Print results for the table in the ICML paper
    print("\n")
    print("***** Predictive MSE + NLL from Table 1 in ICML 2018 paper *****")
    print("MSE is %.5g with 95%% error of %.5g" %
          (np.sum(np.mean(detector.MSE, axis=0)), np.sum(1.96 * stats.sem(detector.MSE))))
    print("NLL is %.5g with 95%% error of %.5g" %
          (np.mean(detector.negative_log_likelihood), 1.96 * stats.sem(detector.negative_log_likelihood)))

    # And finally, print out the settings used
    print("\n")
    print("***** Parameter values and other options *****")
    print("prior_a:", prior_a)
    print("prior_b:", prior_b)
    print("prior_mean_scale:", prior_mean_scale)
    print("prior_var_scale:", prior_var_scale)
    print("intensity:", intensity)
    print("lower_AR:", lower_AR)
    print("upper_AR:", upper_AR)
    print("lower_VAR:", lower_VAR)
    print("upper_VAR:", upper_VAR)
    print("normalize:", normalize)
    print("coupling:", coupling)
    print("hyperpar_opt:", hyperpar_opt)
