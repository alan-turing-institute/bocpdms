#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 13:00:45 2018

@author: jeremiasknoblauch

Description: Reads in and processes Whistler data
             Note: for 0-periods, get NIG model with high confidence of 0s
"""

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

from BVAR_NIG import BVARNIG
from detector import Detector
from Evaluation_tool import EvaluationTool
from cp_probability_model import CpModel


def load_snowfall_data(data_dir, normalise_data, log_transform_data):
    """ The original data can be found at http://mldata.org/repository/data/viewslug/whistler-daily-snowfall/
    The csv files listed here have been preprocessed to remove unused columns of data.
    whistler_dates contains the dates in y-m-d format (not used here), and  whistler_data contains the total
    snow in cm."""

    whistler_data = os.path.join(data_dir, "whistler_data.csv")

    # Read in the data from the csv file
    raw_data = []
    count = 0
    with open(whistler_data) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if count > 0:
                raw_data += row
            count += 1

    raw_data_float = []
    for entry in raw_data:
        raw_data_float.append(float(entry))
    raw_data = raw_data_float

    # Reshape and get the number of time steps. Both spatial dimensions are 1 for this dataset.
    T = int(len(raw_data))
    S1, S2 = 1, 1
    total_snow = np.array(raw_data).reshape(T, 1)

    # Apply log transform if needed (Saatci et al. do this)
    if log_transform_data:
        total_snow = np.log(total_snow + 1)

    # Normalize if needed (Saatchi et al. do this)
    if normalise_data:
        total_snow = (total_snow - np.mean(total_snow)) / np.sqrt(np.var(total_snow))

    return total_snow, S1, S2, T


if __name__ == "__main__":

    # Set some paths
    data_directory = os.path.join(os.getcwd(), "Data")
    output_directory = os.path.join(os.getcwd(), "Output")

    """Step 1: Get parameter values and other options from command line arguments"""

    # Set up the parser
    parser = argparse.ArgumentParser(
        description="Options for applying the BOCPDMS algorithm to the Whistler snowfall dataset.")
    parser.add_argument("-f", "--figures", type=bool, default=False, help="Save figures")
    parser.add_argument("-a", "--prior_a", type=float, default=1.0, help="Initial value of a")
    parser.add_argument("-b", "--prior_b", type=float, default=1.0, help="Initial value of b")
    parser.add_argument("-vs", "--prior_var_scale", type=float, default=1.0,
                        help="Variance scale used to calculate V_0")
    parser.add_argument("-i", "--intensity", type=float, default=100, help="Intensity")
    parser.add_argument("-lAR", "--lower_AR", type=int, default=1, help="Lower lag length for AR models")
    parser.add_argument("-uAR", "--upper_AR", type=int, default=None, help="Upper lag length for AR models")

    # Get settings and parameter values from the command line arguments
    args = parser.parse_args()
    save_figures = args.figures
    prior_a = args.prior_a
    prior_b = args.prior_b
    prior_var_scale = args.prior_var_scale
    intensity = args.intensity
    lower_AR = args.lower_AR
    upper_AR = args.upper_AR    # If not provided, will be set based on length of time-series after loading data

    # Other settings that we shouldn't change
    normalize = True        # Following Saatci et al. (2010)
    log_transform = True    # Also as Saatci et al. (2010)
    intercept_scale = 0     # Our prior belief is that no snow falls (min is like zero before normalized log transform)
    hyperpar_opt = "caron"  # Online optimisation

    """Step 2: Load the data"""

    data, S1, S2, T = load_snowfall_data(data_directory, normalize, log_transform)

    """Step 3: Set up model properties"""

    # In the first model, we'll set prior_mean_beta to the min amount of snow
    minimum = np.min(data)

    # Set up prior on changepoint distribution
    cp_model = CpModel(intensity)

    # If not specified already, base the upper lag limit on T
    mult = 1.5
    if upper_AR is None:
        upper_AR = int(mult * pow(float(T) / np.log(T), 0.25) + 1)

    """Step 4: Set up the models"""

    AR_models = []

    # Fit only constant
    AR_models += [BVARNIG(prior_a=prior_a, prior_b=prior_b,
                          S1=S1, S2=S2,
                          prior_mean_beta=minimum,  # i.e., set prior to 0 snowfall
                          prior_var_scale=prior_var_scale,
                          intercept_grouping=None,
                          nbh_sequence=None,
                          restriction_sequence=None,
                          hyperparameter_optimization=hyperpar_opt)]

    for lag in range(lower_AR, upper_AR + 1):
        # Set the prior of the coefficients s.t. the intercept can be different from
        # the lag-coefficient priors.
        prior_mean_beta = np.array([intercept_scale] + [0] * lag)

        # Fit some dynamic parts
        AR_models += [BVARNIG(prior_a=prior_a, prior_b=prior_b,
                              S1=S1, S2=S2,
                              prior_mean_beta=prior_mean_beta,
                              prior_var_scale=prior_var_scale,
                              intercept_grouping=None,
                              nbh_sequence=[0] * lag,
                              restriction_sequence=[0] * lag,
                              hyperparameter_optimization=hyperpar_opt)]

    # Model universe and model priors
    model_universe = np.array(AR_models)
    model_prior = np.array([1 / len(model_universe)] * len(model_universe))

    """Step 5: Run detector"""

    # Build and run detector
    detector = Detector(data=data,
                        model_universe=model_universe,
                        model_prior=model_prior,
                        cp_model=cp_model,
                        S1=S1, S2=S2, T=T,
                        store_rl=True, store_mrl=True,
                        trim_type="keep_K", threshold=200,
                        training_period=25,  # i.e., we let 2 years pass before MSE computed
                        notifications=1500,
                        save_performance_indicators=True)
    detector.run()

    """Step 5: Store results in an EvaluationTool object and save"""

    EvT = EvaluationTool()
    EvT.build_EvaluationTool_via_run_detector(detector)
    EvT.store_results_to_HD(os.path.join(output_directory, "results_whistler.txt"))

    """Step 6: Optionally, prepare plots"""

    if save_figures:

        # Plot transformed data
        fig, ax = plt.subplots()
        EvT.plot_raw_TS(data.reshape(T, 1), ax=ax)
        fig.savefig(os.path.join(output_directory, "whistler_transformed_data.pdf"), format="pdf", dpi=800)

        # Plot prediction error
        fig, ax = plt.subplots()
        EvT.plot_prediction_error(data, indices=[0], ax=ax,
                                  time_range=np.linspace(2*365, T-upper_AR-1, T-upper_AR-1-2*365, dtype=int))
        fig.savefig(os.path.join(output_directory, "whistler_prediction_error.pdf"), format="pdf", dpi=800)

        # Plot predictions themselves
        fig, ax = plt.subplots()
        EvT.plot_predictions(indices=[0],
                             ax=ax,
                             legend=False,
                             legend_labels=None,
                             legend_position=None,
                             time_range=None,
                             show_var=False,
                             show_CPs=True)
        fig.savefig(os.path.join(output_directory, "whistler_predictions.pdf"), format="pdf", dpi=800)

    """Step 7: Output performance indicators for the paper"""

    # Print results for the table in the ICML paper
    print("\n")
    print("***** Predictive MSE + NLL from Table 1 in ICML 2018 paper *****")
    print("MSE is %.5g with 95%% error of %.5g" %
          (np.mean(detector.MSE), 1.96*scipy.stats.sem(detector.MSE)))
    print("NLL is %.5g with 95%% error of %.5g" %
          (np.mean(detector.negative_log_likelihood), 1.96*scipy.stats.sem(detector.negative_log_likelihood)))

    # And finally, print out the settings used
    print("\n")
    print("***** Parameter values and other options *****")
    print("prior_a:", prior_a)
    print("prior_b:", prior_b)
    print("prior_var_scale:", prior_var_scale)
    print("intensity:", intensity)
    print("lower_AR:", lower_AR)
    print("upper_AR:", upper_AR)
    print("normalize:", normalize)
    print("log_transform", log_transform)
    print("hyperpar_opt:", hyperpar_opt)
