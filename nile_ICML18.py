#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 15:45:27 2018

@author: Jeremias Knoblauch, j.knoblauch@warwick.ac.uk

Description: Reads in the nile data and processes it as a demo
"""

# System packages/modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy
import csv
import os

# Modules of the BOCPDMS algorithm
from cp_probability_model import CpModel
from BVAR_NIG import BVARNIG
from detector import Detector
from Evaluation_tool import EvaluationTool


def load_nile_data(path_to_data):
    """Read in the Nile dataset and convert it to a format suitable for use with the Detector class.
    The original dataset is available from http://mldata.org/repository/data/viewslug/nile-water-level/"""

    """STEP 1: Read in the nile data from nile.txt"""
    raw_data = []
    with open(path_to_data) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            raw_data += row

    raw_data_float = []
    for entry in raw_data:
        raw_data_float.append(float(entry))
    raw_data = raw_data_float

    """STEP 2: Format the nile data so that it can be processed with a Detector
    object and instantiations of ProbabilityModel subclasses"""
    T = int(len(raw_data) / 2)
    S1, S2 = 1, 1  # S1, S2 give you spatial dimensions, but Nile is univariate.
    data = np.array(raw_data).reshape(T, 2)
    dates = data[:, 0]
    river_height = data[:, 1]
    mean, variance = np.mean(river_height), np.var(river_height)

    """STEP 3: Standardize in order to be able to compare with GP-approaches"""
    standardised_river_height = (river_height - mean) / np.sqrt(variance)

    return T, S1, S2, standardised_river_height, river_height, dates


if __name__ == "__main__":

    """STEP 1: Set the current working directory and read in the data."""
    baseline_working_directory = os.getcwd()
    nile_file = os.path.join(baseline_working_directory, "Data", "nile.txt")
    T, S1, S2, river_height, __, __ = load_nile_data(nile_file)     # Use standardised river height

    """STEP 2: Set up initial hyperparameters (will be optimized throughout
    the algorithm) and lag lengths"""

    #  Set up the parser
    parser = argparse.ArgumentParser(
        description="Options for applying the BOCPDMS algorithm to the bee waggle dance dataset.")
    parser.add_argument("-a", "--prior_a", type=float, default=1.0, help="Initial value of a")
    parser.add_argument("-b", "--prior_b", type=float, default=1.0, help="Initial value of b")
    parser.add_argument("-ms", "--prior_mean_scale", type=float, default=0.0,
                        help="Mean scale used to calculate beta_0")
    parser.add_argument("-vs", "--prior_var_scale", type=float, default=0.0075,
                        help="Variance scale used to calculate V_0")
    parser.add_argument("-i", "--intensity", type=float, default=100, help="Intensity")
    parser.add_argument("-lAR", "--lower_AR", type=int, default=1, help="Lower lag length for AR models")
    parser.add_argument("-uAR", "--upper_AR", type=int, default=3, help="Upper lag length for AR models")

    args = parser.parse_args()
    prior_a = args.prior_a          # a, b: inverse Gamma hyperparameters; will be optimized inside detector
    prior_b = args.prior_b
    prior_mean_scale = args.prior_mean_scale
    prior_var_scale = args.prior_var_scale
    intensity = args.intensity
    lower_AR = args.lower_AR        # In the paper, we suggest int(mult*pow(float(T)/np.log(T), 0.25) + 1) for mult = 1
    upper_AR = args.upper_AR        # as the maximum lag length, and 1 for the minimum lag length, but the algorithm is
                                    # relatively insensitive to choosing different pairs

    # Changepoint prior is a constant hazard function
    cp_model = CpModel(intensity)

    # And optimise the hyperparameters online
    hyperpar_opt = "online"

    """STEP 3: Set up the AR-models, run algorithm"""
    AR_models = []
    for lag in range(lower_AR, upper_AR + 1):
        """Generate next model object"""
        AR_models += [BVARNIG(
            prior_a=prior_a, prior_b=prior_b,
            S1=S1, S2=S2,
            prior_mean_scale=prior_mean_scale,
            prior_var_scale=prior_var_scale,
            intercept_grouping=None,
            nbh_sequence=[0] * lag,
            restriction_sequence=[0] * lag,
            hyperparameter_optimization=hyperpar_opt)]

    """STEP 4: Put all model objects together, create model universe, model priors"""
    model_universe = np.array(AR_models)
    model_prior = np.array([1 / len(model_universe)] * len(model_universe))

    """STEP 5: Build and run detector, i.e. the object responsible for executing
    BOCPDMS with multiple (previously specified) models for the segments and a
    CP model specified by cp_model"""
    detector = Detector(
        data=river_height,
        model_universe=model_universe,
        model_prior=model_prior,
        cp_model=cp_model,
        S1=S1, S2=S2, T=T,
        store_rl=True, store_mrl=True,
        trim_type="keep_K", threshold=50,
        notifications=50,
        save_performance_indicators=True)
    detector.run()

    """STEP 6: Store results into EvaluationTool object with plotting capability"""
    EvT = EvaluationTool()
    EvT.build_EvaluationTool_via_run_detector(detector)
    EvT.store_results_to_HD(os.path.join(baseline_working_directory, "Output", "results_nile.txt"))

    """STEP 7: Inspect convergence of the hyperparameters"""
    for lag in range(0, upper_AR + 1 - lower_AR):
        plt.plot(np.linspace(1, len(detector.model_universe[lag].a_list),
                             len(detector.model_universe[lag].a_list)),
                 np.array(detector.model_universe[lag].a_list))
        plt.plot(np.linspace(1, len(detector.model_universe[lag].b_list),
                             len(detector.model_universe[lag].b_list)),
                 np.array(detector.model_universe[lag].b_list))
        plt.savefig(os.path.join(baseline_working_directory, "Output",
                                 "ICML18_ExtraFigure_Nile_ab_lag" +
                                 str(detector.model_universe[lag].lag_length) + ".pdf"),
                    format="pdf", dpi=800)
        plt.cla()

    """STEP 8: Also plot some performance indicators (will usually be printed
    to the console before the plots)"""
    print("\nCPs are ", detector.CPs[-2])
    print("\n***** Predictive MSE + NLL from Table 1 in ICML 2018 paper *****")
    print("MSE is %.5g with 95%% error of %.5g" %
          (np.mean(detector.MSE), 1.96 * scipy.stats.sem(detector.MSE)))
    print("NLL is %.5g with 95%% error of %.5g" %
          (np.mean(detector.negative_log_likelihood), 1.96 * scipy.stats.sem(detector.negative_log_likelihood)))

    """STEP 9: Print out the settings used to get these results"""
    print("\n")
    print("***** Parameter values and other options *****")
    print("prior_a:", prior_a)
    print("prior_b:", prior_b)
    print("prior_mean_scale:", prior_mean_scale)
    print("prior_var_scale:", prior_var_scale)
    print("intensity:", intensity)
    print("lower_AR:", lower_AR)
    print("upper_AR:", upper_AR)
    print("hyperpar_opt:", hyperpar_opt)
