# -*- coding: utf-8 -*-

"""End to end test of changepoint detection for the Nile example."""

import csv
import numpy as np
import os
import pytest

from cp_probability_model import CpModel
from BVAR_NIG import BVARNIG
from detector import Detector


def test_nile_end_to_end():
    """This test is very similar to the script "nile_ICML18.py" - here, we check that we obtain the expected
    numerical results."""

    """Read in the nile data from nile.txt"""
    nile_file = os.path.join(os.getcwd(), "Data", "nile.txt")
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

    """Put data into format compatible with the Detector class"""
    T = int(len(raw_data) / 2)
    S1, S2 = 1, 1
    data = np.array(raw_data).reshape(T, 2)
    dates = data[:, 0]
    river_height = data[:, 1]
    mean, variance = np.mean(river_height), np.var(river_height)
    river_height = (river_height - mean) / np.sqrt(variance)

    """Set up initial hyperparameters (will be optimized throughout 
    the algorithm) and lag lengths"""
    intensity = 100
    cp_model = CpModel(intensity)
    a, b = 1, 1
    prior_mean_scale, prior_var_scale = 0, 0.075

    """Set up the AR-models"""
    upper_AR = 3
    lower_AR = 1
    AR_models = []
    for lag in range(lower_AR, upper_AR + 1):
        """Generate next model object"""
        AR_models += [BVARNIG(
            prior_a=a, prior_b=b,
            S1=S1, S2=S2,
            prior_mean_scale=prior_mean_scale,
            prior_var_scale=prior_var_scale,
            intercept_grouping=None,
            nbh_sequence=[0] * lag,
            restriction_sequence=[0] * lag,
            hyperparameter_optimization="online")]

    """Put all model objects together, create model universe, model priors"""
    model_universe = np.array(AR_models)
    model_prior = np.array([1 / len(model_universe)] * len(model_universe))

    """Build and run detector"""
    detector = Detector(
        data=river_height,
        model_universe=model_universe,
        model_prior=model_prior,
        cp_model=cp_model,
        S1=S1, S2=S2, T=T,
        store_rl=True, store_mrl=True,
        trim_type="keep_K", threshold=50,
        notifications=50,
        save_performance_indicators=True,
        generalized_bayes_rld="kullback_leibler",
        alpha_param_learning="individual",
        alpha_param=0.01,
        alpha_param_opt_t=30,
        alpha_rld_learning=True,
        loss_der_rld_learning="squared_loss",
        loss_param_learning="squared_loss")
    detector.run()

    """Check that the MSE and NLL are close to their expected values"""
    assert np.mean(detector.MSE) == pytest.approx(0.5478817771112774, 1e-8)
    assert np.mean(detector.negative_log_likelihood) == pytest.approx(1.1167794942623417, 1e-8)

    """And check that the final proposed changepoints are those that we expect"""
    assert detector.CPs[-2][0][0] == 3      # First CP at time index 3
    assert detector.CPs[-2][1][0] == 101    # Second CP at time index 101
