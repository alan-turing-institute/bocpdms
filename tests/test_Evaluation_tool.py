# -*- coding: utf-8 -*-

"""Tests for the Evaluation_tool module."""

import pytest
import numpy as np
import os
from Evaluation_tool import EvaluationTool
from cp_probability_model import CpModel
from BVAR_NIG import BVARNIG
from detector import Detector


@pytest.fixture
def example_detector():
    """Returns a detector that has been run to identify CPs in a very simple simulated dataset."""

    # Set the size of the test dataset
    T_region1 = 60
    T_region2 = 40
    T = T_region1 + T_region2
    S1 = 1
    S2 = 1

    # Create some random data and standardise
    np.random.seed(10)
    y1 = np.random.exponential(1.0, T_region1)
    y2 = np.random.exponential(3.0, T_region2)
    y = np.append(y1, y2)
    mean, variance = np.mean(y), np.var(y)
    y = (y - mean) / np.sqrt(variance)

    # Set up the AR-models
    upper_AR = 3
    lower_AR = 1
    AR_models = []
    for lag in range(lower_AR, upper_AR + 1):
        AR_models += [BVARNIG(prior_a=1, prior_b=1,
                              S1=S1, S2=S2,
                              prior_var_scale=0.075,
                              nbh_sequence=[0] * lag,
                              restriction_sequence=[0] * lag)]

    # Build the detector
    detector = Detector(data=y,
                        model_universe=np.array(AR_models),
                        model_prior=np.array([1 / (upper_AR - lower_AR + 1)] * (upper_AR - lower_AR + 1)),
                        cp_model=CpModel(100),
                        S1=S1, S2=S2, T=T,
                        store_rl=True, store_mrl=True,
                        trim_type="keep_K", threshold=50,
                        save_performance_indicators=True,
                        generalized_bayes_rld="kullback_leibler",
                        alpha_param_learning="individual",
                        alpha_param=0.01,
                        alpha_param_opt_t=30,
                        alpha_rld_learning=True,
                        loss_der_rld_learning="squared_loss",
                        loss_param_learning="squared_loss",
                        training_period=40)

    # Return the detector without running
    return detector


def test_constructor():

    # Initialise an EvaluationTool
    evt = EvaluationTool()

    # Check that general properties have been set correctly
    assert evt.type is None
    assert evt.has_true_CPs is False
    assert evt.true_CP_location is None
    assert evt.true_CP_model_index is None
    assert evt.true_CP_model_label is None

    # Check that properties related to plotting functionality are of the correct type
    # (but we aren't concerned about precisely which colours/linestyles are used).
    possible_colours = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')
    possible_linestyles = ('-', '--', '-.', ':')
    assert evt.cushion > 0
    assert evt.CP_color in possible_colours
    assert evt.median_color in possible_colours
    assert evt.max_color in possible_colours
    assert all([c in possible_colours for c in evt.colors])
    assert all([l in possible_linestyles for l in evt.linestyle])

    # Check that the correct list of names has been prepared - results corresponding to this list should be added later
    assert evt.names == ["names", "execution time", "S1", "S2", "T",
                         "trimmer threshold", "MAP CPs", "model labels",
                         "run length log distribution",
                         "model and run length log distribution",
                         "one-step-ahead predicted mean",
                         "one-step-ahead predicted variance",
                         "all run length log distributions",
                         "all model and run length log distributions",
                         "all retained run lengths",
                         "has true CPs",  "true CP locations",
                         "true CP model index", "true CP model labels"]


def test_initialise_from_run_detector(example_detector):

    det = example_detector
    det.run()
    evt = EvaluationTool()
    evt.build_EvaluationTool_via_run_detector(det)

    # If initialised with a detector that has already been run, the evt should be type 4 (of 4)
    assert evt.type == 4

    # Check that the evt created using this detector has stored the expected values
    assert 1.6232197650790918 == pytest.approx(np.mean(evt.MSE), 1e-8)
    assert 1.6059882853108407 == pytest.approx(np.mean(evt.negative_log_likelihood), 1e-8)

    # Check the structure of the list of results
    assert evt.names == evt.results[0]              # Names of quantities are duplicated; should be the same
    assert len(evt.results[0]) == len(evt.results)  # Check that the number of names corresponds to the no. results


def test_add_true_cps(example_detector):

    det = example_detector
    det.run()

    # Add the true CPs to the evt
    evt = EvaluationTool()
    assert evt.has_true_CPs is False
    evt.add_true_CPs(50, 4)
    assert evt.has_true_CPs is True

    # Build the evt using the detector, then check that true CP is stored in results
    evt.build_EvaluationTool_via_run_detector(det)
    assert evt.results[evt.results[0].index("has true CPs")] is True
    assert evt.results[evt.results[0].index("true CP locations")] == 50


def test_initialise_from_not_run_detector(example_detector):

    det = example_detector
    evt = EvaluationTool()

    # Todo: throw a more informative error rather than print statement + AttributeError
    with pytest.raises(AttributeError):
        evt.build_EvaluationTool_via_run_detector(det)


def test_save_and_initialise_from_results(example_detector, tmpdir):

    det = example_detector
    det.run()
    evt = EvaluationTool()
    evt.build_EvaluationTool_via_run_detector(det)

    # Save to HD
    evt.store_results_to_HD(tmpdir.join("test_save_evt.txt"))
    assert os.path.isfile(tmpdir.join("test_save_evt.txt"))

    # Load into a new evt
    evt_load = EvaluationTool()
    evt_load.build_EvaluationTool_via_results(tmpdir.join("test_save_evt.txt"))

    # Check that key quantities have been restored
    assert evt_load.type == 4                                           # Detector has already been run
    assert evt_load.results[1] == evt.results[1]                        # execution time
    for rlld_l, rlld in zip(evt_load.results[8], evt.results[8]):       # run length log distribution
        assert rlld_l == rlld
