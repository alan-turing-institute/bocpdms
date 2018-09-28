#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 11:09:08 2018

@author: jeremiasknoblauch

Description: Process the 30-portfolio data
"""

import argparse
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from scipy import stats

from BVAR_NIG import BVARNIG
from detector import Detector
from Evaluation_tool import EvaluationTool
from cp_probability_model import CpModel


def load_portfolios_data(path_to_data, data_file, dates_file):
    """STEP 1: Read in data and dates"""
    mylist = []
    count = 0
    with open(os.path.join(path_to_data, data_file)) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # DEBUG: Unclear if this is needed
            if count > 0:
                mylist += row
            count += 1

    """transform into floats and bring into numpy format"""
    mylist2 = []
    for entry in mylist:
        mylist2 += [float(entry)]
    data = np.array([mylist2]).reshape(int(len(mylist2) / 30), 30)
    T, S1, S2 = data.shape[0], data.shape[1], 1

    """Read in the dates"""
    myl = []
    count = 0
    with open(os.path.join(path_to_data, dates_file)) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # DEBUG: Unclear if this is needed
            if count > 0:
                myl += row
            count += 1

    dates = []
    for e in myl:
        dates.append(int(e))

    return data, dates, S1, S2, T


def read_nbhs(data_dir, mode):
    """STEP 1: Read in the cutoffs"""
    cutoffs_file = os.path.join(data_dir, mode, "portfolio_grouping_cutoffs.csv")
    mylist_cutoffs = []
    count = 0
    with open(cutoffs_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if count > 0:
                mylist_cutoffs += row
            count += 1

    """STEP 2: Determine the number of nbhs and the number of decades stored"""
    num_decades = (len([x[0] for x in os.walk(os.path.join(data_dir, mode))]) - 1)
    num_nbhs = len(mylist_cutoffs) - 2  # -2 because this includes 0 and 0.99

    """STEP 3: For each decade, read in the nbh structure"""
    list_of_nbhs_all_decades = []
    for decade in range(1, num_decades + 1):
        decade_folder = os.path.join(data_dir, mode, "decade_" + str(decade))

        """STEP 3.1: Read in the list of nbhs for the current decade"""
        list_of_nbhs = []
        for nbh_count in range(1, num_nbhs + 1):
            nbh_file = os.path.join(decade_folder, "portfolio_grouping_" + str(nbh_count) + ".csv")
            mylist_helper = []

            """read in"""
            count = 0
            with open(nbh_file) as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    if count > 0:
                        mylist_helper += row
                    count += 1

            """convert to bool"""
            mylist_helper2 = (np.array(mylist_helper.copy()) == 'TRUE').tolist()

            """put in 30x30 matrix and save in list_of_nbhs"""
            nbh_matrix = np.array(mylist_helper2.copy()).reshape(30, 30)
            list_of_nbhs.append(nbh_matrix.copy())

        """STEP 3.2: Store the nbhs of this decade"""
        list_of_nbhs_all_decades.append(list_of_nbhs.copy())

    """STEP 4: The nbhs are save in matrix form s.t. entry 0 corr. to the 1st 
    furthest away nbh-matrix, entry 1 to the 2nd furthest away, .... etc, 
    so we now need to convert them into the format accepted by BVARNIG objects,
    since that is what we ultimately want them for."""
    nbh_indices_all_decades = []
    for decade in range(1, num_decades + 1):
        nbh_indices = []
        for j in range(0, S1 * S2):
            """each location gets its list of nbhs"""
            nbh_indices.append([])
            for i in np.linspace(num_nbhs - 1, 0, num=num_nbhs, dtype=int):  # range(0, num_nbhs):
                """np array of dim 30x30"""
                nbh_matrix = list_of_nbhs_all_decades[decade - 1][i]  # list_of_nbhs[i]
                """add the i-th nbh to location j's list"""
                indices = np.where(nbh_matrix[j,] > 0)[0].tolist()
                nbh_indices[j].append(indices.copy())
        nbh_indices_all_decades.append(nbh_indices.copy())

    """STEP 5: Lastly, just return the results of our analysis"""
    return num_decades, num_nbhs, nbh_indices_all_decades


def get_neighbourhoods_sic():
    """STEP 5: SIC nbhs"""

    """STEP 5.1: Which industries belong to which SIC code? 
        The SIC codes of the 30 portfolio data can be found here:
        http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html"""
    SIC_memberships = [
        [0, 3],     # Food
        [3],        # Beer
        [3],        # Smoke
        [3, 8],     # Games
        [3],        # Books
        [3],        # Household
        [3],        # Clothes
        [3, 8],     # Health
        [3],        # Chemicals
        [3],        # Textiles
        [2, 3],     # Construction
        [3],        # Steel
        [3],        # Fabricated Products, machinery
        [3],        # Electrical equipment
        [3],        # autos
        [3],        # Carry: ships, airplanes, ...
        [1],        # Mines
        [1],        # Coal
        [1, 3],     # Oil
        [4],        # Utilities
        [4],        # communications
        [8],        # services
        [3, 8],     # business equipment
        [3],        # Paper
        [4],        # transport
        [5],        # wholesale
        [6],        # retail
        [6, 8],     # meals (restaurants, hotel, motel)
        [7],        # finance
        [4]         # other
    ]

    """STEP 5.2: Now, make list of length 9 with entries from 0-8 collecting 
    the portfolio indices belonging to that SIC group"""
    SIC_indices = []
    for SIC_code in range(0,9):
        new_list = []
        count = 0
        for portfolio_index, sublist in enumerate(SIC_memberships):
            for entry in sublist:
                if entry == SIC_code:
                    new_list.append(portfolio_index)
            count = count + 1
        SIC_indices.append(new_list.copy())

    """STEP 5.3: Next, convert this into warranted form: First nbh = all with same
    SIC code. Second nbh = all that do NOT have one of your SIC codes"""

    nbhs_SIC = [[[]]] * 30
    for g_index, group in enumerate(SIC_indices):
        for entry in group:
            if not group == [entry]:
                ind2 = group.copy()
                ind2.remove(entry)
                nbhs_SIC[entry] = [ind2]

    return nbhs_SIC


if __name__ == "__main__":

    #  Set up the parser
    parser = argparse.ArgumentParser(
        description="Options for applying the BOCPDMS algorithm to the bee waggle dance dataset.")
    parser.add_argument("-f", "--figures", type=bool, default=False, help="Show figures")
    parser.add_argument("-s", "--short", type=int, default=None,
                        help="Only use the first s observations (recommended >=350")
    parser.add_argument("-a", "--prior_a", type=float, default=100, help="Initial value of a")
    parser.add_argument("-b", "--prior_b", type=float, default=0.001, help="Initial value of b")
    parser.add_argument("-ms", "--prior_mean_scale", type=float, default=0.0,
                        help="Mean scale used to calculate beta_0")
    parser.add_argument("-vs", "--prior_var_scale", type=float, default=0.001,
                        help="Variance scale used to calculate V_0")
    parser.add_argument("-i", "--intensity", type=float, default=1000, help="Intensity")

    # Get settings and parameter values from the command line arguments
    args = parser.parse_args()
    show_figures = args.figures
    prior_a = args.prior_a
    prior_b = args.prior_b
    prior_mean_scale = args.prior_mean_scale
    prior_var_scale = args.prior_var_scale
    intensity = args.intensity

    # If the short argument isn't set, run the algorithm on the entire dataset
    if args.short is None:
        shortened = False
        shortened_T = None
        print("Using the full dataset")
    else:
        shortened = True
        shortened_T = args.short
        print("Using the first", shortened_T, "observations")

    """Modes when running the code (these shouldn't need to be changed)"""
    AR_selections = [1, 5]
    hyperpar_opt = "caron"          # On-line hyperparameter optimization
    normalize = False
    build_weak_coupling = True
    build_strong_coupling = False   # i.e., each Portfolio has its own parameters
    build_sic_nbhs = True           # i.e., include neighbourhood system built on SIC codes
    build_cor_nbhs = True           # i.e., include NBHS built on contemporaneous correlation
    build_autocorr_nbhs = True      # i.e., include NBHS built on autocorrelation
    AR_nbhs = True
    heavy_tails_transform = True    # Use normal(t(y)) transform as in Turner, Saatci, and al.
    time_frame = "comparison"       # "comparison", last_20", "last_10";
                                    # "comparison" looks at 03/07/1975 -- 31/12/2008,
                                    # "last_20" looks at last 20 years before 31/01/2018
                                    # neighbourhoods will be different depending on the mode

    """folder containing dates and data (with result folders being created at 
    run-time if necessary)"""
    baseline_working_directory = os.getcwd()
    data_directory = os.path.join(baseline_working_directory, "Data", "30PF")
    results_directory = os.path.join(baseline_working_directory, "Output", "30PF")

    """dates, e.g. 25/12/1992 is 19921225, corresponding to the observations"""
    file_name_dates = "portfolio_dates.csv"

    """30 Portfolio data. In same order as original data set"""
    file_name_data = "portfolio_data.csv"

    """prototype of the portfolio grouping names that give the list of nbhs 
    for each location, i.e. give the list of nbhs for each Portfolio."""
    file_name_nbhs_proto = os.path.join(data_directory, "portfolio_grouping_")

    """Load the dates and data for the 30 portfolios"""
    data, dates, S1, S2, T = load_portfolios_data(data_directory, file_name_data, file_name_dates)

    """STEP 2: get the grouping for intercepts"""
    grouping = np.zeros((S1 * S2, S1 * S2))
    for i in range(0, S1 * S2):
        grouping[i, i] = 1
    grouping = grouping.reshape((S1 * S2, S1, S2))

    """STEP 4: Read in the autocorr/corr nbhs, and those from the SIC codes"""
    __, __, nbhs_contemp = read_nbhs(data_directory, "contemporaneous")
    __, __, nbhs_autocorr = read_nbhs(data_directory, "autocorr")
    nbhs_SIC = get_neighbourhoods_sic()

    """STEP 4.2: Depending on the mode, select decades of interest (adding one extra year as a test set,
     as in Saatci et al)"""
    num_trading_days = 252
    if time_frame == "comparison":  # Find the indices that correspond to 03/07/1975 -- 31/12/2008
        decades_of_interest = [2, 3, 4]
        start_test = dates.index(19740703)
        start_algo = dates.index(19750703)
        stop = dates.index(20081231)
        selection = np.linspace(start_test, stop, stop - start_test, dtype=int)
        test_obs = start_algo - start_test
    elif time_frame == "last_20":  # Find the indices that correspond to the last 20 years
        decades_of_interest = [4, 5, 6]
        years_analyzed = 21
        selection = np.linspace(len(dates) - 1 * num_trading_days * years_analyzed, len(dates) - 1,
                                num_trading_days * years_analyzed, dtype=int)
        test_obs = num_trading_days
    elif time_frame == "last_10":  # Find the indices that correspond to the last 10 years
        decades_of_interest = [5, 6, 7]
        years_analyzed = 11
        selection = np.linspace(len(dates) - 1 * num_trading_days * years_analyzed, len(dates) - 1,
                                num_trading_days * years_analyzed, dtype=int)
        test_obs = num_trading_days

    """STEP 4.3: Select decades of interest"""
    contemp_nbhs_interest = []
    autocorr_nbhs_interest = []
    for decade_index in decades_of_interest:
        """at each index of contemp_nbhs_interest, we get a nbh structure"""
        contemp_nbhs_interest.append(nbhs_contemp[decade_index])
        autocorr_nbhs_interest.append(nbhs_autocorr[decade_index])

    """STEP 7: Select the time range, apply transform if needed"""
    data = data[selection, :]
    # Do the transform as lined out in thesis of Ryan Turner if needed
    if heavy_tails_transform:
        data = stats.norm.ppf(stats.t.cdf(data, df=4))

    """Shorten the data artificially"""
    if shortened:
        # T = shortened_T
        data = data[:shortened_T, :]

    """Update T after shortening and selection"""
    T = data.shape[0]

    sic_nbhs_res_seq_list = [
        [[0], [0], [0]]
        # [[0]]
    ]
    contemp_nbhs_res_seq_list = [
        [[0,1,2,3], [0,1,2], [0,1], [0]],
        # [[0], [0], [0], [0], [0]],
        # [[0,1], [0,1], [0,1]],
        # [[0], [0], [0]],
        [[0,1,2,3]],
        # [[0,1,2]],
        # [[0,1]]
        # [[0]]
    ]
    autocorr_nbhs_res_seq_list = [
        [[0,1,2,3], [0,1,2], [0,1], [0]],
        # [[0], [0], [0], [0], [0]],
        # [[0,1], [0,1], [0,1]],
        # [[0], [0], [0]],
        [[0,1,2,3]],
        # [[0,1,2]],
        # [[0,1]]
        # [[0]]
    ]

    # REPORTED IN ICML SUBMISSION:
    # comparison with 3 prior decades, 2nbhs per nbh system: [[0,1,2,3], [0,1,2], [0,1], [0]], [[0,1,2,3]]
    # a=100, b=0.001, int = 1000, beta var prior = 0.001, too many CPs [first run], saved on this machine

    """STEP 9+: Normalize if necessary"""
    if normalize:
        data = ((data - np.mean(data, axis=0)) / np.sqrt(np.var(data, axis=0)))

    """STEP 10: Run detectors"""

    cp_model = CpModel(intensity)

    """Create models"""
    all_models = []

    """STEP 10.2: build AR models"""
    if AR_nbhs:
        AR_models = []
        for lag in AR_selections:
            AR_models += [BVARNIG(
                prior_a=prior_a, prior_b=prior_b,
                S1=S1, S2=S2,
                prior_mean_scale=prior_mean_scale,
                prior_var_scale=prior_var_scale,
                intercept_grouping=grouping,
                general_nbh_sequence=np.array([[[]] * lag] * S2 * S2),
                general_nbh_restriction_sequence=np.array([[[0]] * lag] * S2 * S2),
                general_nbh_coupling="weak coupling",
                hyperparameter_optimization=hyperpar_opt)]
        all_models = all_models + AR_models

    """STEP 10.3: build model universe entries with weak coupling"""
    if build_weak_coupling:
        VAR_models_weak = []
        if build_sic_nbhs:
            """Build nbhs based on SIC-induced nbhs"""
            for res in sic_nbhs_res_seq_list:
                VAR_models_weak += [BVARNIG(
                    prior_a=prior_a, prior_b=prior_b,
                    S1=S1, S2=S2,
                    prior_mean_scale=prior_mean_scale,
                    prior_var_scale=prior_var_scale,
                    intercept_grouping=grouping,
                    general_nbh_sequence=nbhs_SIC,
                    general_nbh_restriction_sequence=res,
                    general_nbh_coupling="weak coupling",
                    hyperparameter_optimization=hyperpar_opt)]
        if build_cor_nbhs:
            """Build nbhs based on contemporaneous corr"""
            for nbhs in contemp_nbhs_interest:
                for res in contemp_nbhs_res_seq_list:
                    VAR_models_weak += [BVARNIG(
                        prior_a=prior_a, prior_b=prior_b,
                        S1=S1, S2=S2,
                        prior_mean_scale=prior_mean_scale,
                        prior_var_scale=prior_var_scale,
                        intercept_grouping=grouping,
                        general_nbh_sequence=nbhs,
                        general_nbh_restriction_sequence=res,
                        general_nbh_coupling="weak coupling",
                        hyperparameter_optimization=hyperpar_opt)]

        if build_autocorr_nbhs:
            """Build nbhs based on autocorr"""
            for nbhs in autocorr_nbhs_interest:
                for res in autocorr_nbhs_res_seq_list:
                    VAR_models_weak += [BVARNIG(
                        prior_a=prior_a, prior_b=prior_b,
                        S1=S1, S2=S2,
                        prior_mean_scale=prior_mean_scale,
                        prior_var_scale=prior_var_scale,
                        intercept_grouping=grouping,
                        general_nbh_sequence=nbhs,
                        general_nbh_restriction_sequence=res,
                        general_nbh_coupling="weak coupling",
                        hyperparameter_optimization=hyperpar_opt)]

        all_models = all_models + VAR_models_weak

    """STEP 10.4: build model universe entries with strong coupling"""
    if build_strong_coupling:
        VAR_models_strong = []
        if build_sic_nbhs:
            """Build nbhs based on SIC-induced nbhs"""
            for res in sic_nbhs_res_seq_list:
                VAR_models_strong += [BVARNIG(
                    prior_a=prior_a, prior_b=prior_b,
                    S1=S1, S2=S2,
                    prior_mean_scale=prior_mean_scale,
                    prior_var_scale=prior_var_scale,
                    intercept_grouping=grouping,
                    general_nbh_sequence=nbhs_SIC,
                    general_nbh_restriction_sequence=res,
                    general_nbh_coupling="strong coupling",
                    hyperparameter_optimization=hyperpar_opt)]
        if build_cor_nbhs:
            """Build nbhs based on contemporaneous corr"""
            for nbhs in contemp_nbhs_interest:
                for res in contemp_nbhs_res_seq_list:
                    VAR_models_strong += [BVARNIG(
                        prior_a=prior_a, prior_b=prior_b,
                        S1=S1, S2=S2,
                        prior_mean_scale=prior_mean_scale,
                        prior_var_scale=prior_var_scale,
                        intercept_grouping=grouping,
                        general_nbh_sequence=nbhs,
                        general_nbh_restriction_sequence=res,
                        general_nbh_coupling="strong coupling",
                        hyperparameter_optimization=hyperpar_opt)]

        if build_autocorr_nbhs:
            """Build nbhs based on autocorr"""
            for nbhs in autocorr_nbhs_interest:
                for res in autocorr_nbhs_res_seq_list:
                    VAR_models_strong += [BVARNIG(
                        prior_a=prior_a, prior_b=prior_b,
                        S1=S1, S2=S2,
                        prior_mean_scale=prior_mean_scale,
                        prior_var_scale=prior_var_scale,
                        intercept_grouping=grouping,
                        general_nbh_sequence=nbhs,
                        general_nbh_restriction_sequence=res,
                        general_nbh_coupling="strong coupling",
                        hyperparameter_optimization=hyperpar_opt)]

        all_models = all_models + VAR_models_strong

    model_universe = np.array(all_models)
    model_prior = np.array([1 / len(model_universe)] * len(model_universe))

    """Build and run detector"""
    detector = Detector(data=data, model_universe=model_universe,
                        model_prior=model_prior,
                        cp_model=cp_model, S1=S1, S2=S2, T=T,
                        store_rl=True, store_mrl=True,
                        trim_type="keep_K", threshold=100,
                        notifications=100,
                        save_performance_indicators=True,
                        training_period=test_obs)
    detector.run()

    """Store results + real CPs into EvaluationTool obj"""
    EvT = EvaluationTool()
    EvT.build_EvaluationTool_via_run_detector(detector)

    """Store that EvT object onto hard drive"""
    detector_path = os.path.join(results_directory, time_frame + "_htt" + str(heavy_tails_transform) +
                                 "_a" + str(prior_a) + "_b" + str(prior_b) + "_i" + str(intensity))

    if not os.path.exists(detector_path):
        os.makedirs(detector_path)

    if shortened:
        EvT.store_results_to_HD(os.path.join(detector_path,
                                             "results_30portfolios_short" + str(shortened_T) + ".txt"))
    else:
        EvT.store_results_to_HD(os.path.join(detector_path, "results_30portfolios.txt"))

    if show_figures:

        fig_p, ax_p = plt.subplots()
        EvT.plot_predictions(indices=[0],
                             print_plt=True,
                             legend=False,
                             legend_labels=None,
                             legend_position=None,
                             time_range=None,
                             show_var=False,
                             show_CPs=True,
                             ax=ax_p)
        plt.close(fig_p)

        fig_rld, ax_rld = plt.subplots()
        EvT.plot_run_length_distr(print_plt=True,
                                  show_MAP_CPs=True,
                                  show_real_CPs=False,
                                  mark_median=False,
                                  log_format=True,
                                  CP_legend=False,
                                  buffer=50,
                                  ax=ax_rld)
        plt.close(fig_rld)

        plt.plot(np.linspace(1, len(detector.model_universe[0].a_list),
                             len(detector.model_universe[0].a_list)),
                 np.array(detector.model_universe[0].a_list))
        plt.plot(np.linspace(1, len(detector.model_universe[0].b_list),
                             len(detector.model_universe[0].b_list)),
                 np.array(detector.model_universe[0].b_list))

    # Print results for the table in the ICML paper
    print("\n")
    print("***** Predictive MSE + NLL from Table 1 in ICML 2018 paper *****")
    print("MSE is %.5g with 95%% error of %.5g" %
          (np.sum(np.mean(detector.MSE, axis=0)), np.sum(1.96 * stats.sem(detector.MSE))))
    print("NLL is %.5g with 95%% error of %.5g" %
          (np.mean(detector.negative_log_likelihood), 1.96 * stats.sem(detector.negative_log_likelihood)))

    # Print some details about the CPs and models
    print("\n")
    print("***** CP and model details *****")
    print("MAP CPs at times", [1996.91 + e[0] / 252 for e in detector.CPs[-2]])
    print("MAP models", [e[1] for e in detector.CPs[-2]])

    # Print out the settings
    print("\n")
    print("***** Parameter values and other options *****")
    print("prior_a:", prior_a)
    print("prior_b:", prior_b)
    print("prior_mean_scale:", prior_mean_scale)
    print("prior_var_scale:", prior_var_scale)
    print("intensity:", intensity)
    print("normalize:", normalize)
    print("build_weak_coupling:", build_weak_coupling)
    print("build_strong_coupling:", build_strong_coupling)
    print("build_sic_nbhs:", build_sic_nbhs)
    print("build_cor_nbhs:", build_cor_nbhs)
    print("build_autocorr_nbhs:", build_autocorr_nbhs)
    print("AR_nbhs:", AR_nbhs)
    print("heavy_tails_transform:", heavy_tails_transform)
    print("time_frame:", time_frame)
    print("hyperpar_opt:", hyperpar_opt)
