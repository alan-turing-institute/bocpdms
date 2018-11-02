#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:18:58 2018

@author: jeremiasknoblauch

Description: Get the Nile data pics
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

from Evaluation_tool import EvaluationTool
from nile_ICML18 import load_nile_data

# Choose whether to generate and save additional figures (not used in the paper)
parser = argparse.ArgumentParser(description="Prepare figures related to the Nile dataset.")
parser.add_argument("-e", "--extra", type=bool, default=False, help="Produce extra figures")
args = parser.parse_args()
save_extra_figures = args.extra

# Ensure that we have type 1 fonts (for ICML publishing guidelines)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Set paths to original data and stored results
baseline_working_directory = os.getcwd()
nile_file = os.path.join(baseline_working_directory, "Data", "nile.txt")
results_file = os.path.join(baseline_working_directory, "Output", "results_nile.txt")
if not os.path.isfile(results_file):
    print("\nCould not find results_nile.txt in the Output directory. Have you run nile_ICML18.py?\n")

"""STEP 1: Read in data and convert to appropriate format"""

# Extract data (height and date) and properties (T: num years; S1 and S2: spatial dimensions) from csv file
T, S1, S2, river_height, unstandardised_river_height, dates = load_nile_data(nile_file)

"""Step 2: Generate EvaluationTool based on the results generated with nile_ICML18.py"""

# Read in results
EvT = EvaluationTool()
EvT.build_EvaluationTool_via_results(results_file) 

# Get MAP CPs in range
print("Identified", len(EvT.results[EvT.names.index("MAP CPs")][-2]), "MAP CPs at years: ")
[print("\t", int(dates[m[0]])) for m in EvT.results[EvT.names.index("MAP CPs")][-2]]

"""STEP 3: Generate run-length distribution plot for paper"""

# Prepare the axes
fig, ax_array = plt.subplots(2, figsize=(8, 5), sharex=True,
                             gridspec_kw={'height_ratios': [10, 14]})
plt.subplots_adjust(hspace=.35, left=None, bottom=None, right=None, top=None)

# Placement of y-labels
ylabel_coords = [-0.065, 0.5]

# Upper panel: plot the time-series
EvT.plot_raw_TS(river_height[2:].reshape(T-2, 1),
                indices=[0],
                xlab=None,
                show_MAP_CPs=True,
                time_range=np.linspace(1, T-2, T-2, dtype=int),
                print_plt=False,
                ylab="River Height", ax=ax_array[0],
                all_dates=np.linspace(int(dates[1]), int(dates[-2]), T-2, dtype=int),
                custom_colors_series=["black"],
                custom_colors_CPs=["blue", "blue"],
                custom_linestyles=["solid"] * 2,
                ylab_fontsize=14,
                ylabel_coords=ylabel_coords,
                set_ylims=(-2.75, 3.95))

# Lower panel: plot the run length distribution
EvT.plot_run_length_distr(buffer=0,
                          show_MAP_CPs=True,
                          mark_median=False,
                          mark_max=True,
                          upper_limit=660,
                          print_colorbar=True,
                          colorbar_location='bottom',
                          log_format=True,
                          aspect_ratio='auto',
                          C1=0, C2=1,
                          time_range=np.linspace(1, T-2, T-2, dtype=int),
                          start=int(dates[2]), stop=int(dates[-1]),
                          event_time_list=[715],    # Nilometer installed in year 715
                          label_list=["nilometer"],
                          space_to_colorbar=0.52,
                          custom_colors=["blue", "blue"],
                          custom_linestyles=["solid"] * 3,
                          custom_linewidth=3,
                          arrow_colors=["black"],
                          number_fontsize=14,
                          arrow_length=135,
                          arrow_thickness=3.0,
                          xlab_fontsize=14,
                          ylab_fontsize=14,
                          arrows_setleft_indices=[0],
                          arrows_setleft_by=[50],
                          zero_distance=0.0,
                          ax=ax_array[1], figure=fig,
                          no_transform=True,
                          date_instructions_formatter=None,
                          date_instructions_locator=None,
                          ylabel_coords=ylabel_coords,
                          xlab="Year",
                          arrow_distance=25)
    

# Save the generated figures
fig.savefig(os.path.join(baseline_working_directory, "Output", "ICML18_Figure5_Nile.pdf"),
            format="pdf", dpi=800)
fig.savefig(os.path.join(baseline_working_directory, "Output", "ICML18_Figure5_Nile.png"),
            format="png", dpi=800)

# Check that the figure has indeed been generated and report
if os.path.isfile(os.path.join(baseline_working_directory, "Output", "ICML18_Figure5_Nile.pdf")):
    print("Figure 5 can be found in the Output directory")

"""STEP 4: Generate additional figures, if requested"""

if save_extra_figures:

    # Plot the prediction error and variance
    fig_pe, ax_pe = plt.subplots()
    EvT.plot_prediction_error(river_height.reshape(T, 1, 1), indices=[0],
                              show_var=True, show_MAP_CPs=True,
                              up_to=250, ax=ax_pe)
    fig_pe.savefig(os.path.join(baseline_working_directory, "Output", "ICML18_ExtraFigure_Nile_PredError.pdf"),
                   format="pdf", dpi=800)
    if os.path.isfile(os.path.join(baseline_working_directory, "Output", "ICML18_ExtraFigure_Nile_PredError.pdf")):
        print("Extra plot of the prediction error can be found in the Output directory")

    # Plot the model posterior
    fig_mp, ax_mp = plt.subplots()
    EvT.plot_model_posterior(indices=[0, 1, 2], log_format=False, up_to=150,
                             plot_type="MAPVariance1_det", show_MAP_CPs=True, ax=ax_mp)
    fig_mp.savefig(os.path.join(baseline_working_directory, "Output", "ICML18_ExtraFigure_Nile_ModelPosterior.pdf"),
                   format="pdf", dpi=800)
    if os.path.isfile(os.path.join(baseline_working_directory, "Output", "ICML18_ExtraFigure_Nile_ModelPosterior.pdf")):
        print("Extra plot of the model posterior can be found in the Output directory")
