# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    pdf_phase_time.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <daniloceano@student.42.fr>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/30 19:37:18 by daniloceano       #+#    #+#              #
#    Updated: 2023/10/31 00:55:17 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as grid_spec
from sklearn.neighbors import KernelDensity
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Constants
SECONDS_IN_AN_HOUR = 3600
ALPHA = 0.05  # Significance level
ANALYSIS_TYPE = '70W-no-continental'
METRICS = ['Total Distance ($10^2$ km)', 'Total Time (Hours)', 'Mean Speed (m/s)']
PHASES = ['incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2', 'residual']
REGIONS = ['Total', 'ARG', 'LA-PLATA', 'SE-BR', 'SE-SAO', 'AT-PEN', 'WEDDELL', 'SA-NAM']
PLOT_LABELS = ['(A)', '(B)', '(C)']
COLOR_PHASES = {
        'incipient': '#65a1e6',
        'intensification': '#f7b538',
        'intensification 2': '#ca6702',
        'mature': '#d62828',
        'mature 2': '#9b2226',
        'decay': '#9aa981',
        'decay 2': '#386641',
        'residual': 'gray',
        'Total': '#1d3557'
    }
KDE_PARAMS = {
    'Total Distance ($10^2$ km)': [2, 1000, 98], # [bandwidth, number of samples, quantile]
    'Total Time (Hours)': [2, 1000, 98], 
    'Mean Speed (m/s)': [1.5, 1000, 99]}

def get_database():
    files = f"../periods_species_statistics/{ANALYSIS_TYPE}/periods_database/periods_database_*.csv"
    data_frames = []
    for file in glob.glob(files):
        data_frames.append(pd.read_csv(file, index_col=0))
    database = pd.concat(data_frames, ignore_index=True)
    # Remove rows where all columns are NaN
    database = database.dropna(how='all')
    # Remove rows where "Mean Speed (m/s)" is NaN 
    # (so we won't compute statistics for the first time steps)
    database = database.dropna(subset=['Mean Speed (m/s)'])      
    return database

def remove_outliers(data, column_name):
    """
    Remove outliers using the IQR method.
    """
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter out the outliers
    return data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]


def metric_to_formatted_string(metric):
    mapping = {
        'Total Time (Hours)': 'total_time',
        'Total Distance ($10^2$ km)': 'total_distance',
        'Mean Speed (m/s)': 'mean_speed'
    }
    return mapping.get(metric, '')

def plot_single_ridge(data, figure_path, phase, label, metric):
    """
    Plot a single ridge plot for given data, combining both seasons in one subplot.
    """
    # Create a figure with a row for each region
    gs = grid_spec.GridSpec(len(REGIONS), 1)
    fig = plt.figure(figsize=(10, 10))

    # Calculate the upper percentile to define the x-axis range
    variable = data[metric]
    upper_percentile = np.percentile(variable, KDE_PARAMS[metric][2]) 
    
    # Set the x-axis limits based on the percentiles
    xmin, xmax = 0, upper_percentile
    
    x_d = np.linspace(xmin, xmax, KDE_PARAMS[metric][1])

    for idx, rg in enumerate(REGIONS):
        region_data = data[(data['Region'] == rg)]
        x = np.array(region_data['Total Time (Hours)'])
        
        kde = KernelDensity(bandwidth=KDE_PARAMS[metric][0], kernel='gaussian')
        kde.fit(x[:, None])

        ax = fig.add_subplot(gs[idx:idx+1, 0])

        logprob = kde.score_samples(x_d[:, None])
        ax.plot(x_d, np.exp(logprob), color="#f0f0f0", lw=1, linestyle='-')
        ax.fill_between(x_d, np.exp(logprob), alpha=1, color='b', linestyle='-')
        
        # setting uniform x and y lims
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(0, np.exp(logprob).max() + 0.2)  # added a bit of padding

        # make background transparent
        rect = ax.patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax.set_yticklabels([])
        ax.yaxis.set_ticks([])  # Remove y-axis ticks

        # Determine the y-limits for the grid lines, which would be confined to the KDE
        y_limit = np.exp(logprob).max() + 0.2

        # Calculate grid line positions
        num_lines = 10
        interval = int(round(xmax, -1)) // num_lines
        interval = max(1, interval) + 1

        # Set x-ticks based on the calculated positions
        x_ticks = list(range(0, int(round(xmax, -1)), interval))
        ax.set_xticks(x_ticks)

        # Draw each grid line manually using axvline
        for grid_x in x_ticks:
            ax.axvline(x=grid_x, ymin=0, ymax=y_limit, color='grey', linestyle='--', linewidth=0.5)

        # Calculate the mean of the data for the current region
        meadian_value = np.median(x)

        # Draw a vertical line at the mean value
        ax.axvline(x=meadian_value, ymin=0, ymax=np.exp(logprob).max() + 0.08,
                            color='grey', linestyle='-', linewidth=4, label="Median")

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax.spines[s].set_visible(False)

        if idx != len(REGIONS)-1:                
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', labelsize=14)

        ax.text(-0.5, 0, rg, fontweight="bold", fontsize=14, ha="right")
        
        # season
        if idx == 0:
            ax.text((xmax - xmin) / 2, np.exp(logprob).max() + 0.025, 
                    f"({label}) {phase.capitalize()}", fontweight="bold", fontsize=14, ha="center")

    ax.text((xmax - xmin) / 2, np.exp(logprob).min() - 0.025,
             f'Total Time (Hours)', ha='center', fontsize=16, fontweight="bold")

    # Adjust the position of each row of axes to create overlap
    gs.update(hspace=-0.85)
    
    # Reduce top margin to remove excess white space
    fig.subplots_adjust(top=1.35)
    
    plt.tight_layout()

    # Save the combined figure for the current phase
    metric_formatted = metric_to_formatted_string(metric)
    fname = os.path.join(figure_path, f'{metric_formatted}_{phase}.png')
    plt.savefig(fname, dpi=200)
    print(f"{fname} created.")

def plot_single_phase(ax, x, x_d, kde, phase, xmin, xmax, metric):
    """
    Modular function to plot a single phase.
    """
    logprob = kde.score_samples(x_d[:, None])
    ax.plot(x_d, np.exp(logprob), color="#f0f0f0", lw=1, linestyle='-')
    ax.fill_between(x_d, np.exp(logprob), alpha=1, color=COLOR_PHASES.get(phase, 'gray'), linestyle='-')
    # setting uniform x and y lims
    ax.set_xlim(xmin, xmax)
    logprob = kde.score_samples(x_d[:, None])
    ax.plot(x_d, np.exp(logprob), color="#f0f0f0", lw=1, linestyle='-')
    ax.fill_between(x_d, np.exp(logprob), alpha=1, color=COLOR_PHASES[phase],
                     linestyle='-', label=phase)
    # make background transparent
    rect = ax.patch
    rect.set_alpha(0)
    # remove borders, axis ticks, and labels
    ax.set_yticklabels([])
    ax.yaxis.set_ticks([])  # Remove y-axis ticks
    # Determine the y-limits for the grid lines, which would be confined to the KDE
    if metric == 'Total Distance ($10^2$ km)':
        y_limit = np.exp(logprob).max() + 0.2
        ymax_fraction = 0.6
    elif metric == "Total Time (Hours)":
        y_limit = np.exp(logprob).max() + 0.15
        ymax_fraction = 0.8
    elif metric == "Mean Speed (m/s)":
        y_limit = np.exp(logprob).max() + 0.1
        ymax_fraction = 1
    ax.set_ylim(0, y_limit)
    # Calculate grid line positions
    num_lines = 10
    interval = int(round(xmax, -1)) // num_lines
    interval = max(1, interval) + 1
    # Set x-ticks based on the calculated positions
    x_ticks = list(range(0, int(round(xmax, -1)), interval))
    ax.set_xticks(x_ticks)
    # Draw each grid line manually using axvline
    for grid_x in x_ticks:
        ax.axvline(x=grid_x, ymin=0, ymax=y_limit * ymax_fraction,
                    color='grey', linestyle='--', linewidth=0.5)
    # Calculate the mean of the data for the current region
    mean_value = np.mean(x)
    meadian_value = np.median(x)
    std = np.std(x)
    # Draw a vertical line at the mean value
    max_kde_value = y_limit
    ymax_value = max_kde_value * ymax_fraction
    ax.axvline(x=mean_value, ymin=0, ymax=ymax_value,
                    color='k', linestyle='-', linewidth=2)
    ax.axvline(x=meadian_value, ymin=0, ymax=ymax_value * 1.3,
                    color='k', linestyle='--', linewidth=2, alpha=0.8)
    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax.spines[s].set_visible(False)
    ax.text(1.05, 0.12, f"{mean_value:.1f} ± {std:.1f}", fontweight="bold",
            fontsize=12, ha="right", transform=ax.transAxes)
    return ax

def plot_ridge_phases(data):
    # Only select data for the region "Total"
    data = data[data['Region'] == 'Total']

    # Define a grid layout with one row for each phase and one column for each metric
    gs = grid_spec.GridSpec(len(PHASES) + 1, len(METRICS))
    fig = plt.figure(figsize=(4 * len(METRICS), len(PHASES) * 1.2))

    # Calculate mean values for each phase
    means = {}
    for phase in PHASES:
        phase_data = data[data['phase'] == phase]
        means[phase] = np.mean(phase_data[METRICS[0]])
    
    # Order the phases by their mean values
    sorted_phases = sorted(means, key=means.get)

    for metric_idx, metric in enumerate(METRICS):
        variable = data[metric]
        upper_percentile = np.percentile(variable, KDE_PARAMS[metric][2])
        xmin, xmax = 0, upper_percentile
        x_d = np.linspace(xmin, xmax, KDE_PARAMS[metric][1])
        for idx, phase in enumerate(sorted_phases):
            ax = fig.add_subplot(gs[idx, metric_idx])
            phase_data = data[data['phase'] == phase]
            x = np.array(phase_data[metric])
            kde = KernelDensity(bandwidth=KDE_PARAMS[metric][0], kernel='gaussian')
            kde.fit(x[:, None])
            ax = plot_single_phase(ax, x, x_d, kde, phase, xmin, xmax, metric)
            if idx != len(PHASES)-1:
                ax.set_xticklabels([])    
            elif idx == len(PHASES)-1:
                ax.set_xlabel(metric)   
                ax.tick_params(axis='x', labelsize=10)
            if idx == 0:
                ax.text(0.5, 0.3, PLOT_LABELS[metric_idx], fontweight="bold",
                        fontsize=14, transform=ax.transAxes)

    # Legend customization
    legend_handles = [plt.Line2D([0], [0], color='k', linestyle='-', linewidth=2, label='Mean'),
                      plt.Line2D([0], [0], color='k', linestyle='--', linewidth=2, alpha=0.8, label='Median')]
    # Add phase colors to the legend
    for phase in sorted_phases:
        legend_handles.append(plt.Line2D([0], [0], color=COLOR_PHASES.get(phase, 'gray'),
                                         label=phase, linewidth=6))
    # Place the legend outside the plotting area
    fig.legend(handles=legend_handles, loc='center left',
                bbox_to_anchor=(0.82, 0.5), fontsize=12, title='Phases')
    # Adjust the position of each row of axes to create overlap
    gs.update(hspace=-0.82, wspace=0.25)
    plt.tight_layout()
    plt.subplots_adjust(top=1.25, right=0.8, bottom=-0.01)
    figure_path = os.path.join('..', 'figures', 'manuscript_life-cycle')
    fname = os.path.join(figure_path, f'PDF_total_all_phases_all_metrics.png')
    plt.savefig(fname, dpi=200)
    print(f"{fname} created.")

def plot_ridge_plots(database):
    # for metric in METRICS:
        # figure_path = os.path.join('..', 'figures', 'manuscript_life-cycle', 'PDFs')
        # os.makedirs(figure_path, exist_ok=True)
        # for phase, label in zip(PHASES, PLOT_LABELS):
        #     print('\n-----------------')
        #     print(f"Plotting for phase: {phase}")
        #     data = database[(database['phase'] == phase) & (database['Season'] == 'Total')]
        #     plot_single_ridge(data, figure_path, phase, label, metric)
    plot_ridge_phases(database)

def main():
    database = get_database()
    database['Total Distance ($10^2$ km)'] = database['Total Distance (km)'] / 100
    plot_ridge_plots(database)

if __name__ == '__main__':
    main()

