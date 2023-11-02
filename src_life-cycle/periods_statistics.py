# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    periods_statistics.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: danilocoutodsouza <danilocoutodsouza@st    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/30 16:09:48 by Danilo            #+#    #+#              #
#    Updated: 2023/11/02 19:59:20 by danilocouto      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import matplotlib.gridspec as grid_spec

from tqdm import tqdm
from sklearn.neighbors import KernelDensity
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Constants
SECONDS_IN_AN_HOUR = 3600
ALPHA = 0.05  # Significance level
ANALYSIS_TYPE = '70W-no-continental'
METRICS = ['Total Distance ($10^2$ km)', 'Total Time (Hours)', 'Mean Speed (m/s)',
            'Mean Vorticity (−1 × 10−5 s−1)', 'Mean Growth rate  (-1 × 10−2 s−1 day-1)']
PHASES = ['incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2', 'residual']
REGIONS = ['Total', 'ARG', 'LA-PLATA', 'SE-BR', 'SE-SAO', 'AT-PEN', 'WEDDELL', 'SA-NAM']
PLOT_LABELS = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)', '(I)']
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
        'Total Time (Hours)': [3, 1000, 95],
        'Total Distance ($10^2$ km)': [2, 1000, 95],
        'Mean Speed (m/s)': [2, 100, 97],
        'Mean Growth rate  (−1 × 10−2 s−1 day-1)': [1.5, 100, 99],
        'Mean Growth rate  (−1 × 10−2 s−1 day-1)': [20, 100, 99] 
    }  # [bandwidth, number of samples, quantile]

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
    database = database.dropna(subset=['Mean Speed (m/s)', 'Mean Growth rate  (−1 × 10−2 s−1 day-1)', 'Mean Growth rate  (−1 × 10−2 s−1 day-1)'])  
    # Simplify units
    database['Total Distance ($10^2$ km)'] = database['Total Distance (km)'] / 100    
    return database

def metric_to_formatted_string(metric):
    mapping = {
        'Total Time (Hours)': 'total_time',
        'Total Distance ($10^2$ km)': 'total_distance',
        'Mean Speed (m/s)': 'mean_speed',
        'Mean Growth rate  (−1 × 10−2 s−1 day-1)': 'mean_intensity',
        'Mean Growth rate  (−1 × 10−2 s−1 day-1)': 'mean_growth'
    }
    return mapping.get(metric, '')

def compare_djf_jja(data_for_region, metric):
    # Extract DJF and JJA data
    djf_data = data_for_region[data_for_region['Season'] == 'DJF'][metric].dropna().values
    jja_data = data_for_region[data_for_region['Season'] =='JJA'][metric].dropna().values

    # Check if lengths are same, if not, trim to shorter length
    min_len = min(len(djf_data), len(jja_data))
    djf_data = djf_data[:min_len]
    jja_data = jja_data[:min_len]

    # Test normality for both DJF and JJA
    _, p_value_djf = stats.shapiro(djf_data)
    _, p_value_jja = stats.shapiro(jja_data)

    # Define significance level, e.g., 0.05
    alpha = 0.05

    if p_value_djf > alpha and p_value_jja > alpha:
        # Both samples are normally distributed
        _, p_value = stats.ttest_rel(djf_data, jja_data)
        test_used = 'paired t-test'
    else:
        # At least one of the samples isn't normally distributed
        _, p_value = stats.wilcoxon(djf_data, jja_data)
        test_used = 'Wilcoxon signed-rank test'

    return test_used, p_value

def plot_single_ridge_season(data, regions, figure_path, phase, metric):
    """
    Plot a single ridge plot for given data, combining both seasons in one subplot.
    """
    # Create a figure with a row for each region
    plt.close("all")
    gs = grid_spec.GridSpec(len(regions), 2)
    fig = plt.figure(figsize=(12, 9))

    variable = data[metric]

    # Calculate the upper percentile to define the x-axis range
    upper_percentile = np.percentile(variable,KDE_PARAMS[metric][2]) 
    
    # Set the x-axis limits based on the percentiles
    xmin, xmax = 0, upper_percentile
    
    x_d = np.linspace(xmin, xmax,KDE_PARAMS[metric][1])

    red_shades = ['#9d0208', '#b30109', '#c7010a', '#d9000b', '#dc2f02', '#e14c03', '#e56904', '#ff9066']
    blue_shades = ['#023e8a', '#0251a0', '#0264b6', '#0077b6', '#0091c3', '#00abd0', '#00c5dd', '#48cae4']

    for idx, rg in enumerate(regions):
        color_djf = red_shades[idx]
        color_jja = blue_shades[idx]

        # Get data for the region
        data_for_region = data[data['Region'] == rg]

        # Compare DJF and JJA data for the region
        test_used, p_value = compare_djf_jja(data_for_region, metric)
        
        for season, col in zip(['DJF', 'JJA'], [0, 1]):                
            season_data = data[(data['Region'] == rg) & (data['Season'] == season)]
            x = np.array(season_data[metric])
            
            kde = KernelDensity(bandwidth=KDE_PARAMS[metric][0], kernel='gaussian')
            kde.fit(x[:, None])

            color = color_djf if season == "DJF" else color_jja

            ax = fig.add_subplot(gs[idx:idx+1, col])

            logprob = kde.score_samples(x_d[:, None])
            ax.plot(x_d, np.exp(logprob), color="#f0f0f0", lw=1, linestyle='-')
            ax.fill_between(x_d, np.exp(logprob), alpha=1, color=color, linestyle='-')
            
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
            interval = round(xmax) // num_lines
            interval = max(1, interval) + 1

            # Set x-ticks based on the calculated positions
            x_ticks = list(range(0, int(round(xmax, -1)), interval))
            ax.set_xticks(x_ticks)

            # Draw each grid line manually using axvline
            for grid_x in x_ticks:
                ax.axvline(x=grid_x, ymin=0, ymax=y_limit, color='grey', linestyle='--', linewidth=0.5)

            # Calculate the mean of the data for the current region
            mean_value = x.mean()
            meadian_value = np.median(x)
            std = np.std(x)

            # Draw a vertical line at the mean value
            ax.axvline(x=mean_value, ymin=0, ymax=np.exp(logprob).max() + 0.2, 
                                color='k', linestyle='-', linewidth=2, label="Mean")
            ax.axvline(x=meadian_value, ymin=0, ymax=np.exp(logprob).max() + 0.2,
                                color='k', linestyle='--', alpha=0.8, linewidth=2, label="Median")        

            spines = ["top", "right", "left", "bottom"]
            for s in spines:
                ax.spines[s].set_visible(False)

            if col == 0:
                ax.text(-0.5, 0, f"{rg}\n", fontweight="bold", fontsize=14, ha="right")
                ax.text(-0.5, 0, f"({mean_value:.1f} ± {std:.1f})", fontsize=10, ha="right")
            else:
                ax.text(np.max(x_d)*1.15, 0.005, f"({mean_value:.1f} ± {std:.1f})", fontsize=10, ha="right")
                if p_value <= ALPHA:
                    ax.text(np.max(x_d) - (np.max(x_d)*1.05), 0.01, "*", fontweight="bold", fontsize=14, ha="right", color="k")

            if idx != len(regions)-1:                
                ax.set_xticklabels([])

            # season
            if idx == 0:
                letter = "A" if col == 0 else "B"
                ax.text((xmax - xmin) / 2, np.exp(logprob).max() + 0.05, f"({letter}) {season}", fontweight="bold", fontsize=14, ha="center")

            # Adjust x-position of the second column to make columns closer
            if col == 1:
                pos = ax.get_position()
                ax.set_position([pos.x0 + 0.01, pos.y0, pos.width, pos.height])
    # Update the title to reflect the metric being analyzed
    metric_name = metric.replace("(", "").replace(")", "").lower()  # Remove any parenthesis
    fig.text(0.5, 0.05, f'{metric_name} - {phase}', ha='center', fontsize=16, fontweight="bold")
    gs.update(hspace=-0.7)
    gs.update(wspace=0.1)
    fig.subplots_adjust(top=1.1)
    plt.tight_layout()
    fname = os.path.join(figure_path, f'Ridge_Plot_{phase}.png')
    plt.savefig(fname, dpi=200)
    print(f"{fname} created.")

def plot_single_ridge(data, figure_path, phase, metric):
    """
    Plot a single ridge plot for given data, combining both seasons in one subplot.
    """
    # Create a figure with a row for each region
    plt.close("all")
    gs = grid_spec.GridSpec(len(REGIONS), 1)
    fig = plt.figure(figsize=(10, 10))

    # Calculate the upper percentile to define the x-axis range
    variable = data[metric]
    upper_percentile = np.percentile(variable,KDE_PARAMS[metric][2]) 
    
    # Set the x-axis limits based on the percentiles
    xmin, xmax = 0, upper_percentile
    
    x_d = np.linspace(xmin, xmax,KDE_PARAMS[metric][1])

    for idx, rg in enumerate(REGIONS):
        region_data = data[(data['Region'] == rg)]
        x = np.array(region_data[metric])
        
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
                    f"{phase.capitalize()}", fontweight="bold", fontsize=14, ha="center")

    ax.text((xmax - xmin) / 2, np.exp(logprob).min() - 0.025,
             metric, ha='center', fontsize=16, fontweight="bold")

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
    if metric == 'Mean Growth rate  (−1 × 10−2 s−1 day-1)':
        ax.set_xlim(xmin, -xmin)
    else:
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
    elif metric == "Mean Vorticity (−1 × 10−5 s−1)":
        y_limit = np.exp(logprob).max() + 0.3
        ymax_fraction = 0.6
    elif metric == "Mean Growth rate  (−1 × 10−2 s−1 day-1)":
        y_limit = np.exp(logprob).max() + 0.01
        ymax_fraction = 10
    ax.set_ylim(0, y_limit)
    # Calculate grid line positions
    num_lines = 10
    interval = int(round(xmax, -1)) // num_lines
    interval = max(1, interval) + 1
    # Set x-ticks based on the calculated positions
    if metric == 'Mean Growth rate  (−1 × 10−2 s−1 day-1)':
        x_ticks = list(range(int(round(xmin, -1)), int(round(xmax, -1)), int(interval * 6)))
    else:
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

def plot_ridge_phases(data, figure_path):
    # Only select data for the region "Total"
    data = data[data['Region'] == 'Total']

    # Define a grid layout with one row for each phase and one column for each metric
    plt.close("all")
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
        upper_percentile = np.percentile(variable,KDE_PARAMS[metric][2])
        if metric == 'Mean Growth rate  (−1 × 10−2 s−1 day-1)':
            xmin = np.percentile(variable, 0.5)
        elif metric == 'Mean Growth rate  (−1 × 10−2 s−1 day-1)':
            xmin = np.min(variable)
        else:
            xmin = 0
        xmax = upper_percentile
        x_d = np.linspace(xmin, xmax,KDE_PARAMS[metric][1])
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
    fname = os.path.join(figure_path, f'PDF_total_all_phases_all_metrics.png')
    plt.savefig(fname, dpi=200)
    print(f"{fname} created.")

def plot_ridge_plots(database, phases):
    for metric in METRICS:
        for phase in PHASES:   
            metric_formatted = metric_to_formatted_string(metric)
            figure_path = os.path.join('..', 'figures', 'periods_statistics', ANALYSIS_TYPE, metric_formatted)
            os.makedirs(figure_path, exist_ok=True)
            print(f"\n-----------------\nPlotting phase: {phase} for {metric}")
            data = database[database['phase'] == phase]
            regions = database['Region'].unique()
            plot_single_ridge_season(data, regions, figure_path, phase, metric) 
            plot_single_ridge(data, figure_path, phase, metric)
    figure_path = os.path.join('..', 'figures', 'periods_statistics', ANALYSIS_TYPE)
    plot_ridge_phases(database, figure_path)

def phases_statistics(database):
    # Specified order for 'Region'
    region_order = ["Total", "ARG", "LA-PLATA", "SE-BR", "SE-SAO", "AT-PEN", "WEDDELL", "SA-NAM"]
    database['Region'] = pd.Categorical(database['Region'], categories=region_order, ordered=True)
    # Expanded loop to iterate over each metric
    for metric in METRICS:
        # Group by 'Region', 'Season' and 'Phase' and compute mean and std for the metric
        grouped = database.groupby(['Region', 'Season', 'phase'])[metric].agg(['mean', 'std']).reset_index()
        # Pivot the dataframe to have 'phase' as columns
        pivot_df = grouped.pivot_table(index=['Region', 'Season'], columns='phase', values=['mean', 'std'], aggfunc='first')
        # Flatten MultiIndex columns and rename them
        pivot_df.columns = ['_'.join(col).strip() for col in pivot_df.columns.values]
        pivot_df.reset_index(inplace=True)
        # Given phases in the desired order
        phase_order = ["incipient", "intensification", "mature", "decay", 
                    "intensification 2", "mature 2", "decay 2", "residual"]
        # Create an ordered list of columns based on the desired order with mean followed by std
        ordered_columns = ["Region", "Season"]
        for phase in phase_order:
            ordered_columns.append(f"mean_{phase}")
            ordered_columns.append(f"std_{phase}")
        # Reorder the dataframe columns
        pivot_df = pivot_df[ordered_columns]
        # Create a custom order for the 'Season' column
        season_order = ["DJF", "JJA"]
        pivot_df['Season'] = pd.Categorical(pivot_df['Season'], categories=season_order, ordered=True)
        # Sort by 'Season' first and then by 'Region'
        pivot_df.sort_values(by=['Season', 'Region'], inplace=True)
        # Identify columns that contain numerical data (excluding 'Region' and 'Season' columns)
        numeric_columns = pivot_df.columns.difference(['Region', 'Season'])
        # Round the numerical columns to 1 decimal place
        pivot_df[numeric_columns] = pivot_df[numeric_columns].round(1)
        # Export results
        metric_formatted = metric_to_formatted_string(metric)
        filename = f"../periods_species_statistics/{ANALYSIS_TYPE}/statistics/stastics_{metric_formatted}.csv"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        pivot_df.to_csv(filename, index=False)

def main():
    database = get_database()
    # Do not use incipient 2, as it should not exist in the data
    phases = [phase for phase in database['phase'].unique() if phase != 'incipient 2']
    # Make plots
    plot_ridge_plots(database, phases)
    # Export statistics
    phases_statistics(database)

if __name__ == '__main__':
    main()