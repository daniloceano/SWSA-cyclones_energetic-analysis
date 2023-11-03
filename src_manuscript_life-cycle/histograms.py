# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    histograms.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: danilocoutodsouza <danilocoutodsouza@st    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/30 19:37:18 by daniloceano       #+#    #+#              #
#    Updated: 2023/11/02 20:39:19 by danilocouto      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import seaborn as sns
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import levene, ttest_ind, mannwhitneyu, anderson, ks_2samp

# Constants
SECONDS_IN_AN_HOUR = 3600
ALPHA = 0.05  # Significance level
ANALYSIS_TYPE = '70W-no-continental'
METRICS = ['Total Distance ($10^2$ km)', 'Total Time (Hours)', 'Mean Speed (m/s)',
            'Mean Vorticity (−1 × 10−5 s−1)', 'Mean Growth rate  (−1 × 10−2 s−1 day-1)']
PHASES = ['Total', 'incipient', 'intensification', 'mature', 'decay', 'intensification 2', 'mature 2', 'decay 2', 'residual']
REGIONS = ['Total', 'ARG', 'LA-PLATA', 'SE-BR', 'SE-SAO', 'AT-PEN', 'WEDDELL', 'SA-NAM']
PLOT_LABELS = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
COLOR_PHASES = {
        'Total': 'k',
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
COLOR_REGIONS = {
        'Total': 'k',
        'ARG': '#70a0cd',
        'LA-PLATA': '#003466',
        'SE-BR': '#004f00',
        'SE-SAO': '#c47900',
        'AT-PEN': '#b2b2b2',
        'WEDDELL': '#810f7c',
        'SA-NAM': '#990002'
}

COLOR_SEASONS = {
    "DJF": "#ae2012",
    "JJA": "#1d3557"}

LABEL_MAPPING = {
        'incipient': 'Ic',
        'incipient 2': 'Ic2',
        'intensification': 'It',
        'intensification 2': 'It2',
        'mature': 'M',
        'mature 2': 'M2',
        'decay': 'D',
        'decay 2': 'D2',
    }

QUANTILE_VALUES = {
        'Total Distance ($10^2$ km)': 0.9,
        'Total Time (Hours)': 0.95,
        'Mean Speed (m/s)': 0.999,
        'Mean Growth rate  (-1 x 10-2 s-1 day-1)': 1,
        'Mean Vorticity (−1 × 10−5 s−1)': 0.98
    }

def get_database():
    """Reads and aggregates data from all database files."""
    files = f"../periods_species_statistics/{ANALYSIS_TYPE}/periods_database/periods_database_*.csv"
    data_frames = []
    for file in glob.glob(files):
        data_frames.append(pd.read_csv(file, index_col=0))
    database = pd.concat(data_frames, ignore_index=True)
    # Remove rows where all columns are NaN
    database = database.dropna(how='all')
    # Remove rows where "Mean Speed (m/s)" is NaN 
    # (so we won't compute statistics for the first time steps)
    database = database.dropna(subset=['Mean Speed (m/s)',
                                       'Mean Growth rate  (−1 × 10−2 s−1 day-1)']) 
    database['Total Distance ($10^2$ km)'] = database['Total Distance (km)'] / 100     
    return database

def metric_to_formatted_string(metric):
    """Converts metric name to a shorter formatted string."""
    mapping = {
        'Total Time (Hours)': 'total_time',
        'Total Distance ($10^2$ km)': 'total_distance',
        'Mean Speed (m/s)': 'mean_speed',
        'Mean Growth rate  (−1 × 10−2 s−1 day-1)': 'mean_intensity',
        'Mean Growth rate  (−1 × 10−2 s−1 day-1)': 'mean_growth'
    }
    return mapping.get(metric, '')

def plot_histogram_for_metric(ax, jja_metric_data, djf_metric_data, metric):
    """Plots histogram and KDE for a specific metric on a given axis."""
    # Compute the upper quantile value across both JJA and DJF data for each metric
    
    upper_bound = max(jja_metric_data.quantile(QUANTILE_VALUES[metric]),
                      djf_metric_data.quantile(QUANTILE_VALUES[metric]))

    # Compute combined bin edges for both datasets
    combined_data = np.concatenate([jja_metric_data, djf_metric_data])
    bins = np.histogram_bin_edges(combined_data, bins=50)

    # Plot histograms using seaborn for better aesthetics with consistent bins
    sns.histplot(jja_metric_data, ax=ax, bins=bins, color=COLOR_SEASONS['JJA'], kde=True, label='JJA')
    sns.histplot(djf_metric_data, ax=ax, bins=bins, color=COLOR_SEASONS['DJF'], kde=True, label='DJF')

    # Add mean lines
    ax.axvline(jja_metric_data.mean(), color=COLOR_SEASONS['JJA'], linestyle='--', linewidth=2)
    ax.axvline(djf_metric_data.mean(), color=COLOR_SEASONS['DJF'], linestyle='--', linewidth=2)

    ax.set_xlim(0, upper_bound)

def apply_statistical_tests(ax, jja_metric_data, djf_metric_data):
    """Applies statistical tests and annotates the plot with the results."""
    # Anderson-Darling test for normality
    result_djf = anderson(djf_metric_data)
    result_jja = anderson(jja_metric_data)

    # Use the 5% significance level, which is typically the third value in the critical_values list
    is_normal_djf = result_djf.statistic < result_djf.critical_values[2]
    is_normal_jja = result_jja.statistic < result_jja.critical_values[2]

    # Statistical significance test based on homogeneity of variance
    _, p_value_levene = levene(jja_metric_data, djf_metric_data)

    # If data is normally distributed and has homogenous variance, we use t-test, otherwise Mann-Whitney U test
    if is_normal_jja and is_normal_djf and p_value_levene > ALPHA:
        _, p_value_test = ttest_ind(jja_metric_data, djf_metric_data)
    else:
        _, p_value_test = mannwhitneyu(jja_metric_data, djf_metric_data)

    # Two-sample Kolmogorov-Smirnov test
    _, p_value_ks = ks_2samp(jja_metric_data, djf_metric_data)

    significance_str = ""
    if p_value_test < 0.05: significance_str = "T/MW: *"
    if p_value_test < 0.01: significance_str += "*"
    if p_value_test < 0.001: significance_str += "*"

    ks_significance_str = ""
    if p_value_ks < 0.05: ks_significance_str = "KS: *"
    if p_value_ks < 0.01: ks_significance_str += "*"
    if p_value_ks < 0.001: ks_significance_str += "*"

    # Annotate the plot with mean values, t-test/Mann-Whitney significance, and KS significance
    ax.text(0.99, 0.94, f'{round(djf_metric_data.mean(), 1)}',
            c=COLOR_SEASONS['DJF'], fontsize=11, transform=ax.transAxes, ha='right', va='top') 
    ax.text(0.99, 0.71, f'{round(jja_metric_data.mean(), 1)}',
            c=COLOR_SEASONS['JJA'], fontsize=11, transform=ax.transAxes, ha='right', va='top')
    ax.text(0.99, 0.45, f'{significance_str}',
            c='k', fontsize=9, transform=ax.transAxes, ha='right', va='top')
    ax.text(0.99, 0.25, f'{ks_significance_str}',
            c='k', fontsize=9, transform=ax.transAxes, ha='right', va='top')

def plot_histograms_with_kde(jja_data, djf_data):
    """Main function to plot histograms with KDE for all metrics and regions."""
    
    for phase in PHASES:  # Looping over each phase
        
        # Filtering data for the current phase
        jja_data_phase = jja_data[jja_data['phase'] == phase]
        djf_data_phase = djf_data[djf_data['phase'] == phase]
        
        # Number of rows (regions) and columns (metrics)
        num_rows = len(REGIONS)
        num_cols = len(METRICS)

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 10), tight_layout=True)

        for i, region in enumerate(REGIONS):
            for j, metric in enumerate(METRICS):
                # Filter data by region for the current phase
                jja_region_data = jja_data_phase[jja_data_phase['Region'] == region]
                djf_region_data = djf_data_phase[djf_data_phase['Region'] == region]

                # Extract the metric values for JJA and DJF
                jja_metric_data = jja_region_data[metric].dropna()
                djf_metric_data = djf_region_data[metric].dropna()

                ax = axes[i, j]

                # Plotting histogram and KDE
                plot_histogram_for_metric(ax, jja_metric_data, djf_metric_data, metric)

                # Apply statistical tests and annotate the plot
                apply_statistical_tests(ax, jja_metric_data, djf_metric_data)

                # Set titles, labels
                ax.xaxis.set_tick_params(labelsize=12)
                ax.set_yticklabels([])
                ax.set_xlabel("")
                if i == 0:
                    ax.set_title(metric, fontsize=14)
                if j == 0:
                    ax.set_ylabel(region, fontsize=14)
                elif j == 2:
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.tick_right()
                    ax.set_ylabel('Count', fontsize=12)
                else:
                    ax.set_ylabel("")

        # Save the plot with a unique filename based on the current phase
        plt.savefig(f"../figures/manuscript_life-cycle/histograms/histograms_statistics_seasonal_{phase}.png", dpi=300)
        plt.close()  # Close the figure to free up memory
        
def plot_histograms_for_total_season(total_data):
    """Plot histograms for 'Total' season with each subplot being a different phase."""
    
    # Number of rows (phases) and columns (metrics)
    num_rows = len(PHASES)
    num_cols = len(METRICS)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10), tight_layout=True)

    for i, phase in enumerate(PHASES):
        for j, metric in enumerate(METRICS):
            ax = axes[i, j]
            # Extract data for the current phase
            phase_data = total_data[total_data['phase'] == phase]
            
            # Determine upper bound for the current metric
            upper_bound = phase_data[metric].quantile(0.95)

            # Loop through each region and plot data on the same subplot
            for region in REGIONS:
                region_data = phase_data[phase_data['Region'] == region]
                metric_data = region_data[metric].dropna()

                # Plotting only KDE
                sns.kdeplot(metric_data, ax=ax, label=region,
                            color=COLOR_REGIONS[region], linewidth=2)

            ax.set_xlim(0, upper_bound)

            # Set titles, labels
            ax.xaxis.set_tick_params(labelsize=12)
            ax.set_yticklabels([])
            ax.set_xlabel("")
            if i == 0:
                ax.set_title(f"{PLOT_LABELS[j]} {metric}", fontsize=14, loc='left')
            if j == 0:
                label = LABEL_MAPPING.get(phase, phase)
                ax.set_ylabel(label, fontsize=14)
            elif j == 2:
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set_ylabel('Count', fontsize=12)
            else:
                ax.set_ylabel("")

    # Placing a single legend outside the plotting area on the right
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='center left', fontsize=12, title="Regions", 
               title_fontsize=14, bbox_to_anchor=(1.15, 6))

    # Adjust layout to make space for the legend
    fig.tight_layout()
    plt.subplots_adjust(right=0.8, hspace=0.5)

    # Save the plot with a unique filename for the "Total" season
    plt.savefig(f"../figures/manuscript_life-cycle/histograms/histograms_total_season.png", dpi=300)


def compare_phases_by_region(data, n_bins=10):
    """Compare each phase for each region by plotting KDE for given metrics.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing the values to plot.
    - n_bins (int): Number of bins for the histograms (unused but retained for possible future use).
    """
    # QUANTILE_VALUES = {
    #     'Total Distance ($10^2$ km)': 0.85,
    #     'Total Time (Hours)': 0.9,
    #     'Mean Speed (m/s)': 0.99
    # }

    # Create subplots
    for metric in METRICS:

        fig, axes = plt.subplots(2, 4, figsize=(12, 9))

        # Determine the global max x-axis value for the current metric across all regions and phases
        global_max_value = round(data[metric].quantile(QUANTILE_VALUES[metric]), -1)

        idx = 0
        for row in range(2):
            for column in range(4):
                ax = axes[row, column]
                region = REGIONS[idx]
                
                # Variables to help position the text neatly
                y_position = 0.99
                y_offset = 0.05
                
                # Collecting maximum values among all phases for the given metric in the given region
                max_values = []
                for phase in PHASES[:-1]:
                    subset = data[(data['Region'] == region) & (data['phase'] == phase)]
                    mean_value = subset[metric].mean()
                    std_value = subset[metric].std()
                    
                    text_str = f"{mean_value:.2f} ± {std_value:.2f}"
                    ax.text(0.95, y_position, text_str, transform=ax.transAxes,
                            fontsize=10, verticalalignment='top', horizontalalignment='right',
                            color=COLOR_PHASES[phase], weight='bold')
                    
                    y_position -= y_offset

                    max_values.append(
                        subset[metric].quantile(QUANTILE_VALUES[metric])
                        )
                    if "2" in phase:
                        ls = '--'
                    else:
                        ls = '-'
                    sns.kdeplot(subset[metric], ax=ax, label=phase,
                                color=COLOR_PHASES[phase], linestyle=ls,
                                linewidth=3, alpha=.8)
                
                # Set the x-axis limit and ticks
                ax.set_xlim(0, global_max_value)
                ax.set_xticks(np.linspace(0, global_max_value, 5))
                
                # Set titles, labels
                ax.xaxis.set_tick_params(labelsize=12)
                ax.set_yticklabels([])
                ax.set_xlabel("")
                ax.set_title(f"{PLOT_LABELS[idx]} {region}", fontsize=14, loc='left',
                            color="k", weight='bold')                    
                if column == 0:
                    ax.set_ylabel('Density', fontsize=14)
                else:
                    ax.set_ylabel("")
                if row == 1:
                    ax.set_xlabel(metric, fontsize=14)

                if row == 1 and column == 3:
                    # Placing a single legend outside the plotting area on the right
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles, labels, loc='center left',
                              fontsize=12, title="Regions", 
                              title_fontsize=14, bbox_to_anchor=(1.1, 1))
                idx += 1

        # Adjust layout to make space for the legend
        plt.subplots_adjust(right=0.8)
        fig.tight_layout(h_pad=-5)

        # Save the plot with a unique filename for the "Total" season
        metric_string = metric_to_formatted_string(metric)
        plt.savefig(f"../figures/manuscript_life-cycle/histograms_phases_regions_{metric_string}.png", dpi=300)

def main():
    database = get_database()
    total_data = database[database['Season'] == 'Total']
    
    jja_data = database[database['Season'] == 'JJA']
    djf_data = database[database['Season'] == 'DJF']
    plot_histograms_with_kde(jja_data, djf_data)
    
    plot_histograms_for_total_season(total_data)

    compare_phases_by_region(database)

if __name__ == '__main__':
    main()
