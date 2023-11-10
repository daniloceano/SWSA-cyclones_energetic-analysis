# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    histograms.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <daniloceano@student.42.fr>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/30 19:37:18 by daniloceano       #+#    #+#              #
#    Updated: 2023/11/09 19:16:27 by daniloceano      ###   ########.fr        #
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
METRICS = ['Total Time (h)', 'Maximum Distance (km)', 'Mean Speed (m/s)',
            'Mean Vorticity (−1 × 10−5 s−1)', 'Mean Growth Rate (10^−5 s^−1 day-1)']
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

METRICS_LATEX_MAPPING = {
    'Total Time (h)': r'Total Time [h]',
    'Maximum Distance (km)': r'Maximum Distance [$10^2$ km]',
    'Mean Speed (m/s)': r'Mean Speed [m$\cdot$s$^{-1}$]',
    'Mean Vorticity (−1 × 10−5 s−1)': r'Mean Vorticity [$-1 \times 10^{-5}$ s$^{-1}$]',
    'Mean Growth Rate (10^−5 s^−1 day-1)': r'Mean Growth rate [$10^{-5}$ s$^{-1}$ day$^{-1}$]'
}

QUANTILE_VALUES = {
        'Maximum Distance (km)': 0.9,
        'Total Time (h)': 0.95,
        'Mean Speed (m/s)': 0.999,
        'Mean Growth Rate (10^−5 s^−1 day-1)': 0.99,
        'Mean Vorticity (−1 × 10−5 s−1)': 0.99
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
                                       'Mean Growth Rate (10^−5 s^−1 day-1)']) 
    database['Maximum Distance (km)'] = database['Maximum Distance (km)'] / 100     
    return database

def metric_to_formatted_string(metric):
    """Converts metric name to a shorter formatted string."""
    mapping = {
        'Total Time (h)': 'total_time',
        'Maximum Distance (km)': 'total_distance',
        'Mean Speed (m/s)': 'mean_speed',
        'Mean Vorticity (−1 × 10−5 s−1)': 'mean_intensity',
        'Mean Growth Rate (10^−5 s^−1 day-1)': 'mean_growth'
    }
    return mapping.get(metric, '')

def plot_histogram_for_metric(ax, jja_metric_data, djf_metric_data, metric):
    """Plots histogram and KDE for a specific metric on a given axis."""
    # Compute the upper quantile value across both JJA and DJF data for each metric
    
    if metric == 'Mean Growth Rate (10^−5 s^−1 day-1)':
        upper_bound = max(jja_metric_data.quantile(QUANTILE_VALUES[metric]),
                        djf_metric_data.quantile(QUANTILE_VALUES[metric]))
        lower_bound = -upper_bound
    else:
        upper_bound = max(jja_metric_data.max(), djf_metric_data.max())
        lower_bound = min(jja_metric_data.min(), djf_metric_data.min())

    # Compute combined bin edges for both datasets
    combined_data = np.concatenate([jja_metric_data, djf_metric_data])
    bins = np.histogram_bin_edges(combined_data, bins=50)

    # Plot histograms using seaborn for better aesthetics with consistent bins
    sns.histplot(jja_metric_data, ax=ax, bins=bins, color=COLOR_SEASONS['JJA'], kde=True, label='JJA')
    sns.histplot(djf_metric_data, ax=ax, bins=bins, color=COLOR_SEASONS['DJF'], kde=True, label='DJF')

    # Add mean lines
    ax.axvline(jja_metric_data.mean(), color=COLOR_SEASONS['JJA'], linestyle='--', linewidth=2)
    ax.axvline(djf_metric_data.mean(), color=COLOR_SEASONS['DJF'], linestyle='--', linewidth=2)

    ax.set_xlim(lower_bound, upper_bound)

def apply_statistical_tests(ax, jja_metric_data, djf_metric_data, metric):
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

    if metric == 'Mean Growth Rate (10^−5 s^−1 day-1)':
        round_factor = 3
    else:
        round_factor = 1
    # Annotate the plot with mean values, t-test/Mann-Whitney significance, and KS significance
    ax.text(0.99, 0.94, f'{round(djf_metric_data.mean(), round_factor)}',
            c=COLOR_SEASONS['DJF'], fontsize=11, transform=ax.transAxes, ha='right', va='top') 
    ax.text(0.99, 0.71, f'{round(jja_metric_data.mean(), round_factor)}',
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
                apply_statistical_tests(ax, jja_metric_data, djf_metric_data, metric)

                # Set titles, labels
                ax.xaxis.set_tick_params(labelsize=12)
                ax.set_yticklabels([])
                ax.set_xlabel("")
                if i == 0:
                    ax.set_title(metric_to_formatted_string(metric), fontsize=14)
                if j == 0:
                    ax.set_ylabel(region, fontsize=14)
                elif j == 2:
                    ax.yaxis.set_label_position("right")
                    ax.yaxis.tick_right()
                    ax.set_ylabel('Count', fontsize=12)
                else:
                    ax.set_ylabel("")

        # Save the plot with a unique filename based on the current phase
        fname = f"../figures/manuscript_life-cycle/histograms/histograms_statistics_seasonal_{phase}.png"
        plt.savefig(fname, dpi=300)
        plt.close()  # Close the figure to free up memory
        print(f"{fname} created.")

def compare_phases_by_region(data, n_bins=10):
    """Compare each phase for each region by plotting KDE for given metrics.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing the values to plot.
    - n_bins (int): Number of bins for the histograms (unused but retained for possible future use).
    """

    # Create subplots
    for metric in METRICS:

        fig, axes = plt.subplots(2, 4, figsize=(12, 9))

        # Determine the global max x-axis value for the current metric across all regions and phases
        global_max_value = data[metric].quantile(QUANTILE_VALUES[metric])

        idx = 0
        for row in range(2):
            for column in range(4):
                ax = axes[row, column]
                region = REGIONS[idx]
                
                # Variables to help position the text neatly
                y_position = 0.99
                x_position = 0.95
                y_offset = 0.05

                if metric == 'Mean Growth Rate (10^−5 s^−1 day-1)':
                    x_position = 0.3
                
                # Collecting maximum values among all phases for the given metric in the given region
                max_values = []
                for phase in PHASES[:-1]:
                    subset = data[(data['Region'] == region) & (data['phase'] == phase)]
                    mean_value = subset[metric].mean()
                    std_value = subset[metric].std()
                    
                    if metric == 'Mean Growth Rate (10^−5 s^−1 day-1)':
                        text_str = f"{mean_value:.3f} ± {std_value:.3f}"
                    else:
                        text_str = f"{mean_value:.2f} ± {std_value:.2f}"
                    ax.text(x_position, y_position, text_str, transform=ax.transAxes,
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
                if metric != 'Mean Growth Rate (10^−5 s^−1 day-1)':
                    ax.set_xlim(0, global_max_value)
                    ax.set_xticks(np.linspace(0, global_max_value, 5))
                else:
                    ax.set_xlim(-global_max_value, global_max_value)
                    ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.7)
                    ax.set_xticks(np.linspace(-global_max_value*1.1, global_max_value, 3))
                
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
                    metric_name, metric_unit = METRICS_LATEX_MAPPING[metric].split(' [')
                    ax.set_xlabel(f"{metric_name}\n[{metric_unit}", fontsize=14)

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
        fname = f"../figures/manuscript_life-cycle/histograms/histograms_phases_regions_{metric_string}.png"
        plt.savefig(fname, dpi=300)
        print(f"{fname} created.")
        plt.close()

def compare_phases_for_total_region(data):
    """Compare each phase for the 'Total' region by plotting KDE for each metric.
    
    Parameters:
    - data (pd.DataFrame): The dataset containing the values to plot.
    """

    # Filter data for the 'Total' region
    data_total_region = data[data['Region'] == 'Total']

    # Determine number of rows and columns based on the number of metrics
    num_metrics = len(METRICS)
    num_rows = (num_metrics + 1) // 2  # Adjusted for 2 columns layout
    num_cols = 2

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5 * num_rows))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i, metric in enumerate(METRICS):
        ax = axes[i]

        # Variables to help position the text neatly
        y_position = 0.99  # Starting y position for text annotations
        x_position = 0.98  # x position for text annotations
        y_offset = 0.07  # Spacing between text annotations

        # Collecting maximum values among all phases for the given metric in the 'Total' region
        max_values = []
        for phase in PHASES[:-1]:
            subset = data_total_region[data_total_region['phase'] == phase]
            mean_value = subset[metric].mean()
            std_value = subset[metric].std()
            text_str = f"{mean_value:.2f} ± {std_value:.2f}"
            if metric == "Mean Growth Rate (10^−5 s^−1 day-1)":
                text_str = f"{mean_value:.3f} ± {std_value:.3f}"
            

            # Annotate the mean and std deviation on the plot
            ax.text(x_position, y_position, text_str, transform=ax.transAxes,
                    horizontalalignment='right', verticalalignment='top',
                    color=COLOR_PHASES[phase], fontsize=12, weight='bold')
            y_position -= y_offset  # Update the y position for the next annotation

            # Plot the KDE for the phase
            if "2" in phase:
                ls = '--'
            else:
                ls = '-'
            sns.kdeplot(subset[metric], ax=ax, label=phase, linestyle=ls,
                        color=COLOR_PHASES[phase], linewidth=5, alpha=.8)
            max_values.append(subset[metric].quantile(QUANTILE_VALUES[metric]))

        ax.text(x_position, y_position, PLOT_LABELS[i], transform=ax.transAxes,
                    horizontalalignment='right', verticalalignment='top',
                    color="k", fontsize=20, weight='bold')

        # Set the x-axis limit
        lower_bound = 0
        if any(term in metric for term in ["Distance", "Time"]):
            upper_bound = np.quantile(data_total_region[metric], 0.95)
        elif metric == 'Mean Growth Rate (10^−5 s^−1 day-1)':
            upper_bound = data_total_region[metric].quantile(QUANTILE_VALUES[metric])
            lower_bound = -upper_bound
        else:
            upper_bound = max(max_values)        
        ax.set_xlim(lower_bound, upper_bound)

        # Set title and labels
        ax.set_ylabel('Density', fontsize=14)
        ax.yaxis.labelpad = 10
        ax.set_xlabel(METRICS_LATEX_MAPPING[metric], fontsize=16)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.xaxis.set_tick_params(labelsize=14)

        if metric == 'Mean Growth Rate (10^−5 s^−1 day-1)':
            ax.axvline(0, color='gray', linestyle='-', linewidth=2, alpha=0.7)

        # Place the legend
        if i == num_metrics - 1:
            # Adjust the bbox_to_anchor parameters and increase the bbox size
            handles, labels = ax.get_legend_handles_labels()
            legend = ax.legend(handles, labels, loc='center left', fontsize=18, 
                            title="Phases", title_fontsize=20, bbox_to_anchor=(1.4, 0.5), frameon=False)


        # Hide unused subplots if METRICS is an odd number
        if num_metrics % 2 != 0 and i == num_metrics - 1:
            fig.delaxes(axes[i + 1])

    # Adjust the layout
    fig.tight_layout(w_pad=-25)

    # Save the plot with a unique filename for the 'Total' region
    fname = f"../figures/manuscript_life-cycle/histograms_phases_total_region.png"
    plt.savefig(fname, dpi=300)
    plt.close()
    print(f"{fname} created.")

def compare_seasonal_phases(data):
    # Dictionary to store the summary statistics and p-values for the tests
    grouped_stats = {}
    comparison_p_values = {}  # Dictionary to store p-values for the tests

    # Iterate over each phase
    for phase in PHASES[:-1]:  # Assuming the last phase is not needed
        # Filter data for the current phase
        phase_data = data[data['phase'] == phase]
        
        # Calculate mean and std for each season and metric
        grouped_stats[phase] = phase_data.groupby('Season').agg(['mean', 'std'])
        
        # Perform statistical tests for DJF and JJA for each metric
        for metric in METRICS:
            jja_values = phase_data[phase_data['Season'] == 'JJA'][metric]
            djf_values = phase_data[phase_data['Season'] == 'DJF'][metric]

            # Ensure there are enough data points to perform the test
            if len(jja_values) > 1 and len(djf_values) > 1:
                # Perform the t-test or Mann-Whitney U test depending on data distribution
                if stats.shapiro(jja_values).pvalue > ALPHA and stats.shapiro(djf_values).pvalue > ALPHA:
                    _, p_value = stats.ttest_ind(jja_values, djf_values, equal_var=False)
                else:
                    _, p_value = stats.mannwhitneyu(jja_values, djf_values)
                # Store the p-value in the dictionary with a tuple key (phase, metric)
                comparison_p_values[(phase, metric)] = p_value
            else:
                # If there are not enough data points, fill the p-value with NaN
                comparison_p_values[(phase, metric)] = np.nan

    # Create formatted tables for each season
    total_table = format_summary_table(grouped_stats, 'Total', comparison_p_values)
    djf_table = format_summary_table(grouped_stats, 'DJF', comparison_p_values)
    jja_table = format_summary_table(grouped_stats, 'JJA', comparison_p_values)

    return total_table, djf_table, jja_table

def format_summary_table(grouped_stats, season, comparison_p_values):
    # Create the metrics table with phases and their corresponding metrics
    metrics_table = pd.DataFrame({'Phase': PHASES[:-1]})
    for metric in METRICS:
        if metric == 'Mean Growth Rate (10^−5 s^−1 day-1)':
            metrics_table[metric] = [
                f"{grouped_stats[phase].loc[season, (metric, 'mean')]:.3f} ± {grouped_stats[phase].loc[season, (metric, 'std')]:.3f}"
                + ('*' if season != 'Total' and comparison_p_values.get((phase, metric)) < ALPHA else '')
                for phase in PHASES[:-1]
            ]
        else:
            metrics_table[metric] = [
                f"{grouped_stats[phase].loc[season, (metric, 'mean')]:.2f} ± {grouped_stats[phase].loc[season, (metric, 'std')]:.2f}"
                + ('*' if season != 'Total' and comparison_p_values.get((phase, metric)) < ALPHA else '')
                for phase in PHASES[:-1]
            ]
    return metrics_table

def create_statistics_table(database):
    total_region_data = database[database['Region'] == 'Total']
    total_table, djf_table, jja_table = compare_seasonal_phases(total_region_data)

    results_path = f'../periods_species_statistics/{ANALYSIS_TYPE}/statistics/'

    # Save the tables to CSV files
    for table, table_name in [(total_table, 'total'), (djf_table, 'djf'), (jja_table, 'jja')]:
        table.to_csv(f'{results_path}/statistics_summary_{table_name}.csv', index=False)

def main():
    database = get_database()
    total_season_data = database[database['Season'] == 'Total']
    
    # jja_data = database[database['Season'] == 'JJA']
    # djf_data = database[database['Season'] == 'DJF']
    # plot_histograms_with_kde(jja_data, djf_data)
    
    # compare_phases_by_region(database)

    compare_phases_for_total_region(total_season_data)

    # create_statistics_table(database)

if __name__ == '__main__':
    main()
