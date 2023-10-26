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
alpha = 0.05  # Significance level

def process_csv_file(csv_file):
    """
    Process the CSV file and calculate phase durations.
    
    Args:
    - csv_file (str): Path to the CSV file
    
    Returns:
    - dict: Dictionary containing phase durations and seasons
    """
    try:
        df = pd.read_csv(csv_file, parse_dates=['start', 'end'], index_col=0)
        
        # Check if mature phase is present
        if 'mature' in df.index:
            mature_month = df.at['mature', 'start'].month
            
            # Deduce season from the 'start' date of mature phase
            if mature_month in  [12, 1, 2, 6, 7, 8]:
                season = 'DJF' if mature_month in [12, 1, 2] else 'JJA'
                
                phase_durations = {
                    phase: ((df.at[phase, 'end'] - df.at[phase, 'start']).total_seconds() / SECONDS_IN_AN_HOUR)
                    for phase in df.index
                }
                return {**phase_durations, "Season": season}
        
        return {}
        
    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")
        return {}
    
def process_phase_data_parallel(data_path):
    """
    Process CSV data in parallel.
    
    Args:
    - data_path (str): Path to the data directory
    
    Returns:
    - dict: Aggregated durations for each phase
    """
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
    durations = defaultdict(list)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_csv_file, csv_files), total=len(csv_files), desc='Processing Files'))

    for result in results:
        for phase, duration in result.items():
            durations[phase].append(duration)

    return durations

def ensure_directory_exists(path):
    """
    Ensure a directory exists; if not, create it.
    
    Args:
    - path (str): Directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def create_seaborn_dataframe(duration_data, region):
    """
    Create a dataframe suitable for Seaborn plotting.
    
    Args:
    - duration_data (dict): Data containing phase durations
    - region (str): Region identifier
    
    Returns:
    - DataFrame: Processed dataframe
    """
    data = {
        'Phase': [],
        'Duration (hours)': [],
        'Region': [],
        'Season': []
    }

    for phase, phase_durations in duration_data.items():
        if phase != "Season":  # Make sure we don't treat 'Season' as a phase
            for duration, season in zip(phase_durations, duration_data["Season"]):
                if season:  # Check if seasons is not None
                    data['Phase'].append(phase)
                    data['Duration (hours)'].append(duration)
                    data['Region'].append(region)
                    data['Season'].append(season)

    return pd.DataFrame(data)

def compare_djf_jja(data_for_region):
    # Extract DJF and JJA data
    djf_data = data_for_region[data_for_region['Season'] == 'DJF']['Duration (hours)'].dropna().values
    jja_data = data_for_region[data_for_region['Season'] =='JJA']['Duration (hours)'].dropna().values

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

def plot_single_ridge(data, regions, figure_path, phase):
    """
    Plot a single ridge plot for given data, combining both seasons in one subplot.
    """
    # Create a figure with a row for each region
    gs = grid_spec.GridSpec(len(regions), 2)
    fig = plt.figure(figsize=(12, 9))

    duration = data['Duration (hours)']

    # [bandwidth, number of samples, quantile]
    kde_params = {
        "incipient": [1, 1000, 99],
        "intensification": [2, 100, 98],
        "mature": [1, 1000, 99.9],
        "decay": [2, 1000, 95],
        "intensification 2": [3, 70, 99],
        "mature 2": [1, 1000, 99.9],
        "decay 2": [2, 1000, 97],
        "residual": [2, 500, 99]
    }

    # Calculate the upper percentile to define the x-axis range
    upper_percentile = np.percentile(duration, kde_params[phase][2]) 
    
    # Set the x-axis limits based on the percentiles
    xmin, xmax = 0, upper_percentile
    
    x_d = np.linspace(xmin, xmax, kde_params[phase][1])

    red_shades = ['#9d0208', '#b30109', '#c7010a', '#d9000b', '#dc2f02', '#e14c03', '#e56904', '#ff9066']
    blue_shades = ['#023e8a', '#0251a0', '#0264b6', '#0077b6', '#0091c3', '#00abd0', '#00c5dd', '#48cae4']

    for idx, rg in enumerate(regions):
        color_djf = red_shades[idx]
        color_jja = blue_shades[idx]

        # Get data for the region
        data_for_region = data[data['Region'] == rg]

        # Compare DJF and JJA data for the region
        test_used, p_value = compare_djf_jja(data_for_region)

        # Determine statistical significance
        if p_value <= alpha:
            significance = "statistically significant"
        else:
            significance = "not statistically significant"
        
        for season, col in zip(['DJF', 'JJA'], [0, 1]):                
            season_data = data[(data['Region'] == rg) & (data['Season'] == season)]
            x = np.array(season_data['Duration (hours)'])
            
            kde = KernelDensity(bandwidth=kde_params[phase][0], kernel='gaussian')
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
                ax.text(-0.5, 0, f"({mean_value:.1f} h)", fontsize=10, ha="right")
            else:
                ax.text(np.max(x_d)*1.15, 0, f"({mean_value:.1f} h)", fontsize=10, ha="right")
                if p_value <= alpha:
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

            print(f"Mean duration for {rg} in {season}: {mean_value:.2f} hours")
            print(f"Max duration for {rg} in {season}: {season_data['Duration (hours)'].max()} hours")
            print(f"For region {rg} using {test_used}, the difference is {significance} (p-value: {p_value:.4f})")

    fig.text(0.5, 0.05, f'Duration (hours) - {phase}', ha='center', fontsize=16, fontweight="bold")

    # Adjust the position of each row of axes to create overlap
    gs.update(hspace=-0.7)
    gs.update(wspace=0.1)
    
    # Reduce top margin to remove excess white space
    fig.subplots_adjust(top=1.1)
    
    plt.tight_layout()

    # Save the combined figure for the current phase
    fname = os.path.join(figure_path, f'Ridge_Plot_{phase}.png')
    plt.savefig(fname, dpi=200)
    print(f"{fname} created.")

def plot_ridge_plots(dfs, figure_path, phases):
    for phase in phases:
        
        print('\n-----------------')
        print(f"Plotting phase: {phase}")
        # Filter the data for the current phase
        data = dfs[dfs['Phase'] == phase]
        regions = dfs['Region'].unique()
            
        plot_single_ridge(data, regions, figure_path, phase)

    # phase = "residual"
    # data = dfs[dfs['Phase'] == phase]
    # plot_single_ridge(data, regions, figure_path, phase)

def phases_statistics(dfs, analysis_type):
    # Specified order for 'Region'
    region_order = ["Total", "ARG", "LA-PLATA", "SE-BR", "SE-SAO", "AT-PEN", "WEDDELL", "SA-NAM"]
    dfs['Region'] = pd.Categorical(dfs['Region'], categories=region_order, ordered=True)

    # Group by 'Region', 'Season' and 'Phase' and compute mean and std
    grouped = dfs.groupby(['Region', 'Season', 'Phase'])['Duration (hours)'].agg(['mean', 'std']).reset_index()

    # Pivot the dataframe to have 'Phase' as columns
    pivot_df = grouped.pivot_table(index=['Region', 'Season'], columns='Phase', values=['mean', 'std'], aggfunc='first')

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
    
    pivot_df.to_csv(f"../periods_species_statistics/{analysis_type}/phase_time/phase_time_stastics_djf_jja.csv", index=False)

def main():
    analysis_type_to_regions = {
        'BY_RG-all': ['RG1', 'RG2', 'RG3', 'all_RG'],
        '70W-no-continental': [False, "ARG", "LA-PLATA", "SE-BR", "SE-SAO", "AT-PEN", "WEDDELL", "SA-NAM"],
        'default': ['']
    }

    analysis_type = '70W-no-continental'
    regions = analysis_type_to_regions.get(analysis_type, analysis_type_to_regions['default'])
    
    figure_path = os.path.join('..', 'figures', 'periods_statistics', analysis_type, 'phase_time')
    ensure_directory_exists(figure_path)

    phase_time_database = f"../periods_species_statistics/{analysis_type}/phase_time_database_djf_jja.csv"

    try:
        dfs = pd.read_csv(phase_time_database)

    except:
        print(f"{phase_time_database} not found, creating it...")
        data_frames = []
        for region in regions:
            region_str = f'_{region}' if region else ''
            data_path = os.path.join('..', 'periods-energetics', analysis_type + region_str)
            
            duration_data = process_phase_data_parallel(data_path)
            print(f'Processed data for region: {region}')

            df = create_seaborn_dataframe(duration_data, region if region else 'Total')
            data_frames.append(df)

        dfs = pd.concat(data_frames)
        dfs.to_csv(phase_time_database)
        print(f"{phase_time_database} created.")

    phases = [phase for phase in dfs['Phase'].unique() if phase not in ['Season', 'incipient 2']]
    plot_ridge_plots(dfs, figure_path, phases)
    phases_statistics(dfs, analysis_type)

if __name__ == '__main__':
    main()