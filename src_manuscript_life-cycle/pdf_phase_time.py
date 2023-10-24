import os
import glob
import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as grid_spec
from sklearn.neighbors import KernelDensity
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from scipy import stats

# Constants
SECONDS_IN_AN_HOUR = 3600

import os
import glob
import colorsys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as grid_spec
from sklearn.neighbors import KernelDensity
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from scipy import stats

# Constants
SECONDS_IN_AN_HOUR = 3600

def process_csv_file(csv_file):
    """
    Process the CSV file and calculate phase durations.
    
    Args:
    - csv_file (str): Path to the CSV file
    
    Returns:
    - dict: Dictionary containing phase durations
    """
    df = pd.read_csv(csv_file, parse_dates=['start', 'end'], index_col=0)

    # Remove whenever there is only one phase
    if len(df) <= 1:
        return {}
    
    else:
        phase_durations = {
            phase: ((df.loc[phase, 'end'] - df.loc[phase, 'start']).total_seconds() / SECONDS_IN_AN_HOUR)
            for phase in df.index
        }
        return phase_durations
    
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
    }

    for phase, phase_durations in duration_data.items():
        if phase != "Season":  # Make sure we don't treat 'Season' as a phase
            for duration in phase_durations:
                data['Phase'].append(phase)
                data['Duration (hours)'].append(duration)
                data['Region'].append(region)

    return pd.DataFrame(data)


def plot_single_ridge(data, regions, figure_path, phase, label):
    """
    Plot a single ridge plot for given data, combining both seasons in one subplot.
    """
    # Create a figure with a row for each region
    gs = grid_spec.GridSpec(len(regions), 1)
    fig = plt.figure(figsize=(10, 10))

    duration = data['Duration (hours)']

    # [bandwidth, number of samples, quantile]
    kde_params = {
        "incipient": [2, 1000, 98],
        "intensification": [2, 100, 98],
        "mature": [2, 1000, 99],
        "decay": [2, 1000, 95],
        "intensification 2": [2, 70, 97],
        "mature 2": [2, 1000, 99],
        "decay 2": [2, 1000, 97],
        "residual": [2, 500, 97]
    }

    # Calculate the upper percentile to define the x-axis range
    upper_percentile = np.percentile(duration, kde_params[phase][2]) 
    
    # Set the x-axis limits based on the percentiles
    xmin, xmax = 0, upper_percentile
    
    x_d = np.linspace(xmin, xmax, kde_params[phase][1])

    red_shades = ['#9d0208', '#b30109', '#c7010a', '#d9000b', '#dc2f02', '#e14c03', '#e56904', '#ff9066']
    blue_shades = ['#023e8a', '#0251a0', '#0264b6', '#0077b6', '#0091c3', '#00abd0', '#00c5dd', '#48cae4']

    for idx, rg in enumerate(regions):
        region_data = data[(data['Region'] == rg)]
        x = np.array(region_data['Duration (hours)'])
        
        kde = KernelDensity(bandwidth=kde_params[phase][0], kernel='gaussian')
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

        if idx != len(regions)-1:                
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', labelsize=14)

        ax.text(-0.5, 0, rg, fontweight="bold", fontsize=14, ha="right")
        
        # season
        if idx == 0:
            ax.text((xmax - xmin) / 2, np.exp(logprob).max() + 0.025, 
                    f"({label}) {phase.capitalize()}", fontweight="bold", fontsize=14, ha="center")

    ax.text((xmax - xmin) / 2, np.exp(logprob).min() - 0.025,
             f'Duration (hours)', ha='center', fontsize=16, fontweight="bold")

    # Adjust the position of each row of axes to create overlap
    gs.update(hspace=-0.85)
    
    # Reduce top margin to remove excess white space
    fig.subplots_adjust(top=1.35)
    
    plt.tight_layout()

    # Save the combined figure for the current phase
    fname = os.path.join(figure_path, f'Ridge_Plot_{phase}.png')
    plt.savefig(fname, dpi=200)
    print(f"{fname} created.")

def plot_ridge_phases(data):
    # Only select data for the region "Total"
    data = data[data['Region'] == 'Total']

    phases = ['incipient', 'mature', 'mature 2', 'residual',
              'intensification',  'intensification 2', 'decay', 'decay 2']
    
    colors_phases = {'incipient': '#65a1e6',
                     'intensification': '#f7b538',
                     'intensification 2': '#dc9209',
                     'mature': '#d62828',
                     'mature 2': '#c7010a',
                     'decay': '#9aa981',
                     'decay 2': '#5d6a48',
                     'residual': 'gray'}

    duration = data['Duration (hours)']
    
    gs = grid_spec.GridSpec(len(phases), 1)
    fig = plt.figure(figsize=(12, len(phases) * 1.5))  # Adjusted figure height based on number of phases

    # Set the x-axis limits based on the percentiles
    upper_percentile = np.percentile(duration, 99) 
    xmin, xmax = 0, upper_percentile
    
    x_d = np.linspace(xmin, xmax, 1000)

    for idx, phase in enumerate(phases):
        phase_data = data[data['Phase'] == phase]
        x = np.array(phase_data['Duration (hours)'])
        
        kde = KernelDensity(bandwidth=2, kernel='gaussian')
        kde.fit(x[:, None])

        ax = fig.add_subplot(gs[idx:idx+1, 0])

        logprob = kde.score_samples(x_d[:, None])

        ax.plot(x_d, np.exp(logprob), color="#f0f0f0", lw=1, linestyle='-')
        ax.fill_between(x_d, np.exp(logprob), alpha=1, color=colors_phases[phase], linestyle='-')
        
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
        max_kde_value = np.exp(logprob).max()
        ymax_fraction = 2.8
        ymax_value = max_kde_value * ymax_fraction
        ax.axvline(x=meadian_value, ymin=0, ymax=ymax_value,
                        color='k', linestyle='-', linewidth=4, label="Median")

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax.spines[s].set_visible(False)

        if idx != len(phases)-1:                
            ax.set_xticklabels([])
        else:
            ax.tick_params(axis='x', labelsize=14)

        ax.text(0.99, 0.01, phase, fontweight="bold", fontsize=14,
                 ha="right", transform=ax.transAxes)

    ax.text((xmax - xmin) / 2, np.exp(logprob).min() - 0.025,
            'Duration (hours)', ha='center', fontsize=16, fontweight="bold")

    # Adjust the position of each row of axes to create overlap
    gs.update(hspace=-0.82)
    
    # Reduce top margin to remove excess white space
    fig.subplots_adjust(top=1.35)
    
    plt.tight_layout()

    figure_path = os.path.join('..', 'figures', 'manuscript_life-cycle')
    fname = os.path.join(figure_path, 'Ridge_Plot_total_All_Phases.png')
    plt.savefig(fname, dpi=200)
    print(f"{fname} created.")


def plot_ridge_plots(dfs, figure_path, phases, labels):
    regions = dfs['Region'].unique()
    # for phase, label in zip(phases, labels):
    #     print('\n-----------------')
    #     print(f"Plotting for phase: {phase}")
    #     data = dfs[dfs['Phase'] == phase]
    #     plot_single_ridge(data, regions, figure_path, phase, label)
    
    # phase, label = ('residual', 'H')
    # plot_single_ridge(dfs, dfs['Region'].unique(), figure_path, phase, label)

    plot_ridge_phases(dfs)

def main():
    analysis_type_to_regions = {
        'BY_RG-all': ['RG1', 'RG2', 'RG3', 'all_RG'],
        '70W-no-continental': [False, "ARG", "LA-PLATA", "SE-BR", "SE-SAO", "AT-PEN", "WEDDELL", "SA-NAM"],
        'default': ['']
    }

    analysis_type = '70W-no-continental'
    regions = analysis_type_to_regions.get(analysis_type, analysis_type_to_regions['default'])
    
    figure_path = os.path.join('..', 'figures', 'manuscript_life-cycle', 'phase_time_all_seasons')
    ensure_directory_exists(figure_path)

    phase_time_database = f"../periods_species_statistics/{analysis_type}/phase_time_all_seasons.csv"


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

    phases = ['incipient', 'intensification', 'mature', 'decay',
        'intensification 2', 'mature 2', 'decay 2', 'residual']
    
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    
    plot_ridge_plots(dfs, figure_path, phases, labels)

if __name__ == '__main__':
    main()

