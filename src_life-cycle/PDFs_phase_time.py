import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as grid_spec
from sklearn.neighbors import KernelDensity
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

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
    try:
        df = pd.read_csv(csv_file, parse_dates=['start', 'end'], index_col=0)
        phase_durations = {
            phase: ((df.loc[phase, 'end'] - df.loc[phase, 'start']).total_seconds() / SECONDS_IN_AN_HOUR)
            for phase in df.index
        }
        return phase_durations
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
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))[:200]
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
        'Region': region
    }

    for phase, phase_durations in duration_data.items():
        data['Phase'].extend([phase] * len(phase_durations))
        data['Duration (hours)'].extend(phase_durations)

    return pd.DataFrame(data)

def plot_single_ridge(data, regions, phase, colors, fig, gs, figure_path):
    """
    Plot a single ridge plot for given data.
    
    Args:
    - data (DataFrame): Data to plot
    - regions (list): List of regions
    - phase (str): Phase identifier
    - colors (list): List of colors for the plot
    - fig (Figure): Matplotlib figure object
    - gs (GridSpec): Matplotlib GridSpec object
    - figure_path (str): Path to save the figure
    """
    ax_objs = []
    duration = data['Duration (hours)']
    xmin, xmax = duration.min(), duration.max()
    x_d = np.linspace(xmin, xmax, 1000)
    
    for idx, rg in enumerate(regions):
        x = np.array(data[data['Region'] == rg]['Duration (hours)'])
        
        kde = KernelDensity(bandwidth=1, kernel='gaussian')
        kde.fit(x[:, None])

        logprob = kde.score_samples(x_d[:, None])
        ax_objs.append(fig.add_subplot(gs[idx:idx+1, 0:]))

        # plotting the distribution
        ax_objs[-1].plot(x_d, np.exp(logprob), color="#f0f0f0", lw=1)
        ax_objs[-1].fill_between(x_d, np.exp(logprob), alpha=1, color=colors[idx % len(colors)])

        # setting uniform x and y lims
        ax_objs[-1].set_xlim(xmin, xmax)
        ax_objs[-1].set_ylim(0, np.exp(logprob).max() + 0.2)  # added a bit of padding

        # make background transparent
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # remove borders, axis ticks, and labels
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].yaxis.set_ticks([])  # Remove y-axis ticks

        if idx == len(regions) - 1:
            ax_objs[-1].set_xlabel("Duration (hours)", fontsize=16, fontweight="bold")
        else:
            ax_objs[-1].set_xticklabels([])

        # Determine the y-limits for the grid lines, which would be confined to the KDE
        y_limit = np.exp(logprob).max() + 0.2

        # Calculate grid line positions
        num_lines = 10
        interval = round(xmax) // num_lines
        interval = max(1, interval) + 1 
        
        # Draw each grid line manually using axvline
        for grid_x in range(0, int(xmax), interval):
            ax_objs[-1].axvline(x=grid_x, ymin=0, ymax=y_limit, color='grey', linestyle='--', linewidth=0.5)

        # Calculate the mean of the data for the current region
        mean_value = x.mean()

        # Draw a vertical line at the mean value
        ax_objs[-1].axvline(x=mean_value, ymin=0, ymax=np.exp(logprob).max() + 0.1, 
                            color='black', linestyle='-', linewidth=2, label="Mean")

        spines = ["top", "right", "left", "bottom"]
        for s in spines:
            ax_objs[-1].spines[s].set_visible(False)

        ax_objs[-1].text(-0.02, 0, rg, fontweight="bold", fontsize=14, ha="right")

        gs.update(hspace=-0.7)
        fig.text(0.07, 0.75, phase, fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(figure_path, f'Ridge_Plot_{phase}.png'))

def plot_ridge_plots(dfs, figure_path, phases):
    """
    Plot ridge plots for the given data.
    
    Args:
    - dfs (DataFrame): Data to plot
    - figure_path (str): Path to save the plots
    - phases (list): List of phase identifiers
    """
    colors = ['#3333ff', '#4040ff', '#5959cc', '#737399', '#868699', '#a68099', '#b38699', '#c79999']
    
    for phase in phases:
        data = dfs[dfs['Phase'] == phase]
        regions = dfs['Region'].unique()
        gs = grid_spec.GridSpec(len(regions), 1)
        fig = plt.figure(figsize=(12, 9))
        plot_single_ridge(data, regions, phase, colors, fig, gs, figure_path)

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

    data_frames = []
    for region in regions:
        region_str = f'_{region}' if region else '_SAt'
        data_path = os.path.join('..', 'periods-energetics', analysis_type + region_str)
        
        duration_data = process_phase_data_parallel(data_path)
        print(f'Processed data for region: {region}')

        df = create_seaborn_dataframe(duration_data, region if region else 'Total')
        data_frames.append(df)

    phases = list(duration_data.keys())
    dfs = pd.concat(data_frames)

    plot_ridge_plots(dfs, figure_path, phases)

if __name__ == '__main__':
    main()