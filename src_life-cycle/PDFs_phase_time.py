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

# Constants
SECONDS_IN_AN_HOUR = 3600

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
        
        # Deduce season from the 'start' date
        month = df.loc['incipient', 'start'].month
        season = 'DJF' if month in [12, 1, 2] else ('JJA' if month in [6, 7, 8] else None)
        
        phase_durations = {
            phase: ((df.loc[phase, 'end'] - df.loc[phase, 'start']).total_seconds() / SECONDS_IN_AN_HOUR)
            for phase in df.index
        }
        return {**phase_durations, "Season": season}
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

def plot_single_ridge(data, regions, figure_path, phase):
    """
    Plot a single ridge plot for given data, combining both seasons in one subplot.
    """
    # Create a figure with a row for each region
    fig, ax = plt.subplots(nrows=len(data['Region'].unique()), ncols=2, figsize=(12, 9))
    
    duration = data['Duration (hours)']
    xmin, xmax = duration.min(), duration.max()
    x_d = np.linspace(xmin, xmax, 1000)

    # Define the vertical gap between rows to create overlapping effect
    vertical_gap = 0.07 
    horizontal_gap = 0.01

    blue_shades = ['#010339','#03045e','#023e8a','#0077b6','#0096c7','#00b4d8','#48cae4','#a4defb']
    red_shades = ['#1c0008','#370617','#6a040f','#9d0208','#d00000','#dc2f02','#e85d04','#ff9066']

    for idx, rg in enumerate(regions):
        color_djf = red_shades[idx]
        color_jja = blue_shades[idx]
        for season, col in zip(['DJF', 'JJA'], [0, 1]):                
            season_data = data[(data['Region'] == rg) & (data['Season'] == season)]
            x = np.array(season_data['Duration (hours)'])
            
            kde = KernelDensity(bandwidth=1, kernel='gaussian')
            kde.fit(x[:, None])

            color = color_djf if season == "DJF" else color_jja

            logprob = kde.score_samples(x_d[:, None])
            ax[idx][col].plot(x_d, np.exp(logprob), color="#f0f0f0", lw=1, linestyle='-')
            ax[idx][col].fill_between(x_d, np.exp(logprob), alpha=1, color=color, linestyle='-')
            
            # setting uniform x and y lims
            ax[idx][col].set_xlim(xmin, xmax)
            ax[idx][col].set_ylim(0, np.exp(logprob).max() + 0.2)  # added a bit of padding

            # make background transparent
            rect = ax[idx][col].patch
            rect.set_alpha(0)

            # remove borders, axis ticks, and labels
            ax[idx][col].set_yticklabels([])
            ax[idx][col].yaxis.set_ticks([])  # Remove y-axis ticks

            # Determine the y-limits for the grid lines, which would be confined to the KDE
            y_limit = np.exp(logprob).max() + 0.2

            # Calculate grid line positions
            num_lines = 10
            interval = round(xmax) // num_lines
            interval = max(1, interval) + 1 
            
            # Draw each grid line manually using axvline
            for grid_x in range(0, int(xmax), interval):
                ax[idx][col].axvline(x=grid_x, ymin=0, ymax=y_limit, color='grey', linestyle='--', linewidth=0.5)

            # Calculate the mean of the data for the current region
            mean_value = x.mean()

            color_line = 'r' if season == "DJF" else 'b'
            # Draw a vertical line at the mean value
            ax[idx][col].axvline(x=mean_value, ymin=0, ymax=np.exp(logprob).max() + 0.1, 
                                color=color_line, linestyle='-', linewidth=2, label="Mean")

            spines = ["top", "right", "left", "bottom"]
            for s in spines:
                ax[idx][col].spines[s].set_visible(False)
            if col == 0:
                ax[idx][col].text(-0.02, 0, rg, fontweight="bold", fontsize=14, ha="right")
            
            # x-label
            if idx == len(regions) - 1:
                fig.text(0.5, 0.55, 'Duration (hours)', ha='center', fontsize=16, fontweight="bold")
            ax[idx][col].set_xticklabels([])

            # season
            if idx == 0:
                letter = "A" if col == 0 else "B"
                ax[idx][col].text(10, 0.2, f"({letter}) {season}", fontweight="bold", fontsize=14, ha="right")

            # Adjust x-position of the second column to make columns closer
            if col == 1:
                pos = ax[idx][col].get_position()
                ax[idx][col].set_position([pos.x0 + horizontal_gap, pos.y0, pos.width, pos.height])

        # Adjust the position of each row of axes to create overlap
        pos = ax[idx][0].get_position()
        ax[idx][0].set_position([pos.x0, pos.y0 + idx * vertical_gap, pos.width, pos.height])
        ax[idx][1].set_position([pos.x0 + pos.width + horizontal_gap, pos.y0 + idx * vertical_gap, pos.width, pos.height])

    # Save the combined figure for the current phase
    # plt.tight_layout()
    plt.savefig(os.path.join(figure_path, f'Ridge_Plot_{phase}.png'))

def plot_ridge_plots(dfs, figure_path, phases):
    for phase in phases:
        
        print(f"Plotting phase: {phase}")
        # Filter the data for the current phase
        data = dfs[dfs['Phase'] == phase]
        regions = dfs['Region'].unique()
            
        plot_single_ridge(data, regions, figure_path, phase)


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
        region_str = f'_{region}' if region else ''
        data_path = os.path.join('..', 'periods-energetics', analysis_type + region_str)
        
        duration_data = process_phase_data_parallel(data_path)
        print(f'Processed data for region: {region}')

        df = create_seaborn_dataframe(duration_data, region if region else 'Total')
        data_frames.append(df)

    phases = [key for key in list(duration_data.keys()) if key not in ['Season', 'incipient 2']]
    dfs = pd.concat(data_frames)

    plot_ridge_plots(dfs, figure_path, phases)

if __name__ == '__main__':
    main()