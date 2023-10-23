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
    - dict: Dictionary containing phase durations and seasons
    """
    try:
        df = pd.read_csv(csv_file, parse_dates=['start', 'end'], index_col=0)
        
        # Deduce season from the 'start' date
        month = df['start'].min().month
        if month in  [12, 1, 2, 6, 7, 8]:
            season = 'DJF' if month in [12, 1, 2] else 'JJA'
        
            phase_durations = {
                phase: ((df.loc[phase, 'end'] - df.loc[phase, 'start']).total_seconds() / SECONDS_IN_AN_HOUR)
                for phase in df.index
            }
            return {**phase_durations, "Season": season}
        else:
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
        "intensification 2": [3, 70, 100],
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
            
            # Draw each grid line manually using axvline
            for grid_x in range(0, int(xmax), interval):
                ax.axvline(x=grid_x, ymin=0, ymax=y_limit, color='grey', linestyle='--', linewidth=0.5)

            # Calculate the mean of the data for the current region
            mean_value = x.mean()

            # Draw a vertical line at the mean value
            ax.axvline(x=mean_value, ymin=0, ymax=np.exp(logprob).max() + 0.2, 
                                color='k', linestyle='-', linewidth=2, label="Mean")

            spines = ["top", "right", "left", "bottom"]
            for s in spines:
                ax.spines[s].set_visible(False)

            if col == 0:
                ax.text(-0.5, 0, rg, fontweight="bold", fontsize=14, ha="right")
            
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
        print(f"DJF mean duration: {data['Duration (hours)'].mean():.2f} hours")
        print(f"DJF max duration: {data['Duration (hours)'].max():.2f} hours")
        print(f"JJA mean duration: {data[data['Season'] == 'JJA']['Duration (hours)'].mean():.2f} hours")
        print(f"JJA max duration: {data[data['Season'] == 'JJA']['Duration (hours)'].max():.2f} hours")

    # phase = "residual"
    # data = dfs[dfs['Phase'] == phase]
    # plot_single_ridge(data, regions, figure_path, phase)

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

    phase_time_database = f"../periods_species_statistics/{analysis_type}/phase_time.csv"

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

if __name__ == '__main__':
    main()