import pandas as pd
import glob
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def process_csv_file(csv_file):
    df = pd.read_csv(csv_file, parse_dates=['start', 'end'], index_col=0)
    phase_durations = {}
    for phase in df.index:
        start_time = df.loc[phase, 'start']
        end_time = df.loc[phase, 'end']
        duration_hours = (end_time - start_time).total_seconds() / 3600
        phase_durations[phase] = duration_hours
    return phase_durations

def process_phase_data_parallel(data_path):
    csv_files = glob.glob(f'{data_path}*.csv')

    durations = {}
    
    # Use ThreadPoolExecutor to process files in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(process_csv_file, csv_files), total=len(csv_files), desc='Processing Files'))

    for result in results:
        for phase, duration in result.items():
            if phase not in durations:
                durations[phase] = []
            durations[phase].append(duration)

    return durations

def create_seaborn_dataframe(duration_data, region):
    data = {'Phase': [], 'Duration (hours)': [], 'Region': []}

    for phase, phase_durations in duration_data.items():
        data['Phase'].extend([phase] * len(phase_durations))
        data['Duration (hours)'].extend(phase_durations)
    
    data['Region'] = region

    df = pd.DataFrame(data)
    return df

def plot_ridge_plots(dfs, figure_path, phases, rgs):
    for phase in phases:
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

        df = dfs[dfs['Phase'] == phase]
        df_max = int(df["Duration (hours)"].max())

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
        pal= sns.color_palette("Set2", 12)
        g = sns.FacetGrid(df, row="Region", hue="Region", aspect=15, height=.5, palette=pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "Duration (hours)",
            bw_adjust=.5, clip_on=False,
            fill=True, alpha=1, linewidth=1.5)
        g.map(sns.kdeplot, "Duration (hours)", clip_on=False, color="w", lw=2, bw_adjust=.5)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates
        def label(x, color, label):
            ax = plt.gca()
            ax.text(1, 0.2, label, fontweight="bold", color=color,
                    ha="right", va="center", transform=ax.transAxes, zorder=100)
        g.map(label, "Duration (hours)")

        for i, (ax, region) in enumerate(zip(g.axes.flat, rgs)):
            region = region if region else 'Total'
            ax.set_xlim(0, max(df["Duration (hours)"]))  # Set the x-axis lower bound to 0
            region_mean = df[df["Region"] == region]["Duration (hours)"].mean()
            ax.axvline(x=region_mean, color="k", linestyle="-",
                       label=f"Mean", alpha=0.8)

            # Add vertical gridlines for ticks on the x-axis (e.g., every 10 hours)
            for i in range(0, df_max, 5):
                ax.axvline(x=i, color="k", linestyle="--", lw=0.25, alpha=0.5, zorder=1)

        # Set the subplots to overlap
        g.figure.subplots_adjust(hspace=-.25)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="")
        g.despine(bottom=True, left=True)
        
        # Save the ridge plot for the phase
        plt.savefig(os.path.join(figure_path, f'Ridge_Plot_{phase}.png'))

def main():
    analysis_type = '70W-no-continental'

    figure_path = f'../figures/periods_statistics/{analysis_type}/phase_time/'
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)

    if analysis_type == 'BY_RG-all':
        rgs = ['RG1', 'RG2', 'RG3', 'all_RG']
    elif analysis_type == '70W-no-continental':
        rgs = [False, "ARG", "LA-PLATA", "SE-BR", "SE-SAO", "AT-PEN", "WEDDELL", "SA-NAM"]
    else:
        rgs = ['']

    data_frames = []
    for rg in rgs:
        rg_str = f'_{rg}' if rg else '_SAt'
        data_path = f'../periods-energetics/{analysis_type}{rg_str}/' if rg else f'../periods-energetics/{analysis_type}/'
        duration_data = process_phase_data_parallel(data_path)
        print(f'Processed data for RG: {rg}')

        # Create a DataFrame for Seaborn
        region = rg if rg else 'Total'
        df = create_seaborn_dataframe(duration_data, region)
        data_frames.append(df)

    phases = list(duration_data.keys())  # Get the list of phases from the last processed data
    dfs = pd.concat(data_frames)
    # Plot ridge plots for each phase
    plot_ridge_plots(dfs, figure_path, phases, rgs)

if __name__ == '__main__':
    main()
