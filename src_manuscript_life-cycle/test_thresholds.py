import os
from glob import glob 
import pandas as pd
import matplotlib.pyplot as plt

from cyclophaser import determine_periods
from cyclophaser.determine_periods import periods_to_dict, process_vorticity

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

# Configuration variables
TRACKS_DIRECTORY = "../processed_tracks_with_periods/"
OUTPUT_DIRECTORY = '../figures/manuscript_life-cycle/changing_thresholds/'
LABELS = ["(A)", "(B)", "(C)", "(D)"]

def plot_all_periods(phases_dict, ax, vorticity, label):
    colors_phases = {'incipient': '#65a1e6',
                      'intensification': '#f7b538',
                        'mature': '#d62828',
                          'decay': '#9aa981',
                          'residual': 'gray'}

    ax.plot(vorticity.time, vorticity.zeta, linewidth=10, color='gray', alpha=0.8, label=r'ζ')
    ax.plot(vorticity.time, vorticity.vorticity_smoothed, linewidth=6,
             c='#1d3557', alpha=0.8, label=r'$ζ_{fs}$')
    ax.plot(vorticity.time, vorticity.vorticity_smoothed2, linewidth=3,
             c='#e63946', alpha=0.6, label=r'$ζ_{fs^{2}}$')

    if len(vorticity.time) < 50:
        dt = pd.Timedelta(1, unit='h')
    else:
       dt = pd.Timedelta(0, unit='h')

    # Shade the areas between the beginning and end of each period
    for phase, (start, end) in phases_dict.items():
        # Extract the base phase name (without suffix)
        base_phase = phase.split()[0]

        # Access the color based on the base phase name
        color = colors_phases[base_phase]

        # Fill between the start and end indices with the corresponding color
        ax.fill_between(vorticity.time, vorticity.zeta.values,
                         where=(vorticity.time >= start) & (vorticity.time <= end + dt),
                        alpha=0.5, color=color, label=base_phase)

    # if i == 34:
    #     ax.legend(loc='upper right', bbox_to_anchor=(4.1, 0.25), fontsize=14)

    ax.text(0.85, 0.84, label, fontsize=16, fontweight='bold', ha='left', va='bottom', transform=ax.transAxes)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    date_format = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)
    ax.set_xlim(vorticity.time.min(), vorticity.time.max())
    ax.set_ylim(vorticity.zeta.min() - 0.25e-5, 0)

    # Add this line to set x-tick locator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))  

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

    ax.set_yticks([])
    ax.set_xticks([])

# Set initial thresholds and vorticity processing arguments
initial_periods_args = {
    'threshold_intensification_length': 0.075,
    'threshold_intensification_gap': 0.075,
    'threshold_mature_distance': 0.125,
    'threshold_mature_length': 0.03,
    'threshold_decay_length': 0.075,
    'threshold_decay_gap': 0.075,
    'threshold_incipient_length': 0.4
}
process_vorticity_args = {
    "use_filter": 'auto',
    "replace_endpoints_with_lowpass": 24,
    "use_smoothing": 'auto',
    "use_smoothing_twice": 'auto', 
    "savgol_polynomial": 3,
    "cutoff_low": 168,
    "cutoff_high": 48.0
}

system_ids = [20101172, 20001176, 19840092, 19970580, 20170528]
adjustments = [-1, -0.5, -0.25, 0, 0.25, 0.5, 1]

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# Iterate through each parameter in initial_periods_args
for param, base_value in initial_periods_args.items():
    fig, axes = plt.subplots(len(system_ids), len(adjustments), figsize=(20, 10))
    fig.suptitle(f"Variations in {param}")

    i = 0
    for col_index, adj in enumerate(adjustments):
        adjusted_value = base_value * (1 + adj)
        adjusted_periods_args = initial_periods_args.copy()
        adjusted_periods_args[param] = adjusted_value

        for row_index, system_id in enumerate(system_ids):
            tracks = glob(TRACKS_DIRECTORY + "*.csv")

            print(f"Plotting {param} = {adjusted_value} for {system_id}")
            
            for track_file in tracks:
                tracks_df = pd.read_csv(track_file, index_col=0)
                desired_cyclone = tracks_df[tracks_df['track_id'] == system_id]

                if desired_cyclone.empty:
                    continue  # Skip if no track files for this system_id

                desired_cyclone = tracks_df[tracks_df['track_id'] == int(system_id)]

                desired_cyclone.loc[:, 'date'] = pd.to_datetime(desired_cyclone['date'])
                desired_cyclone.loc[:, 'vor42'] = desired_cyclone['vor42'] * -1e-5

                df_periods = determine_periods(desired_cyclone['vor42'].tolist(), x=desired_cyclone['date'].tolist(), 
                                            plot=False, plot_steps=False, export_dict=False, 
                                            process_vorticity_args=process_vorticity_args, 
                                            periods_args=adjusted_periods_args)
                periods_dict = periods_to_dict(df_periods)
                ax = axes[row_index, col_index]

                zeta_df = desired_cyclone[['date', 'vor42']].rename(columns={'vor42': 'zeta'})
                zeta_df.set_index('date', inplace=True)
                zeta_df.index.name = 'time'
                vorticity = process_vorticity(zeta_df)

                if row_index_index == 0:
                    label = f"{adj}"
                else:
                    label = ""

                plot_all_periods(periods_dict, ax, vorticity, label)
                i += 1

    # Finalizing the figure
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(OUTPUT_DIRECTORY + f"figure_{param}.png")
    plt.close(fig)
    print(f"Saved figure {param}")