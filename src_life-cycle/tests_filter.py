from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from scipy.signal import convolve

from glob import glob

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors

import os

import xarray as xr
import pandas as pd
import numpy as np

from determine_periods import get_periods, plot_didactic, array_vorticity

def plot_all_periods(periods_dict, vorticity, Carol_vorticity, periods_outfile_path=None):
    colors_phases = {'incipient': '#65a1e6',
                      'intensification': '#f7b538',
                        'mature': '#d62828',
                          'decay': '#9aa981',
                          'residual': 'gray'}

    # Create a new figure if ax is not provided
    fig, ax = plt.subplots(figsize=(6.5, 5))

    # Plot the vorticity data
    ax.plot(vorticity.time, vorticity.zeta, linewidth=2.5, color='gray', label=r'ζ')

    # Plot filtered vorticities
    scaling_factor = np.max(vorticity.zeta) / np.max(vorticity.zeta_filt)
    ax.plot(vorticity.time, vorticity.zeta_filt * scaling_factor, linewidth=2, c='#d68c45', label=r'$ζ_{f}$')
    ax.plot(vorticity.time, vorticity.zeta_smoothed * scaling_factor, linewidth=2, c='#1d3557', label=r'$ζ_{fs}$')
    ax.plot(vorticity.time, vorticity.zeta_smoothed2 * scaling_factor, linewidth=2, c='#e63946', label=r'$ζ_{fs^{2}}$')

    # Plot Carol's vorticity
    scaling_factor = np.max(vorticity.zeta) / np.max(Carol_vorticity['vor42'])
    ax.plot(pd.to_datetime(Carol_vorticity['date']),
             Carol_vorticity['vor42'] * float(scaling_factor),
             c='k', alpha=0.6, linestyle='--',  linewidth=2, label=r'$ζ_{TRACK} \times -1^{-5}$')

    legend_labels = set()  # To store unique legend labels

    # Shade the areas between the beginning and end of each period
    for phase, (start, end) in periods_dict.items():
        # Extract the base phase name (without suffix)
        base_phase = phase.split()[0]

        # Access the color based on the base phase name
        color = colors_phases[base_phase]

        # Fill between the start and end indices with the corresponding color
        ax.fill_between(vorticity.time, vorticity.zeta.values, where=(vorticity.time >= start) & (vorticity.time <= end),
                        alpha=0.4, color=color, label=base_phase)

        # Add the base phase name to the legend labels set
        legend_labels.add(base_phase)

    # Add legend labels for Vorticity and ζ
    for label in [r'ζ', r'$ζ_{f}$', r'$ζ_{fs}$', r'$ζ_{TRACK} \times -1^{-5}$', r'$ζ_{fs^{2}}$']:
        legend_labels.add(label)

    # Set the title
    ax.set_title('Vorticity Data with Periods')

    if periods_outfile_path is not None:
        # Remove duplicate labels from the legend
        handles, labels = ax.get_legend_handles_labels()
        unique_labels = []
        for label in labels:
            if label not in unique_labels and label in legend_labels:
                unique_labels.append(label)

        ax.legend(handles, unique_labels, loc='upper right', bbox_to_anchor=(1.5, 1))

        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        date_format = mdates.DateFormatter("%Y-%m-%d")
        ax.xaxis.set_major_formatter(date_format)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()

        fname = f"{periods_outfile_path}.png"
        plt.savefig(fname, dpi=500)
        print(f"{fname} created.")


Carol_tracks_file = '../stats_tracks/BY_RG/tracks-RG1_q0.99.csv'
Carol_tracks = pd.read_csv(Carol_tracks_file)
Carol_tracks.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp',
                         'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']


for file in sorted(glob('../LEC_results-q0.99/*')):

    print(file)

    cyclone_id = os.path.basename(file).split('_')[0].split('-')[2]
    RG =  os.path.basename(file).split('-')[0]

    try:
        track_file = glob(f'{file}/*track')[0]
    except:
        print('failed for',cyclone_id)
        continue

    Carol_track_id = Carol_tracks[Carol_tracks['track_id'] == int(cyclone_id)]
    Carol_vorticity = Carol_track_id[['date','vor42']]
    
    output_directory = f'../figures/tests_filter/0.99/'

    os.makedirs(output_directory, exist_ok=True)

    # Set the output file names
    periods_outfile_path = output_directory + 'periods'
    periods_didatic_outfile_path = output_directory + 'periods_didatic'

    # Read the track file and extract the vorticity data
    track = pd.read_csv(track_file, parse_dates=[0], delimiter=';', index_col=[0])
    zeta_df = pd.DataFrame(track['min_zeta_850'].rename('zeta'))    

    indexes = zeta_df.index

    vorticity = array_vorticity(zeta_df.copy())

    # Determine the periods
    periods_dict, df = get_periods(vorticity.copy())

    # Create plots
    plot_all_periods(periods_dict, vorticity, Carol_vorticity,
                    periods_outfile_path=f"{periods_outfile_path}_{RG}-0.99-{cyclone_id}")
    plot_didactic(df, vorticity, f"{periods_didatic_outfile_path}_{RG}-0.99-{cyclone_id}")
