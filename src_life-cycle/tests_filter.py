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

from determine_periods import get_periods, plot_didactic

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
    ax.plot(vorticity.time, vorticity.zeta_filt2 * scaling_factor, linewidth=2, c='#e63946', label=r'$ζ_{fs^{2}}$')

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


def pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    """
    order = ((window - 1) // 2) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1.0, n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2.0 * np.pi * cutoff * k) / (np.pi * k)
    w[n - 1 : 0 : -1] = firstfactor * sigma
    w[n + 1 : -1] = firstfactor * sigma
    return w[1:-1]

def lanczos_filter(variable, window_length_lanczo, frequency):
    weights = pass_weights(window_length_lanczo, 1.0 / frequency)
    filtered_variable = convolve(variable, weights, mode="same")
    return filtered_variable



def find_peaks_valleys(series):
    """
    Find peaks, valleys, and zero locations in a pandas series

    Args:
    series: pandas Series

    Returns:
    result: pandas Series with nans, "peak", "valley", and 0 in their respective positions
    """
    # Extract the values of the series
    data = series.values

    # Find peaks, valleys, and zero locations
    peaks = argrelextrema(data, np.greater_equal)[0]
    valleys = argrelextrema(data, np.less_equal)[0]
    zeros = np.where(data == 0)[0]

    # Create a series of NaNs
    
    result = pd.Series(index=series.time, dtype=object)
    result[:] = np.nan

    # Label the peaks, valleys, and zero locations
    result.iloc[peaks] = 'peak'
    result.iloc[valleys] = 'valley'
    result.iloc[zeros] = 0

    return result

def array_vorticity(zeta_df, window_length_lanczo, frequency, window_length_savgol):
    """
    Calculate derivatives of the vorticity and filter the resulting series

    Args:
    df: pandas DataFrame

    Returns:
    xarray DataArray
    """
    # Convert dataframe to xarray
    da = zeta_df.to_xarray().copy()

    # # Filter vorticity for both periods
    zeta_filtred = lanczos_filter(da.zeta.copy(), window_length_lanczo, frequency)
    zeta_filtred = xr.DataArray(zeta_filtred, coords={'time':zeta_df.index})


    da = da.assign(variables={'zeta_filt': zeta_filtred})

    savgol_polynomial = 3
   
    zeta_smoothed = xr.DataArray(
        savgol_filter(zeta_filtred, window_length_savgol, savgol_polynomial, mode="nearest"),
        coords={'time':zeta_df.index})
    da = da.assign(variables={'zeta_smoothed': zeta_smoothed})

    window_length_savgol_2nd = window_length_savgol // 2 | 1
    if window_length_savgol_2nd < savgol_polynomial:
        window_length_savgol_2nd = 3

    zeta_filt2 = xr.DataArray(
        savgol_filter(zeta_smoothed, window_length_savgol_2nd, savgol_polynomial, mode="nearest"),
                        coords={'time':zeta_df.index})
    da = da.assign(variables={'zeta_filt2': zeta_filt2})

    dz_dt = da.zeta.differentiate('time', datetime_unit='h')
    dz_dt2 = dz_dt.differentiate('time', datetime_unit='h')
    
    dzfilt_dt = zeta_filt2.differentiate('time', datetime_unit='h')
    dzfilt_dt2 = dzfilt_dt.differentiate('time', datetime_unit='h')
    
    # Filter derivatives
    dz_dt_filt = xr.DataArray(
        savgol_filter(dzfilt_dt, window_length_savgol, savgol_polynomial, mode="nearest"),
        coords={'time':zeta_df.index})
    dz_dt2_filt = xr.DataArray(
        savgol_filter(dzfilt_dt2, window_length_savgol, savgol_polynomial, mode="nearest"),
        coords={'time':zeta_df.index})
    
    dz_dt_filt2 = xr.DataArray(
        savgol_filter(dz_dt_filt, window_length_savgol, savgol_polynomial, mode="nearest"),
        coords={'time':zeta_df.index})
    dz_dt2_filt2 = xr.DataArray(
        savgol_filter(dz_dt2_filt, window_length_savgol, savgol_polynomial, mode="nearest"),
        coords={'time':zeta_df.index})

    # Assign variables to xarray
    da = da.assign(variables={'dz_dt': dz_dt,
                              'dz_dt2': dz_dt2,
                              'dz_dt_filt': dz_dt_filt,
                              'dz_dt2_filt': dz_dt2_filt,
                              'dz_dt_filt2': dz_dt_filt2,
                              'dz_dt2_filt2': dz_dt2_filt2,
                              })

    return da

def plot_vorticity(ax, variable, color):
    """
    Plot the vorticity series and the filtered vorticity series with peaks and valleys marked

    Args:
    vorticity: xarray DataArray
    frequency: int
    """
    # Find peaks and valleys in the filtered vorticity series
    peaks_valleys = find_peaks_valleys(variable)

    # Extract peak and valley indices
    peaks = peaks_valleys[peaks_valleys == 'peak'].index
    valleys = peaks_valleys[peaks_valleys == 'valley'].index

    line = ax.plot(indexes, variable, color=color)
    ax.scatter(peaks, variable.loc[peaks], color=color, marker='o')
    ax.scatter(valleys, variable.loc[valleys], color=color, marker='o', facecolors='none')
    
    return line

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
    lengh_zeta = len(zeta_df)
    # frequency = 48
    high_frequency = 24
    window_length_savgol = lengh_zeta // 2 | 1
    window_length_lanczo = lengh_zeta // 20
    sampling_frequency = int((zeta_df.index[1] - zeta_df.index[0]).total_seconds() / 3600)


    vorticity = array_vorticity(zeta_df.copy(), window_length_lanczo, high_frequency, window_length_savgol)

    # Determine the periods
    try:
        periods_dict, df = get_periods(vorticity.copy())
    except:
        print(f'Error')
        continue

    # Create plots
    plot_all_periods(periods_dict, vorticity, Carol_vorticity,
                    periods_outfile_path=f"{periods_outfile_path}_{RG}-0.99-{cyclone_id}")
    plot_didactic(df, vorticity, f"{periods_didatic_outfile_path}_{RG}-0.99-{cyclone_id}")
