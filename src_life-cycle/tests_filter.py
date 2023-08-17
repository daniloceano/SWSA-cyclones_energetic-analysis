from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

from glob import glob

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as colors

import os

import xarray as xr
import pandas as pd
import numpy as np

from determine_periods import get_periods, plot_all_periods, plot_didactic, export_periods_to_csv

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def filter_var(variable, window_lenght):
    # window_lenght = round(len(variable)/2)
    # if (window_lenght % 2) == 0:
    #     window_lenght += 1
    return savgol_filter(variable, window_lenght, 3, mode="nearest")

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

def array_vorticity(df, window_lenght):
    """
    Calculate derivatives of the vorticity and filter the resulting series

    Args:
    df: pandas DataFrame

    Returns:
    xarray DataArray
    """
    # Convert dataframe to xarray
    da = df.to_xarray()

    # Filter vorticity twice
    zeta_filt = xr.DataArray(filter_var(da.zeta, window_lenght), coords={'time':df.index})
    da = da.assign(variables={'zeta_filt': zeta_filt})
    zeta_filt2 = xr.DataArray(filter_var(zeta_filt, window_lenght), coords={'time':df.index})
    da = da.assign(variables={'zeta_filt2': zeta_filt2})

    dz_dt = da.zeta.differentiate('time', datetime_unit='h')
    dz_dt2 = dz_dt.differentiate('time', datetime_unit='h')

    # Calculate derivatives of the double-filtered vorticity
    dzfilt2_dt = da.zeta_filt2.differentiate('time', datetime_unit='h')
    dzfilt2_dt2 = dzfilt2_dt.differentiate('time', datetime_unit='h')

    # Filter derivatives
    dz_dt_filt2 = xr.DataArray(filter_var(dzfilt2_dt, window_lenght), coords={'time':df.index})
    dz_dt2_filt2 = xr.DataArray(filter_var(dzfilt2_dt2, window_lenght), coords={'time':df.index})

    # Assign variables to xarray
    da = da.assign(variables={'dz_dt': dz_dt,
                              'dz_dt2': dz_dt2,
                              'dzfilt2_dt': dzfilt2_dt,
                              'dzfilt2_dt2': dzfilt2_dt2,
                              'dz_dt_filt2': dz_dt_filt2,
                              'dz_dt2_filt2': dz_dt2_filt2})

    return da

def plot_vorticity(axs, vorticity, window_length, color):
    """
    Plot the vorticity series and the filtered vorticity series with peaks and valleys marked

    Args:
    vorticity: xarray DataArray
    window_length: int
    """

    ax1, ax2 = axs

    z_fil = vorticity.zeta_filt
    z_fil2 = vorticity.zeta_filt2

    for ax, variabe in zip([ax1, ax2], [z_fil, z_fil2]):

        label = f'Window {window_length}' if ax == ax1 else ''

        # Find peaks and valleys in the filtered vorticity series
        peaks_valleys = find_peaks_valleys(variabe)

        # Extract peak and valley indices
        peaks = peaks_valleys[peaks_valleys == 'peak'].index
        valleys = peaks_valleys[peaks_valleys == 'valley'].index

        line = ax.plot(indexes, variabe, color=color, label=label)
        ax.scatter(peaks, variabe.loc[peaks], color=color, marker='o')
        ax.scatter(valleys, variabe.loc[valleys], color=color, marker='o', facecolors='none')
    
    return line

for file in glob('../tracks_LEC-format/ALL/intense/*'):

    print(file)

    cyclone_id = os.path.basename(file).split('_')[1]

    track_file = f'../LEC_results-10MostIntense/10MostIntense-{cyclone_id}_ERA5_track-15x15/10MostIntense-{cyclone_id}_ERA5_track-15x15_track'
    output_directory = f'../figures/tests_filter/{cyclone_id}/'

    os.makedirs(output_directory, exist_ok=True)

    # Set the output file names
    periods_outfile_path = output_directory + 'periods'
    periods_didatic_outfile_path = output_directory + 'periods_didatic'

    # Read the track file and extract the vorticity data
    track = pd.read_csv(track_file, parse_dates=[0], delimiter=';', index_col=[0])
    zeta_df = pd.DataFrame(track['min_zeta_850'].rename('zeta'))    

    #indexes = range(len(zeta_df))
    indexes = zeta_df.index

    lengh_zeta = len(zeta_df)
    min_value = round(lengh_zeta*0.15) // 2 + 1
    max_value = 49 if lengh_zeta > 49 else lengh_zeta // 2

    cmap = plt.get_cmap('coolwarm')
    cmap = truncate_colormap(cmap, 0.3, 0.9, max_value+1)

    fig1 = plt.figure(figsize=(12, 8))
    ax1 = fig1.add_subplot(121)
    ax2 = fig1.add_subplot(122)
    axs = [ax1, ax2]
    handles = []  # To store legend handles
    labels = []   # To store legend labels

    for window_length in range(min_value, max_value+4, 4):
        print(f'Window length: {window_length}')
        if window_length % 2 != 0:
            vorticity = array_vorticity(zeta_df, window_length)

            # Determine if the window length is half of vorticity length or +1
            is_half_length = window_length == max_value // 2 or window_length == max_value // 2 + 1 
            if is_half_length:
                color = 'black'  # Set color to black
            else:
                # Get a color from the colormap based on the normalized window length
                color = cmap((window_length - 5) / (max_value - 5))

            line = plot_vorticity(axs, vorticity, window_length, color)
            handles.append(line[0])
            labels.append(f'Window {window_length}')

            # Determine the periods
            try:
                periods_dict, df = get_periods(vorticity)
            except:
                print(f'Error with window length {window_length}')
                continue

            # Create plots
            plot_all_periods(periods_dict, df, ax=None, vorticity=vorticity.zeta,
                            periods_outfile_path=f"{periods_outfile_path}_{window_length}")
            plot_didactic(df, vorticity, f"{periods_didatic_outfile_path}_{window_length}")
            # export_periods_to_csv(periods_dict, f"{periods_outfile_path}_{window_length}")

    ax1.plot(vorticity.time, vorticity.zeta, c='k', alpha=0.8, lw=0.5, label=r'$ζ_{f}$')
    ax1.set_title("filter ζ", fontweight='bold', horizontalalignment='center')

    ax2.plot(vorticity.time, vorticity.zeta, c='k', alpha=0.8, lw=0.5, label=r"$ζ_{filt}$")
    ax2.set_title("filtered ζ", fontweight='bold', horizontalalignment='center')

    for ax in fig1.axes:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        date_format = mdates.DateFormatter("%D %H")
        ax.xaxis.set_major_formatter(date_format)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')        

    # Create a legend outside the figure on the right
    fig1.legend(handles, labels, loc='center right', bbox_to_anchor=(0.94, 0.5))
    fig1.subplots_adjust(right=0.8)  # Make space for the legend

    fname = os.path.join(output_directory, 'test_filter.png')
    fig1.savefig(fname, bbox_inches='tight')
    print(f"{fname} created.")

