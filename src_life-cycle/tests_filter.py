from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from scipy.signal import firwin, lfilter

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

def low_pass_weights(window, cutoff):
    """Calculate weights for a low pass Lanczos filter.

    Args:

    window: int
        The length of the filter window.

    cutoff: float
        The cutoff frequency in inverse time steps.

    """
    order = ((window - 1) // 2 ) + 1
    nwts = 2 * order + 1
    w = np.zeros([nwts])
    n = nwts // 2
    w[n] = 2 * cutoff
    k = np.arange(1., n)
    sigma = np.sin(np.pi * k / n) * n / (np.pi * k)
    firstfactor = np.sin(2. * np.pi * cutoff * k) / (np.pi * k)
    w[n-1:0:-1] = firstfactor * sigma
    w[n+1:-1] = firstfactor * sigma
    return w[1:-1]

def lanczos_filter(variable, window_lenght, frequency):
    # Define the cutoff frequency for the Lanczos filter (cycles per sample)
    cutoff_frequency = 1 / frequency  # 12 hours

    # Generate the Lanczos filter coefficients
    t = np.arange(-(window_lenght - 1) / 2, (window_lenght - 1) / 2 + 1)
    lanczos_filter = np.sinc(2 * cutoff_frequency * t) * np.sinc(t / (window_lenght - 1))

    # Normalize the filter coefficients
    lanczos_filter /= lanczos_filter.sum()

    # Apply the Lanczos filter to the zeta data
    filtered_zeta = lfilter(lanczos_filter, 1.0, variable)

    return filtered_zeta


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

def array_vorticity(zeta_df, window_lenght, frequency):
    """
    Calculate derivatives of the vorticity and filter the resulting series

    Args:
    df: pandas DataFrame

    Returns:
    xarray DataArray
    """
    # Convert dataframe to xarray
    da = zeta_df.to_xarray().copy()

    # Filter vorticity twice
    zeta_filtred = xr.DataArray(lanczos_filter(da.zeta.copy(), window_lenght, frequency),
                              coords={'time':zeta_df.index})
    da = da.assign(variables={'zeta_filt': zeta_filtred})

    savgol_window_lenght = (len(zeta_df) // 5) + 1
    savgol_window_lenght_dz = (len(zeta_df) // 2) + 1
    savgol_polynomial = 3
   
    zeta_smoothed = xr.DataArray(
        savgol_filter(zeta_filtred, savgol_window_lenght, savgol_polynomial, mode="nearest"),
        coords={'time':zeta_df.index})
    da = da.assign(variables={'zeta_filt2': zeta_smoothed})

    dz_dt = da.zeta.differentiate('time', datetime_unit='h')
    dz_dt2 = dz_dt.differentiate('time', datetime_unit='h')
    
    dzfilt_dt = zeta_smoothed.differentiate('time', datetime_unit='h')
    dzfilt_dt2 = dzfilt_dt.differentiate('time', datetime_unit='h')
    
    # Filter derivatives
    dz_dt_filt2 = xr.DataArray(
        savgol_filter(dzfilt_dt, savgol_window_lenght_dz, savgol_polynomial, mode="nearest"),
        coords={'time':zeta_df.index})
    dz_dt2_filt2 = xr.DataArray(
        savgol_filter(dzfilt_dt2, savgol_window_lenght_dz, savgol_polynomial, mode="nearest"),
        coords={'time':zeta_df.index})

    # Assign variables to xarray
    da = da.assign(variables={'dz_dt': dz_dt,
                              'dz_dt2': dz_dt2,
                              #'dzfilt2_dt': dzfilt_dt,
                              #'dzfilt2_dt2': dzfilt_dt2,
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

for file in sorted(glob('../tracks_LEC-format/ALL/intense/*'))[:1]:

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
    # min_value = round(lengh_zeta * 0.3)
    # max_value = round(lengh_zeta * 0.6)

    cmap = plt.get_cmap('coolwarm')
    cmap = truncate_colormap(cmap, 0.1, 0.9, 5)

    window_length = lengh_zeta * 0.1

    frequencies = [12, 24]

    for i, frequency in enumerate(frequencies):

        print(f'Frequency: {frequency}')
        fig1 = plt.figure(figsize=(14, 8))
       
        vorticity = array_vorticity(zeta_df.copy(), window_length, frequency)

        variables = [vorticity.zeta,
                    vorticity.zeta,
                    vorticity.dz_dt  * 0.25,
                    vorticity.dz_dt2  * 0.025]
        
        filtered_variables = [vorticity.zeta_filt,
                              vorticity.zeta_filt2,
                              vorticity.dz_dt_filt2,
                              vorticity.dz_dt2_filt2]
        
        titles = [r"$ζ_{f}$",
                  r"$ζ_{fs}$",
                  r"$\frac{\partial ζ_{fs}}{\partial t}_s$",
                  r"$\frac{\partial^{2} ζ_{fs}}{\partial t^{2}}_s$"]
        
        colors = ["#264653", "#2a9d8f", "#e76f51", "#f4a261"]

        for i, variable_fil in enumerate(filtered_variables):

            ax = fig1.add_subplot(2, 2, i + 1)

            plot_vorticity(ax, variable_fil, colors[i])

            ax.plot(pd.to_datetime(vorticity.time), variables[i], c='k', alpha=0.8, lw=1)
            ax.set_title(titles[i], fontweight='bold', horizontalalignment='center')

            date_format = mdates.DateFormatter("%d %HZ")
            ax.xaxis.set_major_formatter(date_format)
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))  # Adjust interval as needed
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.subplots_adjust(hspace=0.5)

        fname = os.path.join(output_directory, f'test_filter_{frequency}h.png')
        fig1.savefig(fname, bbox_inches='tight')
        print(f"{fname} created.")

        # Determine the periods
        try:
            periods_dict, df = get_periods(vorticity)
        except:
            print(f'Error')
            continue

        # Create plots
        plot_all_periods(periods_dict, df, ax=None, vorticity=vorticity.zeta,
                        periods_outfile_path=f"{periods_outfile_path}_{frequency}")
        plot_didactic(df, vorticity, f"{periods_didatic_outfile_path}_{frequency}")
        # export_periods_to_csv(periods_dict, f"{periods_outfile_path}_{window_length}") 
      

    # Create a legend outside the figure on the right
    # fig1.legend(handles, labels, loc='center right', bbox_to_anchor=(0.6, 0.5))
    # fig1.subplots_adjust(right=0.8)  # Make space for the legend
