# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_density.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/09 12:48:17 by Danilo            #+#    #+#              #
#    Updated: 2023/09/30 09:55:47 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Adapted from Carolina B. Gramcianinov (cbgramcianinov@gmail.com)

Reads the raw tracks and the processed periods for each system and produces netCDF files
containing the density map of each period in  2.5 degree global grid
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from scipy.ndimage.filters import gaussian_filter
from glob import glob
import xarray as xr

def get_tracks(RG, season=False):

    if season:
        print('Merging tracks for RG:', RG, 'and season:', season)
    else:
        print('Merging tracks for RG:', RG)

    month_season_map = {
    12: 'DJF', 1: 'DJF', 2: 'DJF',
    3: 'MAM', 4: 'MAM', 5: 'MAM',
    6: 'JJA', 7: 'JJA', 8: 'JJA',
    9: 'SON', 10: 'SON', 11: 'SON'
}
    if analysis_type == 'BY_RG-all':
        track_columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']
        if RG != 'all':
            str_RG = f'RG{RG}'
            results_directories = [f'../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG{RG}_csv/']
        else:
            str_RG = 'all RGs'
            results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
                                '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
                                '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/']
    else:
        track_columns = ['track_id', 'date', 'lon vor', 'lat vor', 'vor42']
        str_RG = 'all systems'
        results_directories = ['../raw_data/SAt/']

    tracks = pd.DataFrame(columns = track_columns)

    for results_directory in results_directories:  
                files = glob(f'{results_directory}*')

                for file in files:  
                    try:
                        tmp = pd.read_csv(file)
                    except:
                        continue

                    tmp.columns = track_columns

                    # Check season, if season is given
                    if season:
                        system_end = pd.to_datetime(tmp['date'].iloc[-1])
                        system_month = system_end.month
                        corresponding_season = month_season_map[system_month]
                        if corresponding_season == season:
                            tracks = pd.concat([tracks,tmp])
                    else:
                        tracks = pd.concat([tracks,tmp])
                    
    x = tracks['lon vor'].values 
    tracks['lon vor'] = np.where(x > 180, x - 360, x)

    cyclone_ids = tracks['track_id'].unique()
    print(f"Number of cyclones for {str_RG}: {len(cyclone_ids)}")

    return tracks, cyclone_ids

def process_track(cyclone_id, tracks, periods_directory, filter_residual=False):

    track = tracks[tracks['track_id'] == cyclone_id].copy()
    track['date'] = pd.to_datetime(track['date'])
    track['lon vor'] = (track['lon vor'] + 180) % 360 - 180

    periods = pd.read_csv(glob(f'{periods_directory}*{cyclone_id}*')[0])
    periods.columns = ['period', 'start', 'end']
    periods = periods.set_index('period')
    
    dt = track['date'].iloc[1] - track['date'].iloc[0]

    # Convert period timestamps to datetime objects
    corresponding_periods = pd.DataFrame([], columns=['period'], index=track['date'])
    for phase in list(periods.index):
        periods.loc[phase] = pd.to_datetime(periods.loc[phase])
        period_dates = pd.date_range(start=periods.loc[phase][0], end=periods.loc[phase][1], freq=dt)

        corresponding_periods['period'].loc[period_dates] = phase

    # Add a new column 'period' to the track DataFrame
    track['period'] = corresponding_periods['period'].values

    if filter_residual:
        if 'residual' in track['period'].values:
            track = track[track['period'] != 'residual']

    return track

def compute_density(tracks_with_periods, num_time):
    """
    Computing track density using KDE folowing the idea of K. Hodges 
    (e.g., Hoskins and Hodges, 2005)
    """
    # (1) Creating a global grid with 128 x 64 (lon, lat): 2.5 degree
    k = 64
    longrd = np.linspace(-180, 180, 2 * k)
    latgrd = np.linspace(-87.863, 87.863 , k)
    tx, ty = np.meshgrid(longrd, latgrd)
    mesh = np.vstack((ty.ravel(), tx.ravel())).T
    mesh *= np.pi / 180.

    pos = tracks_with_periods[['lat vor', 'lon vor']]
    x = pos['lon vor'].values
    y = pos['lat vor'].values

    # (2) Building the KDE for the positions
    h = np.vstack([y, x]).T
    h *= np.pi / 180.  # Convert lat/long to radians
    bdw = 0.05
    kde = KernelDensity(bandwidth=bdw, metric='haversine',
        kernel='gaussian', algorithm='ball_tree').fit(h)

    # We evaluate the kde() function on the grid.
    v = np.exp(kde.score_samples(mesh)).reshape((k, 2 * k))

    # Converting KDE values to scalled density
    # (a) cyclone number: multiply by total number of genesis (= pos.shape[0])
    # (b) area: divide by R ** 2 (R = Earth Radius)
    # (c) area: scalle by 1.e6
    # (d) time: divide by the number the time unit that you wish (month or year, as you wish)
    #
    # --> The final unit is genesis/area/time
    # - here, area is 10^6 km^2 and time is months (within the dataset)
    #
    # --> obs: the absolute values obtained here for the track density are not comparable with the
    # ones obtained by Hoskin and Hodges (2005) due to a diffent normalization. (this need to be improved)
    # 

    R = 6369345.0 * 1e-3 # Earth radius in meters at 40ºS (WGS 84 reference ellipsoid)
    factor = (1 / (R * R) ) * 1.e6
    density = v * pos.shape[0] * factor /  num_time 

    return density, longrd, latgrd

# analysis_type = 'BY_RG-all'
# analysis_type = 'all'
analysis_type = '70W'

# Set up direcotries
periods_directory = f'../periods-energetics/{analysis_type}/'
output_directory = f'../periods_species_statistics/{analysis_type}/track_density'
os.makedirs(output_directory, exist_ok=True)

initial_year, final_year = 1979, 2020
num_years = final_year - initial_year

# List of season names
seasons = ['JJA', 'MAM', 'SON', 'DJF', False]

# List of RGs
RGs = ['1', '2', '3', 'all'] if analysis_type == 'BY_RG-all' else ['all']

for RG in RGs:

    for season in seasons:

        num_time = 3 * num_years if season else 12

        tracks, cyclone_ids = get_tracks(RG, season)

        tracks_with_periods = pd.DataFrame(columns = tracks.columns)
        for cyclone_id in cyclone_ids:
            tmp = process_track(cyclone_id, tracks, periods_directory, filter_residual=False)
            tracks_with_periods = pd.concat([tracks_with_periods, tmp])
        tracks_with_periods.reset_index(drop=True, inplace=True)

        # Initialize an empty dictionary to hold the DataArrays
        data_dict = {}

        for phase in tracks_with_periods['period'].unique():
            if str(phase) == 'nan':
                continue
            print(f'Computing density for {phase}...')
            density, lon, lat = compute_density(tracks_with_periods[tracks_with_periods['period'] == phase], num_time)
            
            # Create DataArray
            data = xr.DataArray(density, coords={'lon': lon, 'lat': lat}, dims=['lat', 'lon'])
            
            # Add the DataArray to the dictionary with the phase as the key
            data_dict[phase] = data

        dataset = xr.Dataset(data_dict)

        if analysis_type == 'BY_RG-all':
            fname = f'{output_directory}/track_density_RG{RG}' if RG != 'all' else f'{output_directory}/track_density_all-RG'
        else:
            fname = f'{output_directory}/track_density'
        fname += f'_{season}.nc' if season else '.nc'
        
        dataset.to_netcdf(fname)
        print(f'Wrote {fname}')

