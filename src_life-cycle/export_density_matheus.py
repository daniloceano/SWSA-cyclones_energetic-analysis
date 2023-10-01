# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_density.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/09 12:48:17 by Danilo            #+#    #+#              #
#    Updated: 2023/10/01 11:50:47 by Danilo           ###   ########.fr        #
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
from export_periods import filter_tracks

earth_radius_km = 6371.0

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2]) 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = earth_radius_km * c
    return distance

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
        str_RG = f'{analysis_type} systems'
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

    tracks, _ = filter_tracks(tracks, analysis_type)

    cyclone_ids = tracks['track_id'].unique()
    print(f"Number of cyclones for {str_RG}: {len(cyclone_ids)}")

    # Calculating distance that cyclone traveled
    tracks['distance'] = np.nan

    for cyclone_id in cyclone_ids:
        print(f'cyclone_id: {cyclone_id}')
        track = tracks[tracks['track_id'] == cyclone_id].copy()
        track['date'] = pd.to_datetime(track['date'])
        track['distance'] = haversine(track['lon vor'].shift(), track['lat vor'].shift(), track['lon vor'], track['lat vor'])
        tracks['distance'].loc[tracks['track_id'] == cyclone_id] = track['distance']


    return tracks, cyclone_ids

RG = '1'
analysis_type = '70W-48h'
season = False


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
    str_RG = f'{analysis_type} systems'
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

tracks, _ = filter_tracks(tracks, analysis_type)

cyclone_ids = tracks['track_id'].unique()
print(f"Number of cyclones for {str_RG}: {len(cyclone_ids)}")

# Calculating distance that cyclone traveled
tracks['distance'] = np.nan

for cyclone_id in cyclone_ids[0:1]:
    track = tracks[tracks['track_id'] == cyclone_id].copy()
    track['date'] = pd.to_datetime(track['date'])
    track['distance'] = haversine(track['lon vor'].shift(), track['lat vor'].shift(), track['lon vor'], track['lat vor'])
    tracks['distance'].loc[tracks['track_id'] == cyclone_id] = track['distance']




