# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    expoert_density_v3.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/09 12:48:17 by Danilo            #+#    #+#              #
#    Updated: 2023/10/02 18:03:12 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Adapted from Carolina B. Gramcianinov (cbgramcianinov@gmail.com)

Reads the raw tracks and the processed periods for each system and produces netCDF files
containing the density map of each period in  2.5 degree global grid
"""

import os
import sys 
import cProfile
import concurrent.futures
import multiprocessing

import numpy as np
import pandas as pd
import xarray as xr

from sklearn.neighbors import KernelDensity

from glob import glob

from export_periods import filter_tracks


def get_tracks(RG, analysis_type, season=False):
    season_message = f" and season: {season}" if season else ''
    print(f"Merging tracks for RG: {RG}{season_message}")

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
            results_directories = [
                '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
                '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
                '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/'
            ]
    else:
        track_columns = ['track_id', 'date', 'lon vor', 'lat vor', 'vor42']
        str_RG = f'{analysis_type} systems'
        results_directories = ['../raw_data/SAt/']

    tracks = pd.DataFrame(columns=track_columns)

    def process_file(file, processed_files):
        try:
            tmp = pd.read_csv(file)
            tmp.columns = track_columns
            processed_files[0] += 1
            sys.stdout.write(f'\rProcessed: {processed_files[0]} out of {processed_files[1]} files')
            sys.stdout.flush()
            return tmp
        except Exception as e:
            print(f"Error processing file {file}: {e}")
            return None

    def process_directory(results_directory):
        directory_tracks = pd.DataFrame(columns=track_columns)
        files = glob(f'{results_directory}*')
        processed_files = [0, len(files)]  # Keep track of processed files
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_file, file, processed_files): file for file in files}
            for future in concurrent.futures.as_completed(futures):
                file = futures[future]
                try:
                    tmp = future.result()
                    if tmp is not None:
                        if season:
                            system_end = pd.to_datetime(tmp['date'].iloc[-1])
                            system_month = system_end.month
                            corresponding_season = month_season_map[system_month]
                            if corresponding_season == season:
                                directory_tracks = pd.concat([directory_tracks, tmp])
                        else:
                            directory_tracks = pd.concat([directory_tracks, tmp])
                except Exception as e:
                    print(f"Error processing file {file}: {e}")
        print('')
        return directory_tracks

    with concurrent.futures.ThreadPoolExecutor() as executor:
        all_directory_tracks = list(executor.map(process_directory, results_directories))

    for directory_tracks in all_directory_tracks:
        tracks = pd.concat([tracks, directory_tracks])

    x = tracks['lon vor'].values
    tracks['lon vor'] = np.where(x > 180, x - 360, x)

    tracks, _ = filter_tracks(tracks, analysis_type)

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

def process_track_parallel(cyclone_id, tracks, periods_directory, filter_residual=False):
    return process_track(cyclone_id, tracks, periods_directory, filter_residual)

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

def process_track_parallel(args):
    cyclone_id, tracks, periods_directory, filter_residual = args
    return process_track(cyclone_id, tracks, periods_directory, filter_residual)

def main():
    # analysis_type = 'BY_RG-all'
    # analysis_type = 'all'
    # analysis_type = '70W'
    # analysis_type = '48h'
    # analysis_type = '70W-48h'
    # analysis_type = '70W-1000km'
    analysis_type = '70W-1500km'

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

            tracks, cyclone_ids = get_tracks(RG, analysis_type, season)

            # print('Filtering tracks...')
            # tracks, _ = filter_tracks(tracks, analysis_type)

            print('Starting to process individual tracks...')

            tracks_with_periods = pd.DataFrame(columns = tracks.columns)

            processed_cyclones = 0
            tracks_with_periods = pd.DataFrame(columns=tracks.columns)
            arguments_list = [(cyclone_id, tracks, periods_directory, False) for cyclone_id in cyclone_ids]

            with multiprocessing.Pool() as pool:
                results = pool.map(process_track_parallel, arguments_list)

            for result in results:
                if result is not None:
                    tracks_with_periods = pd.concat([tracks_with_periods, result])
                    processed_cyclones += 1
                    sys.stdout.write(f'\rProcessed: {processed_cyclones}/{len(cyclone_ids)}')
                    sys.stdout.flush()        
            print('')  # Print a newline to separate the progress indicator
            
            # Reset the index and inplace
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

if __name__ == "__main__":
    cProfile.run("main()", sort="cumulative")
