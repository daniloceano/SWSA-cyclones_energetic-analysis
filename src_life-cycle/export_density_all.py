# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_density_parallel_v2.py                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/09 12:48:17 by Danilo            #+#    #+#              #
#    Updated: 2023/10/06 20:18:55 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Adapted from Carolina B. Gramcianinov (cbgramcianinov@gmail.com)

Reads the raw tracks and the processed periods for each system and produces netCDF files
containing the density map of each period in  2.5 degree global grid
"""

import os
import pandas as pd
import xarray as xr
import numpy as np

import concurrent.futures
from concurrent.futures import ProcessPoolExecutor 

from datetime import timedelta
from tqdm import tqdm
from glob import glob
from multiprocessing import Pool
from sklearn.neighbors import KernelDensity

def read_csv_file(file):
    return pd.read_csv(file, header=None)

def get_tracks():
    print(f"Reading raw track files...")
    file_list = glob("../raw_data/SAt/*.csv")
    with Pool() as pool:
        dfs = pool.map(read_csv_file, file_list)
    print("Done, now merging tracks...")
    tracks = pd.concat(dfs, ignore_index=True)
    print(f"Done.")

    track_columns = ['track_id', 'date', 'lon vor', 'lat vor', 'vor42']
    tracks.columns = track_columns
    tracks['lon vor'] = np.where(tracks['lon vor'] > 180, tracks['lon vor'] - 360, tracks['lon vor'])
    return tracks

def process_period_file(args):
    period_file, periods_directory, tracks = args
    periods = pd.read_csv(os.path.join(periods_directory, period_file))
    periods.columns = ['period', 'start', 'end']
    periods = periods.set_index('period')

    tracks_id = int(period_file.split('_')[-1].split('.csv')[0])
    track = tracks[tracks['track_id'] == tracks_id].copy()
    track['date'] = pd.to_datetime(track['date'])

    dt = timedelta(hours=1)
    track_periods = track[['track_id', 'date']].copy()
    track_periods['period'] = np.nan

    for phase in list(periods.index):
        periods.loc[phase] = pd.to_datetime(periods.loc[phase])
        period_dates = pd.date_range(start=periods.loc[phase][0], end=periods.loc[phase][1], freq=dt)
        track_periods.loc[track_periods['date'].isin(period_dates), 'period'] = phase

    return track_periods

def get_periods(analysis_type, periods_directory, tracks):
    print(f"Merging periods for {analysis_type}...")

    period_files = sorted(os.listdir(periods_directory))
    # period_files = period_files[:100]

    arguments = [(period_file, periods_directory, tracks) for period_file in period_files]

    processed_files = 0
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_period_file, arguments), total=len(period_files)))
        processed_files += 1

    final_df = pd.concat(results, ignore_index=True)
    final_df = final_df.sort_values(by=['track_id', 'date']).reset_index(drop=True)
    print("Done.")
    return final_df

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

def export_density(season_tracks, num_time):
    data_dict = {}

    for phase in season_tracks['period'].unique():
        if str(phase) == 'nan':
            continue
        print(f'Computing density for {phase}...')
        density, lon, lat = compute_density(season_tracks[season_tracks['period'] == phase], num_time)
        
        # Create DataArray
        data = xr.DataArray(density, coords={'lon': lon, 'lat': lat}, dims=['lat', 'lon'])
        
        # Add the DataArray to the dictionary with the phase as the key
        data_dict[phase] = data
    return data_dict

def compute_density_for_phase(phase, season_tracks, num_time):
    if str(phase) == 'nan':
        return None, None, None  # Skip NaN phases
    print(f'Computing density for {phase}...')

    # Filter tracks for the current phase
    phase_tracks = season_tracks[season_tracks['period'] == phase]

    density, lon, lat = compute_density(phase_tracks, num_time)

    return phase, density, lon, lat

def main():
    # analysis_type = 'all'
    # analysis_type = '70W'
    # analysis_type = '48h'
    # analysis_type = '70W-48h'
    # analysis_type = '70W-1000km'
    # analysis_type = '70W-1500km'
    # analysis_type = '70W-decayC'
    analysis_type = '70W-no-continental'
    periods_directory = f'../periods-energetics/{analysis_type}/'
    output_directory = f'../periods_species_statistics/{analysis_type}/track_density'
    os.makedirs(output_directory, exist_ok=True)

    tracks = get_tracks()
    periods = get_periods(analysis_type, periods_directory, tracks)
    print(f"Periods and tracks have been obtained.")

    # Filter tracks for the track_ids in periods and reset the index
    filtered_tracks = tracks[tracks['track_id'].isin(periods['track_id'])].reset_index(drop=True)

    # Create a dictionary mapping 'track_id' and 'date' to 'period'
    period_mapping = periods.set_index(['track_id', 'date'])['period'].to_dict()

    # Add the 'period' column to filtered_tracks based on 'track_id' and 'date' mapping
    filtered_tracks['period'] = filtered_tracks.set_index(['track_id', 'date']).index.map(period_mapping)

    filtered_tracks['date'] = pd.to_datetime(filtered_tracks['date'])

    seasons = ['JJA', 'MAM', 'SON', 'DJF', False]

    with ProcessPoolExecutor() as executor:
        futures = []
        for season in seasons:
            if season:
                if season == 'JJA':
                    season_months = [6, 7, 8]  # June, July, August
                elif season == 'MAM':
                    season_months = [3, 4, 5]  # March, April, May
                elif season == 'SON':
                    season_months = [9, 10, 11]  # September, October, November
                elif season == 'DJF':
                    season_months = [12, 1, 2]  # December, January, February
                # Filter tracks for the specific months of the season
                season_tracks = filtered_tracks[filtered_tracks['date'].dt.month.isin(season_months)]
            else:
                season_tracks = filtered_tracks

            unique_years_months = season_tracks['date'].dt.to_period('M').unique()
            num_time = len(unique_years_months)
            print(f"Total number of time months: {num_time}")

            # Use the executor to parallelize the KDE computation
            futures.append(executor.submit(compute_density_for_phase, season, season_tracks, num_time))

        data_dict = {}
        for future in futures:
            phase, density, lon, lat = future.result()
            if phase is not None:
                data = xr.DataArray(density, coords={'lon': lon, 'lat': lat}, dims=['lat', 'lon'])
                data_dict[phase] = data

        dataset = xr.Dataset(data_dict)

        # Save the results to NetCDF files
        for season in seasons:
            fname = f'{output_directory}/track_density'
            fname += f'_{season}.nc' if season else '.nc'
            dataset.to_netcdf(fname)
            print(f'Wrote {fname}')

if __name__ == '__main__':
    main()
