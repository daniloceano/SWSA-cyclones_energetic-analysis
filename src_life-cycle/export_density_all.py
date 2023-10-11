# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_density_all.py                              :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/09 12:48:17 by Danilo            #+#    #+#              #
#    Updated: 2023/10/11 09:21:05 by Danilo           ###   ########.fr        #
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
import multiprocessing

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

def check_first_position_inside_area(cyclone_id, tracks, area_bounds):
    cyclone_track = tracks[tracks['track_id'] == cyclone_id]
    first_position = cyclone_track.head(1)  # Get the first row
    first_lat = first_position['lat vor'].values[0]
    first_lon = first_position['lon vor'].values[0]

    min_lon, min_lat, max_lon, max_lat = area_bounds

    # Check if the first position is inside the specified area
    is_inside_area = (min_lat <= first_lat <= max_lat) and (min_lon <= first_lon <= max_lon)

    return cyclone_id, is_inside_area

def filter_tracks_area(tracks, region):
    print(f"Filtering tracks for region: {region}...")

    regions = {
        "SE-BR": [(-37, -23, -52, -38)],
        "LA-PLATA": [(-52, -23, -69, -38)],
        "ARG": [(-50, -39, -70, -55)],
        "SE-SAO": [(30, -37, -15, -55)],
        "SA-NAM": [(20, -21, 8, -33)],
        "AT-PEN": [(-44, -58, -65, -69)],
        "WEDDELL": [(-10, -72, -65, -85)]
    }

    if region not in regions:
        raise ValueError(f"Invalid region '{region}'. Region must be one of: {', '.join(regions.keys())}")

    region_bounds = regions[region]

    # Extract the correct region bounds
    area_bounds = region_bounds[0]  # Assuming there's only one set of bounds for the selected region

    # Count the number of systems before filtering
    num_systems_before = len(tracks['track_id'].unique())

    # Create a list of unique cyclone IDs
    unique_cyclone_ids = tracks['track_id'].unique()

    # Create a pool for multiprocessing
    with multiprocessing.Pool() as pool:
        results = pool.starmap(check_first_position_inside_area, [(cyclone_id, tracks, area_bounds) for cyclone_id in unique_cyclone_ids])

    # Extract valid cyclone IDs from the results
    valid_track_ids = [cyclone_id for cyclone_id, is_valid in results if is_valid]

    # Filter the 'tracks' DataFrame to keep only cyclones with the first position inside the defined area
    filtered_tracks = tracks[tracks['track_id'].isin(valid_track_ids)]

    # Count the number of systems after filtering
    num_systems_after = len(filtered_tracks['track_id'].unique())

    # Print the final filter message and the number of systems before and after filtering
    print(f"Removed cyclones with the first position inside the defined area.")
    print(f"Number of systems before filtering: {num_systems_before}")
    print(f"Number of systems after filtering: {num_systems_after}")

    return filtered_tracks

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
        if pd.notna(phase):
            print(f'Computing density for {phase}...')
            density, lon, lat = compute_density(season_tracks[season_tracks['period'] == phase], num_time)

            # Handle the 'nan' case by using 'no_data' as the name
            name = str(phase) if pd.notna(phase) else 'no_data'
            
            # Create DataArray with the appropriate name
            data = xr.DataArray(density, coords={'lon': lon, 'lat': lat}, dims=['lat', 'lon'], name=name)
            
            # Add the DataArray to the dictionary with the name as the key
            data_dict[name] = data

    return data_dict

def main(analysis_type):

    print(f"Analysis type: {analysis_type}")
    analysis_type = '70W-no-continental'

    periods_directory = f'../periods-energetics/{analysis_type}/'
    output_directory = f'../periods_species_statistics/{analysis_type}/track_density'
    os.makedirs(output_directory, exist_ok=True)

    tracks = get_tracks()
    tracks = filter_tracks_area(tracks, "SE-BR")
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

    for season in seasons:
        # num_time = 3 * num_years if season else 12
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
        
        data_dict = export_density(season_tracks, num_time)

        dataset = xr.Dataset(data_dict)

        fname = f'{output_directory}/track_density'
        fname += f'_{season}.nc' if season else '.nc'
        dataset.to_netcdf(fname)
        print(f'Wrote {fname}')

if __name__ == '__main__':
#     analysis_types = [
#     'all',
#     '70W',
#     '48h',
#     '70W-48h',
#     '70W-1000km',
#     '70W-1500km',
#     '70W-decayC',
#     '70W-no-continental'
# ]

    main()
