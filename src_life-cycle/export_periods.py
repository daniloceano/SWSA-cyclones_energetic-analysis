# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_periods.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 16:45:03 by Danilo            #+#    #+#              #
#    Updated: 2023/10/03 14:54:59 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script exports the cyclone periods to a csv file in '../periods-energetics/BY_RG-all/'
for each cyclone

"""

import glob
import multiprocessing
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd 

from cyclophaser import determine_periods


def haversine(lon1, lat1, lon2, lat2):
    earth_radius_km = 6371.0

    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = earth_radius_km * c
    return distance

def calculate_distance(cyclone_id, track_data):
    track = track_data[track_data['track_id'] == cyclone_id].copy()
    track['date'] = pd.to_datetime(track['date'])
    track['distance'] = haversine(track['lon vor'].shift(), track['lat vor'].shift(), track['lon vor'], track['lat vor'])
    return track

def check_last_position_on_continent(cyclone_id, tracks, continent_gdf):
    cyclone_track = tracks[tracks['track_id'] == cyclone_id]
    last_position = cyclone_track.tail(1)  # Get the last row
    last_position_on_continent = gpd.points_from_xy(last_position['lon vor'], last_position['lat vor'])
    last_position_on_continent = gpd.GeoSeries(last_position_on_continent, crs=continent_gdf.crs)
    last_position_on_continent = last_position_on_continent.within(continent_gdf.unary_union)
    return cyclone_id, not last_position_on_continent.all()

def check_on_continent_percentage(cyclone_id, tracks, continent_gdf, threshold_percentage):
    cyclone_track = tracks[tracks['track_id'] == cyclone_id]
    
    # Calculate the number of time steps where the cyclone is on the continent
    positions_on_continent = gpd.points_from_xy(cyclone_track['lon vor'], cyclone_track['lat vor'])
    positions_on_continent = gpd.GeoSeries(positions_on_continent, crs=continent_gdf.crs)
    positions_on_continent = positions_on_continent.within(continent_gdf.unary_union)
    on_continent_count = positions_on_continent.sum()
    
    # Calculate the total number of time steps in the cyclone's lifetime
    total_time_steps = len(cyclone_track)
    
    # Calculate the percentage of time on the continent
    percentage_on_continent = (on_continent_count / total_time_steps) * 100
    
    return cyclone_id, percentage_on_continent < threshold_percentage

def process_cyclone(args):
    id_cyclone, track_file, periods_outfile_path, periods_didatic_outfile_path, periods_csv_outfile_path, RG = args
    plt.close('all') # save memory

    # Set the output file names
    periods_csv_outfile = f"{periods_csv_outfile_path}{RG}_{id_cyclone}"
    periods_outfile = f"{periods_outfile_path}{RG}_{id_cyclone}"
    periods_didatic_outfile = f"{periods_didatic_outfile_path}{RG}_{id_cyclone}"

    # Create temporary files for cyclophaser function
    tracks = pd.read_csv(track_file)
    tracks.columns = track_columns
    track = tracks[tracks['track_id']==id_cyclone][['date','vor42']]
    track = track.rename(columns={"date":"time"})
    track['vor42'] = - track['vor42'] * 1e-5
    tmp_file = (f"tmp_{RG}-{id_cyclone}.csv")
    track.to_csv(tmp_file, index=False, sep=';')

    # if not periods_outfile_exists:
    options = {
        "vorticity_column": 'vor42',
        "plot": False,
        "plot_steps": False,
        "export_dict": periods_csv_outfile,
        "process_vorticity_args": {
            "use_filter": False,
            "use_smoothing_twice": "auto"}
    }

    try:
        determine_periods(tmp_file, **options)
            
    except Exception as e:
        error_msg = str(e)
        with open("error_log.txt", "a") as file:
            file.write(f"Error processing cyclone: {id_cyclone} - RG: {RG} in track_file: {track_file}\n")
            file.write(f"Error message: {error_msg}\n\n")

    os.remove(tmp_file)

def filter_tracks(tracks, analysis_type):
    print("Filtering tracks...")

    RG = analysis_type

    # Initialize a message to keep track of filtering actions
    filter_message = ""

    # Filter the 'tracks' DataFrame to keep only the systems that are west of 70W
    if '70W' in analysis_type:
        ids_west_70 = tracks[tracks['lon vor'] < -70]['track_id'].unique()
        tracks = tracks[~tracks['track_id'].isin(ids_west_70)]
        filter_message += "Removed systems west of 70W."

    # Filter the 'tracks' DataFrame to keep only the systems that have a duration of at least 48 hours
    if '48h' in analysis_type:
        tracks['date'] = pd.to_datetime(tracks['date'])
        grouped = tracks.groupby('track_id')

        # Calculate the duration of each system in hours
        system_durations = grouped['date'].max() - grouped['date'].min()
        system_durations = system_durations.dt.total_seconds() / 3600  # Convert to hours

        # Select only systems with a duration of at least 48 hours
        min_duration_hours = 48
        valid_track_ids = system_durations[system_durations >= min_duration_hours].index

        # Filter the 'tracks' DataFrame to keep only the systems that meet the duration criteria
        tracks = tracks[tracks['track_id'].isin(valid_track_ids)]
        filter_message += " Removed systems with less than 48 hours duration."

    if 'km' in analysis_type:
        minimum_allowed_distance = float(analysis_type.split('-')[-1].split('km')[0])
        # Calculating distance that cyclone traveled
        tracks = tracks.assign(distance=np.nan)

        id_cyclones = tracks['track_id'].unique()

        # Create a list of arguments for the calculate_distance function
        arguments_list = [(id_cyclone, tracks) for id_cyclone in id_cyclones]

        # Use multiprocessing Pool to execute the function in parallel
        with multiprocessing.Pool() as pool:
            result_tracks = pool.starmap(calculate_distance, arguments_list)

        # Combine the results into a single DataFrame
        tracks = pd.concat(result_tracks, ignore_index=True)
        
        # Calculate total distance for each system
        total_distance = tracks.groupby('track_id')['distance'].sum()

        # Filter out systems with total distance greater than 1000 km
        filtered_track_ids = total_distance[total_distance >= minimum_allowed_distance].index

        # Apply the filter to the original DataFrame
        tracks = tracks[tracks['track_id'].isin(filtered_track_ids)]
        filter_message += f" Removed systems with less than {minimum_allowed_distance} km total distance."

    if 'decayC' in analysis_type:
        # Load a shapefile or GeoDataFrame representing the continent boundaries
        continent_shapefile = "ne_50m_land/ne_50m_land.shp"
        continent_gdf = gpd.read_file(continent_shapefile)

        # Create a list of unique cyclone IDs
        unique_cyclone_ids = tracks['track_id'].unique()

        # Create a pool for multiprocessing
        with multiprocessing.Pool() as pool:
            results = pool.starmap(check_last_position_on_continent, [(cyclone_id, tracks, continent_gdf) for cyclone_id in unique_cyclone_ids])

        # Extract valid cyclone IDs from the results
        valid_track_ids = [cyclone_id for cyclone_id, is_valid in results if is_valid]

        # Filter the 'tracks' DataFrame to keep only cyclones not ending on the continent
        tracks = tracks[tracks['track_id'].isin(valid_track_ids)]
        filter_message += "Removed cyclones with their last positions on the continent."

    if  'no-continental' in analysis_type:
        # Load a shapefile or GeoDataFrame representing the continent boundaries
        continent_shapefile = "path_to_continent_shapefile.shp"
        continent_gdf = gpd.read_file(continent_shapefile)

        # Calculate the threshold percentage (80%)
        threshold_percentage = 80

        # Create a list of unique cyclone IDs
        unique_cyclone_ids = tracks['track_id'].unique()

        # Create a pool for multiprocessing
        with multiprocessing.Pool() as pool:
            results = pool.starmap(check_on_continent_percentage, [(cyclone_id, tracks, continent_gdf, threshold_percentage) for cyclone_id in unique_cyclone_ids])

        # Extract valid cyclone IDs from the results
        valid_track_ids = [cyclone_id for cyclone_id, is_valid in results if is_valid]

        # Filter the 'tracks' DataFrame to keep only cyclones not exceeding the threshold percentage on the continent
        tracks = tracks[tracks['track_id'].isin(valid_track_ids)]
        filter_message += f"Removed cyclones on the continent for {threshold_percentage}% or more of their lifetime."

    # Print the final filter message
    print(filter_message)
    
    return tracks, RG

testing = False
# analysis_type = 'BY_RG-all'
# analysis_type = 'all'
# analysis_type = '70W' 
# analysis_type = '48h'
# analysis_type = '70W-48h'
# analysis_type = '70W-1000km'
# analysis_type = '70W-1500km'
# analysis_type = '70W-decayC'
analysis_type = '70W-no-continent'

print("Initializing periods analysis for: ", analysis_type) if not testing else print("Testing")

if testing == True:
    output_directory = './'
    periods_outfile_path = output_directory + './'    
    periods_didatic_outfile_path = output_directory + './'
    periods_csv_outfile_path = './'
    results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/']

else:
    output_directory = '../figures/'
    periods_outfile_path = output_directory + f'periods/{analysis_type}/'    
    periods_didatic_outfile_path = output_directory + f'periods_didactic/{analysis_type}/'
    periods_csv_outfile_path = f'../periods-energetics/{analysis_type}/'

    if analysis_type == 'BY_RG-all':
        track_columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']
        results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
                        '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
                        '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/']
    
    else:
        track_columns = ['track_id', 'date', 'lon vor', 'lat vor', 'vor42']
        results_directories = ['../raw_data/SAt/']


os.makedirs(periods_outfile_path, exist_ok=True)
os.makedirs(periods_didatic_outfile_path, exist_ok=True)
os.makedirs(periods_csv_outfile_path, exist_ok=True)

if __name__ == '__main__':

    for results_dir in results_directories:
        for track_file in sorted(glob.glob(f'{results_dir}/*')):

            if testing:
                if '1980' not in track_file:
                    continue

            # Check if the track_file is empty
            try:
                tracks = pd.read_csv(track_file)
            except pd.errors.EmptyDataError:
                with open("error_log.txt", "a") as file:
                    file.write(f"Empty track file: {track_file} - Skipping processing.\n")
                continue

            # Check if track_file contains "40W" and skip processing if it does
            if "40W" in track_file:
                with open("error_log.txt", "a") as file:
                    file.write(f"Skipping track file: {track_file} - Contains '40W'.\n")
                continue
            
            tracks.columns = track_columns

            # Parameters for each type of analysis
            if analysis_type == 'BY_RG-all':
                if 'RG1' in track_file:
                    RG = 'RG1'
                elif 'RG2' in track_file:
                    RG = 'RG2'
                elif 'RG3' in track_file:
                    RG = 'RG3'

            elif analysis_type == 'all':
                RG = 'SAt'

            else:
                tracks, RG = filter_tracks(tracks, analysis_type)

            id_cyclones = tracks['track_id'].unique()

            # Create a list of arguments for the process_cyclone function
            arguments_list = [(id_cyclone, track_file, periods_outfile_path, periods_didatic_outfile_path, periods_csv_outfile_path, RG) for id_cyclone in id_cyclones]

            # Use multiprocessing Pool to execute the function in parallel
            with multiprocessing.Pool() as pool:
                pool.map(process_cyclone, arguments_list)
