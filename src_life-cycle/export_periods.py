# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_periods.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 16:45:03 by Danilo            #+#    #+#              #
#    Updated: 2023/10/16 18:51:39 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script exports the cyclone periods to a csv file in '../periods-energetics/BY_RG-all/'
for each cyclone

"""

import multiprocessing
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd 

from cyclophaser import determine_periods
from multiprocessing import Pool
from glob import glob

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
    id_cyclone, tracks, periods_outfile_path, periods_didatic_outfile_path, periods_csv_outfile_path, RG = args
    plt.close('all') # save memory

    # Set the output file names
    periods_csv_outfile = f"{periods_csv_outfile_path}{RG}_{id_cyclone}"

    track = tracks[tracks['track_id'] == id_cyclone]

    # Create temporary files for cyclophaser function
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
        "SE-BR": [(-52, -38, -37, -23)],
        "LA-PLATA": [(-69, -38, -52, -23)],
        "ARG": [(-70, -55, -50, -39)],
        "SE-SAO": [(-15, -55, 30, -37)],
        "SA-NAM": [(8, -33, 20, -21)],
        "AT-PEN": [(-65, -69, -44, -58)],
        "WEDDELL": [(-65, -85, -10, -72)]
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

def filter_tracks(tracks, analysis_type):
    print("Filtering tracks...")

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
        continent_shapefile = "ne_50m_land/ne_50m_land.shp"
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
    
    return tracks

testing = False
# analysis_type = 'BY_RG-all'
# analysis_type = 'all'
# analysis_type = '70W' 
# analysis_type = '48h'
# analysis_type = '70W-48h'
# analysis_type = '70W-1000km'
# analysis_type = '70W-1500km'
# analysis_type = '70W-decayC'
analysis_type = '70W-no-continental'

# RGs = [False]
RGs = ["SE-BR", "LA-PLATA","ARG", "SE-SAO", "SA-NAM",
               "AT-PEN", "WEDDELL", False]

print("Initializing periods analysis for: ", analysis_type) if not testing else print("Testing")

output_directory = '../figures/'
results_dir = '../raw_data/SAt/'

track_columns = ['track_id', 'date', 'lon vor', 'lat vor', 'vor42']

if __name__ == '__main__':

    # Get all tracks for SAt
    tracks = get_tracks()

    # Filter for analysis type
    tracks = filter_tracks(tracks, analysis_type)

    for RG in RGs:
        print(f"RG: {RG}") if RG else print("RG: SAt")

        RG_str = f"_{RG}" if RG else ""

        periods_outfile_path = output_directory + f'periods/{analysis_type}_{RG_str}/' 
        periods_didatic_outfile_path = output_directory + f'periods_didactic/{analysis_type}_{RG_str}/' 
        periods_csv_outfile_path = f'../periods-energetics/{analysis_type}_{RG_str}/'
        os.makedirs(periods_outfile_path, exist_ok=True)
        os.makedirs(periods_didatic_outfile_path, exist_ok=True)
        os.makedirs(periods_csv_outfile_path, exist_ok=True)

        tracks_RG = filter_tracks_area(tracks, RG) if RG else tracks

        id_cyclones = tracks_RG['track_id'].unique()

        # Create a list of arguments for the process_cyclone function
        arguments_list = [(id_cyclone, tracks_RG, periods_outfile_path, periods_didatic_outfile_path, periods_csv_outfile_path, RG) for id_cyclone in id_cyclones]

        # Use multiprocessing Pool to execute the function in parallel
        with multiprocessing.Pool() as pool:
            pool.map(process_cyclone, arguments_list)
