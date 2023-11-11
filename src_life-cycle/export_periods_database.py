import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from export_density_all import get_tracks
from geopy.distance import geodesic
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

ANALYSIS_TYPE = 'all'
PATH_TO_TRACKS = "../processed_tracks_with_periods/"
TRACKS_PATTERN = "ff_cyc_SAt_era5_"
SECONDS_IN_AN_HOUR = 3600

def get_tracks(year: int, month: int):
    data_folder = os.path.abspath(PATH_TO_TRACKS)  # Absolute path
    month_str = f"{month:02d}"  # Format month as two digits
    fname = f"ff_cyc_SAt_era5_{year}{month_str}.csv"
    file_path = os.path.join(data_folder, fname)

    try:
        tracks = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        logging.info(f"File not found: {file_path}")
        return None
    except pd.errors.ParserError:
        logging.info(f"Error parsing file: {file_path}")
        return None
    except Exception as e:
        logging.info(f"An error occurred: {e}")
        return None
    
    tracks['lon vor'] = np.where(tracks['lon vor'] > 180, tracks['lon vor'] - 360, tracks['lon vor'])
    tracks.sort_values(by=['track_id', 'date'], inplace=True, kind='mergesort')
    tracks['date'] = pd.to_datetime(tracks['date'])
    return tracks

def filter_tracks(tracks):
    """
    Subset corresponding tracks for analysis type
    """
    periods_directory = os.path.join('..', 'periods-energetics', ANALYSIS_TYPE)
    csv_files = glob(os.path.join(periods_directory, '*.csv'))
    valid_ids = [int(os.path.basename(csv_file).split('.')[0].split('_')[1]) for csv_file in csv_files] + [os.path.basename(csv_file).split('.')[0].split('_')[1] for csv_file in csv_files]
    tracks = tracks[tracks['track_id'].isin(valid_ids)]
    return tracks

def calculate_distances_per_phase(tracks):
    # Group by track_id and phase, then extract first and last positions
    first_positions = tracks.groupby(['track_id', 'phase']).first().reset_index()[['track_id', 'phase', 'lat vor', 'lon vor']]
    last_positions = tracks.groupby(['track_id', 'phase']).last().reset_index()[['track_id', 'phase', 'lat vor', 'lon vor']]

    # Calculating straight-line distance for each cyclone and phase
    straight_line_distances = []
    for (first, last) in zip(first_positions.iterrows(), last_positions.iterrows()):
        if first[1]['track_id'] == last[1]['track_id'] and first[1]['phase'] == last[1]['phase']:
            start_coords = (first[1]['lat vor'], first[1]['lon vor'])
            end_coords = (last[1]['lat vor'], last[1]['lon vor'])
            distance = geodesic(start_coords, end_coords).kilometers
            straight_line_distances.append({
                'track_id': first[1]['track_id'], 
                'phase': first[1]['phase'], 
                'Straight Line Distance (km)': distance
            })

    straight_line_distance_df = pd.DataFrame(straight_line_distances)

    return straight_line_distance_df

def calculate_distances(tracks):
    # Extracting first and last positions for each cyclone
    first_positions = tracks.groupby('track_id').first().reset_index()[['track_id', 'lat vor', 'lon vor']]
    last_positions = tracks.groupby('track_id').last().reset_index()[['track_id', 'lat vor', 'lon vor']]

    # Calculating straight-line distance for each cyclone
    straight_line_distances = []
    for (first, last) in zip(first_positions.iterrows(), last_positions.iterrows()):
        start_coords = (first[1]['lat vor'], first[1]['lon vor'])
        end_coords = (last[1]['lat vor'], last[1]['lon vor'])
        distance = geodesic(start_coords, end_coords).kilometers
        straight_line_distances.append({'track_id': first[1]['track_id'], 'Straight Line Distance (km)': distance})

    straight_line_distance_df = pd.DataFrame(straight_line_distances)

    return straight_line_distance_df

def create_database(tracks_year):
    # Sort by track_id and date
    tracks_year.sort_values(by=['track_id', 'date'], inplace=True)

    # Calculate time difference in hours
    tracks_year['time_diff'] = tracks_year.groupby('track_id')['date'].diff().dt.total_seconds() / SECONDS_IN_AN_HOUR

    # Handle NaN values in time_diff (for the first entry of each track)
    tracks_year['time_diff'].fillna(0, inplace=True)

    # Calculate the Duration for each phase and for the total life cycle
    total_time_per_phase = tracks_year.groupby(['track_id', 'phase'])['time_diff'].sum().reset_index(name='Total Time (h)')
    total_time = total_time_per_phase.groupby('track_id')['Total Time (h)'].sum().reset_index()
    total_time['phase'] = 'Total'

    # Mean speed per phase and cyclone
    mean_speed_per_phase = tracks_year.groupby(['track_id', 'phase'])['Speed (m/s)'].mean().reset_index(name='Mean Speed (m/s)')
    mean_speed = mean_speed_per_phase.groupby('track_id')['Mean Speed (m/s)'].mean().reset_index()
    mean_speed['phase'] = 'Total'

    # Total distance per phase and cyclone
    total_distance_per_phase = tracks_year.groupby(['track_id', 'phase'])['Distance (km)'].sum().reset_index(name='Total Distance (km)')
    total_distance = total_distance_per_phase.groupby('track_id')['Total Distance (km)'].sum().reset_index()
    total_distance['phase'] = 'Total'

    # Mean voriticy per phase and cyclone 
    mean_vorticity_per_phase = tracks_year.groupby(['track_id', 'phase'])['vor42'].mean().reset_index(name='Mean Vorticity (−1 × 10−5 s−1)')
    mean_vorticity = mean_vorticity_per_phase.groupby('track_id')['Mean Vorticity (−1 × 10−5 s−1)'].mean().reset_index()
    mean_vorticity['phase'] = 'Total'

    # Mean growth rate per phase and cyclone
    mean_growth_rate_per_phase = tracks_year.groupby(['track_id', 'phase'])['Growth Rate (10^−5 s^−1 day^-1)'].mean().reset_index(name='Mean Growth Rate (10^−5 s^−1 day^-1)')
    mean_growth_rate = mean_growth_rate_per_phase.groupby('track_id')['Mean Growth Rate (10^−5 s^−1 day^-1)'].mean().reset_index()
    mean_growth_rate['phase'] = 'Total'

    # Calculate straight-line distance
    straight_line_distance_per_phase = calculate_distances_per_phase(tracks_year)
    straight_line_distance = calculate_distances(tracks_year)
    straight_line_distance['phase'] = 'Total'

    # Concatenate the total lifecycle data with the per-phase data
    total_time_per_phase = pd.concat([total_time_per_phase, total_time], ignore_index=True)
    mean_speed_per_phase = pd.concat([mean_speed_per_phase, mean_speed], ignore_index=True)
    total_distance_per_phase = pd.concat([total_distance_per_phase, total_distance], ignore_index=True)
    mean_vorticity_per_phase = pd.concat([mean_vorticity_per_phase, mean_vorticity], ignore_index=True)
    mean_growth_rate_per_phase = pd.concat([mean_growth_rate_per_phase, mean_growth_rate], ignore_index=True)
    straight_line_distance_per_phase = pd.concat([straight_line_distance_per_phase, straight_line_distance], ignore_index=True)

    # Merging the dataframes
    merged_data = pd.merge(total_time_per_phase, mean_speed_per_phase, on=['track_id', 'phase'], how='outer')
    merged_data = pd.merge(merged_data, total_distance_per_phase, on=['track_id', 'phase'], how='outer')
    merged_data = pd.merge(merged_data, mean_vorticity_per_phase, on=['track_id', 'phase'], how='outer')
    merged_data = pd.merge(merged_data, mean_growth_rate_per_phase, on=['track_id', 'phase'], how='outer')
    merged_data = pd.merge(merged_data, straight_line_distance_per_phase, on=['track_id', 'phase'], how='outer')

    # Extract unique track_id with Genesis Season and Genesis Region
    genesis_attributes = tracks_year[['track_id', 'Genesis Season', 'Genesis Region']].drop_duplicates()

    # Merge with merged_data
    merged_data = pd.merge(merged_data, genesis_attributes, on='track_id', how='left')

    return merged_data

def process_year_month(year, month):
    month_str = f"{month:02d}"
    logging.info(f"Processing year: {year}, month: {month_str}")
    tracks_year = get_tracks(year, month)
    if tracks_year is None:
        logging.info(f"No data available for year {year}, month {month}. Skipping...")
        return

    tracks_year = filter_tracks(tracks_year)

    # Create database if it doesn't exist
    database_path = f"../periods_species_statistics/{ANALYSIS_TYPE}/periods_database/"
    os.makedirs(database_path, exist_ok=True)
    database = os.path.join(database_path, f"periods_database_{year}{month_str}.csv")
    try:
        pd.read_csv(database + "ff")
        logging.info(f"{database} already exists.")
    except FileNotFoundError:
        logging.info(f"{database} not found, creating it...")
        merged_data_frames = create_database(tracks_year)
        merged_data_frames.to_csv(database)
        logging.info(f"{database} created.")

raw_tracks = sorted(glob(os.path.join(PATH_TO_TRACKS, f"{TRACKS_PATTERN}*.csv")))
years = np.unique([int(os.path.basename(raw_track).split(TRACKS_PATTERN)[1].split(".")[0][:4]) for raw_track in raw_tracks])

# Prepare list of year-month combinations
year_month_combinations = [(year, month) for year in years for month in range(1, 13)]

# Configure logging
logging.basicConfig(filename='export_periods_database.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use ThreadPoolExecutor for parallel processing
with ThreadPoolExecutor(max_workers=4) as executor:  # Adjust the number of workers as needed
    list(tqdm(executor.map(lambda x: process_year_month(*x), year_month_combinations), total=len(year_month_combinations)))
