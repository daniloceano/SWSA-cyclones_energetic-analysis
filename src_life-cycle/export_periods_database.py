# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_periods_database.py                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <daniloceano@student.42.fr>    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/27 19:48:00 by Danilo            #+#    #+#              #
#    Updated: 2023/11/09 19:59:03 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import os
import glob
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from export_density_all import get_tracks
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

SECONDS_IN_AN_HOUR = 3600

def haversine_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the Haversine distance between two lat-long points in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def process_csv_file(csv_file, tracks):
    """
    Process a single CSV file and update the tracks dataframe.
    """
    df_phases = pd.read_csv(csv_file, parse_dates=['start', 'end'], index_col=0)
    updates = {}
    for phase, row in df_phases.iterrows():
        mask = (tracks['date'] >= row['start']) & (tracks['date'] <= row['end'])
        updates[phase] = mask
    return updates

def filter_csv_file(f, track_id_set):
    return any(track_id in f for track_id in track_id_set)

def map_filter_func(args):
    f, track_id_set = args
    return f, filter_csv_file(f, track_id_set)

def get_filtered_csv_files_parallel(csv_files, track_ids, year):
    csv_files = [f for f in csv_files if str(year) in os.path.basename(f)]
    track_id_set = set(map(str, track_ids))
    filtered_files = []
    print("Starting the filtering process...")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        # Using tqdm to display a progress bar
        for f, included in tqdm(executor.map(map_filter_func, [(f, track_id_set) for f in csv_files]), total=len(csv_files), desc='Filtering CSV Files'):
            if included:
                filtered_files.append(f)
    print(f"Filtering completed. {len(filtered_files)} files matched out of {len(csv_files)} total files.")
    return filtered_files


def process_phase_data_parallel(tracks, data_path):
    """
    Process phase data in parallel.
    """
    print("Reading periods...")
    csv_files = glob.glob(os.path.join(data_path, '*.csv'))
    track_ids = tracks['track_id'].unique()
    year = tracks['year'].unique()[0]
    filtered_csv_files = get_filtered_csv_files_parallel(csv_files, track_ids, year)
    tracks['date'] = pd.to_datetime(tracks['date'])    
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(lambda csv_file: process_csv_file(csv_file, tracks),
                                          filtered_csv_files), total=len(filtered_csv_files),
                                            desc='Processing Files'))
    for result in results:
        for phase, mask in result.items():
            tracks.loc[mask, 'phase'] = phase
    print("Done.")
    return tracks

def compute_distance_chunk(chunk):
    chunk['previous_lon'] = chunk.groupby('track_id')['lon vor'].shift(1)
    chunk['previous_lat'] = chunk.groupby('track_id')['lat vor'].shift(1)
    chunk['Distance'] = haversine_distance(
        chunk['previous_lat'], 
        chunk['previous_lon'], 
        chunk['lat vor'], 
        chunk['lon vor']
    )
    chunk.drop(columns=['previous_lon', 'previous_lat'], inplace=True)
    return chunk

def compute_distance_parallel(tracks_df, num_workers=None):
    print("Computing distance..")
    if num_workers is None:
        num_workers = os.cpu_count()
    chunks = np.array_split(tracks_df, num_workers)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        processed_chunks = list(tqdm(executor.map(compute_distance_chunk, chunks), total=num_workers, desc="Processing Chunks"))
    print("Done.")
    return pd.concat(processed_chunks)

def filter_seasons(tracks, seasons):
    """
    Filter the tracks to include only those occurring in the provided seasons.
    """
    print("Filtering seasons...")
    tracks.loc[:, 'month'] = pd.to_datetime(tracks['date']).dt.month
    if seasons == 'DJF':
        tracks_season =  tracks[tracks['month'].isin([12, 1, 2])]
    elif seasons == 'JJA':
        tracks_season = tracks[tracks['month'].isin([6, 7, 8])]
    else:
        tracks_season = tracks
    print("Done.")
    return tracks_season

def process_data(tracks_distance_periods):
    """
    Process data for analysis.
    """
    print("Processing data...")
    # Remove whenever the phase is NaN
    tracks_distance_periods = tracks_distance_periods[~pd.isna(tracks_distance_periods['phase'])]

    # First, sort the dataframe by track_id and date to ensure the order is correct
    tracks_distance_periods.sort_values(by=['track_id', 'date'], inplace=True)

    # Set Distance to NaN for the first occurrence of each phase for each track_id
    tracks_distance_periods['is_first_occurrence'] = ~tracks_distance_periods.duplicated(subset=['track_id', 'phase'])
    tracks_distance_periods.loc[tracks_distance_periods['is_first_occurrence'], 'Distance'] = np.nan
    tracks_distance_periods.drop(columns=['is_first_occurrence'], inplace=True)  # Clean up the temporary column

    # Time difference in hours for each track
    tracks_distance_periods['time_diff'] = tracks_distance_periods.groupby('track_id')['date'].diff().dt.total_seconds() / SECONDS_IN_AN_HOUR
    
    # Calculate the Maximum Distance (km) for each phase
    # Define a function to calculate the distance between the first and last point of each phase
    def calculate_max_distance(group):
        if len(group) > 1:
            return haversine_distance(group.iloc[-1]['lon vor'], group.iloc[-1]['lat vor'],
                                      group.iloc[0]['lon vor'], group.iloc[0]['lat vor'])
        else:
            return 0

    # Apply the function to each group
    max_distance = tracks_distance_periods.groupby(['track_id', 'phase']).apply(calculate_max_distance).reset_index(name='Maximum Distance (km)')

    # Total distance per phase
    total_distance = tracks_distance_periods.groupby(['track_id', 'phase'])['Distance'].sum().reset_index(name='Total Distance (km)')
    
    # Total time per phase
    total_time = tracks_distance_periods.groupby(['track_id', 'phase'])['time_diff'].sum().reset_index(name='Total Time (h)')
    
    # Mean vorticity per phase
    mean_vorticity = tracks_distance_periods.groupby(['track_id', 'phase'])['vor42'].mean().reset_index(name='Mean Vorticity (−1 × 10−5 s−1)')
    
    # Calculate the difference in vorticity over time to find the growth rate
    # Assuming 'vor42' column is the vorticity and has units [−1 × 10^−5 s^−1]
    # Calculate the rolling 3-hour difference (3 previous rows including the current row)
    tracks_distance_periods['vor42_diff'] = tracks_distance_periods.groupby('track_id')['vor42'].diff()
    
    # Here we assume 'date' column is spaced at 1-hour intervals
    # Use a rolling window to calculate the mean over 3-hour periods
    tracks_distance_periods['vor42_3h_diff'] = tracks_distance_periods.groupby('track_id')['vor42_diff'].rolling(window=3).sum().reset_index(0,drop=True)
    
    # Compute the growth rate as the mean of these 3-hour differences
    # Dividing by 3 to convert the sum into an average over the 3-hour window
    growth_rate = (tracks_distance_periods.groupby(['track_id', 'phase'])['vor42_3h_diff'].mean() / 3).reset_index(name='Mean Growth Rate (10^−5 s^−1 3h-1)')

    # Convert the mean growth rate from per 3-hour to per day
    growth_rate['Mean Growth Rate (10^−5 s^−1 day-1)'] = growth_rate['Mean Growth Rate (10^−5 s^−1 3h-1)'] * (24 / 3)

    # Merge all the calculated metrics back into a single dataframe
    merged_df = pd.merge(total_distance, total_time, on=['track_id', 'phase'])
    merged_df = pd.merge(merged_df, mean_vorticity, on=['track_id', 'phase'])
    merged_df = pd.merge(merged_df, growth_rate, on=['track_id', 'phase'])
    merged_df = pd.merge(merged_df, max_distance, on=['track_id', 'phase'])

    # Calculate the mean speed
    merged_df['Mean Speed (km/h)'] = merged_df['Total Distance (km)'] / merged_df['Total Time (h)']
    merged_df['Mean Speed (m/s)'] = merged_df['Mean Speed (km/h)'] * (1000 / 3600)

    # Drop the temporary column used for calculations
    merged_df.drop(columns=['Mean Speed (km/h)'], inplace=True)
    tracks_distance_periods.drop(columns=['vor42_diff', 'vor42_3h_diff'], inplace=True)

    print("Done")
    return merged_df

def compute_maximum_distance_for_track(track_group):
    """
    Compute the maximum distance for a track group, which is the distance
    between the first and last positions.
    """
    if len(track_group) > 1:
        first_row = track_group.iloc[0]
        last_row = track_group.iloc[-1]
        return haversine_distance(last_row['lon vor'], last_row['lat vor'],
                                  first_row['lon vor'], first_row['lat vor'])
    else:
        return 0

def compute_totals(tracks_df, tracks_season_distance_periods):
    # Group by track_id and sum up the 'Total Distance (km)' and 'Duration'
    total_phase = tracks_df.groupby('track_id').agg({
        'Total Distance (km)': 'sum',
        'Total Time (h)': 'sum',
        'Mean Speed (m/s)': 'mean',
        'Mean Vorticity (−1 × 10−5 s−1)': 'mean',
        'Mean Growth Rate (10^−5 s^−1 day-1)': 'mean'
    }).reset_index()

    # Extract the first and last positions for each track from the original tracks dataframe
    first_positions = tracks_season_distance_periods.groupby('track_id').first().reset_index()
    last_positions = tracks_season_distance_periods.groupby('track_id').last().reset_index()

    # Compute the Maximum Distance for the "total" phase for each track
    max_distances = []
    for _, first_row in first_positions.iterrows():
        track_id = first_row['track_id']
        last_row = last_positions[last_positions['track_id'] == track_id].iloc[0]
        max_distance = compute_maximum_distance_for_track(pd.DataFrame([first_row, last_row]))
        max_distances.append(max_distance)

    # Add a new column 'Maximum Distance (km)' to total_phase
    total_phase['Maximum Distance (km)'] = max_distances

    # Add a new column 'phase' with value 'Total'
    total_phase['phase'] = 'Total'

    # Append the new dataframe to the original dataframe
    df = pd.concat([tracks_df, total_phase], ignore_index=True, sort=False)

    return df

def create_database(tracks, regions, analysis_type):
    data_frames = []
    for region in regions:
        for season in ["Total", "DJF", "JJA"]: 
            print(f"Region: {region}, Season: {season}")
            region_str = f'_{region}' if region else ''
            output_directory = os.path.join('..', 'periods_species_statistics', analysis_type, 'duration')
            periods_directory = os.path.join('..', 'periods-energetics', analysis_type + region_str)
            os.makedirs(output_directory, exist_ok=True)
            if season != "Total":
                tracks_season = filter_seasons(tracks, season)
            else:
                tracks_season = tracks.copy()
            tracks_season_distance = compute_distance_parallel(tracks_season)
            tracks_season_distance_periods = process_phase_data_parallel(tracks_season_distance, periods_directory)
            df = process_data(tracks_season_distance_periods)
            df = compute_totals(df, tracks_season_distance_periods)
            df['Region'] = 'Total' if not region else region
            df['Season'] = season
            data_frames.append(df)
            print(f'Processed data for region: {region}, season: {season}')

    merged_data_frames = pd.concat(data_frames)
    return merged_data_frames

def main():
    analysis_type = 'all'
    print(f"Analysis type: {analysis_type}")
    regions = [False, "ARG", "LA-PLATA", "SE-BR", "SE-SAO", "AT-PEN", "WEDDELL", "SA-NAM"]
    tracks = get_tracks()
    tracks['date'] = pd.to_datetime(tracks['date']) 
    tracks['year'] = tracks['date'].dt.year
    tracks = tracks[tracks['year'] != 2021]
    unique_years = tracks['year'].unique()
    for year in unique_years:
        print(f"Processing year: {year}")
        duration_database = os.path.join(
            "..",
            "periods_species_statistics",
            analysis_type,
            "periods_database",
            f"periods_database_{year}.csv"
            )
        tracks_year = tracks[tracks['year'] == year]
        try:
            merged_data_frames = pd.read_csv(duration_database)
        except FileNotFoundError:
            print(f"{duration_database} not found, creating it...")
            merged_data_frames = create_database(tracks_year, regions, analysis_type)
        print(f"Periods and tracks have been obtained for year: {year}.")
        merged_data_frames.to_csv(duration_database, index=False)

if __name__ == "__main__":
    main()
