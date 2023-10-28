# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_periods_database.py                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/10/27 19:48:00 by Danilo            #+#    #+#              #
#    Updated: 2023/10/27 23:01:18 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #



import os
import glob
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from export_density_all import get_tracks
from concurrent.futures import ProcessPoolExecutor

SECONDS_IN_AN_HOUR = 3600

def optimize_dataframe(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df[numeric_cols] = df[numeric_cols].round(2)
    for col in df.select_dtypes(include=['object']):
        num_unique_values = len(df[col].unique())
        num_total_values = len(df[col])
        if num_unique_values / num_total_values < 0.5:
            df[col] = df[col].astype('category')
    # Convert float64 to float32
    df_float = df.select_dtypes(include=['float64'])
    df[df_float.columns] = df_float.astype(np.float32)
    
    return df

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
    chunk_size = 10000
    chunks = pd.read_csv(csv_file, parse_dates=['start', 'end'], index_col=0, chunksize=chunk_size)
    updates = {}
    for df_phases in chunks:
        for phase, row in df_phases.iterrows():
            mask = (tracks['date'] >= row['start']) & (tracks['date'] <= row['end'])
            updates[phase] = mask
    return updates

def filter_csv_file(f, track_id_set):
    df_sample = pd.read_csv(f, nrows=10)
    return any(track_id in df_sample.values for track_id in track_id_set)

def map_filter_func(args):
    f, track_id_set = args
    return f, filter_csv_file(f, track_id_set)

def get_filtered_csv_files_parallel(csv_files, track_ids):
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
    filtered_csv_files = get_filtered_csv_files_parallel(csv_files, track_ids) 
    tracks['date'] = pd.to_datetime(tracks['date'])    
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(tqdm(executor.map(lambda csv_file: process_csv_file(csv_file, tracks),
                                          filtered_csv_files), total=len(filtered_csv_files),
                                            desc='Processing Files'))
    for result in results:
        for phase, mask in result.items():
            tracks.loc[mask, 'phase'] = phase
    print("Done.")
    return tracks

def compute_distance_chunk(chunk):
    prev_lon = chunk.groupby('track_id')['lon vor'].shift(1)
    prev_lat = chunk.groupby('track_id')['lat vor'].shift(1)
    
    chunk['Distance'] = haversine_distance(
        prev_lat, 
        prev_lon, 
        chunk['lat vor'], 
        chunk['lon vor']
    )
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
    tracks['month'] = pd.to_datetime(tracks['date']).dt.month
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
    tracks_distance_periods['time_diff'] = tracks_distance_periods.groupby('track_id')['date'].diff().dt.total_seconds() / SECONDS_IN_AN_HOUR
    total_distance = tracks_distance_periods.groupby(['track_id', 'phase'])['Distance'].sum().reset_index(name='Total Distance (km)')
    total_time = tracks_distance_periods.groupby(['track_id', 'phase'])['time_diff'].sum().reset_index(name='Total Time (Hours)')
    merged_df = pd.merge(total_distance, total_time, on=['track_id', 'phase'])
    merged_df['Mean Speed (km/h)'] = merged_df['Total Distance (km)'] / merged_df['Total Time (Hours)']
    merged_df['Mean Speed (m/s)'] = merged_df['Mean Speed (km/h)'] * (1000 / 3600)
    merged_df.drop(columns=['Mean Speed (km/h)'], inplace=True)
    print("Done")
    return merged_df

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
            tracks_season_distance = optimize_dataframe(tracks_season_distance)
            tracks_season_distance_periods = process_phase_data_parallel(tracks_season_distance, periods_directory)
            df = process_data(tracks_season_distance_periods)
            df['Region'] = 'Total' if not region else region
            df['Season'] = season
            data_frames.append(df)
            print(f'Processed data for region: {region}, season: {season}')

    merged_data_frames = pd.concat(data_frames)
    return merged_data_frames

def main():
    analysis_type = '70W-no-continental'
    print(f"Analysis type: {analysis_type}")
    regions = [False, "ARG", "LA-PLATA", "SE-BR", "SE-SAO", "AT-PEN", "WEDDELL", "SA-NAM"]
    duration_database = os.path.join("..", "periods_species_statistics", analysis_type, "duration_time", "periods_database_djf_jja.csv")
    tracks = get_tracks()
    try:
        merged_data_frames = pd.read_csv(duration_database)
    except FileNotFoundError:
        print(f"{duration_database} not found, creating it...")
        merged_data_frames = create_database(tracks, regions, analysis_type)
    print(f"Periods and tracks have been obtained.")
    merged_data_frames.to_csv(duration_database, index=False)

if __name__ == "__main__":
    main()
