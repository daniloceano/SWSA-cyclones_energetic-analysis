from glob import glob
from geopy.distance import great_circle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

REGIONS = { # lon_min, lat_min, lon_max, lat_max
        "SE-BR": [(-52, -38, -37, -23)],
        "LA-PLATA": [(-69, -38, -52, -23)],
        "ARG": [(-70, -55, -50, -39)],
        "SE-SAO": [(-15, -55, 30, -37)],
        "SA-NAM": [(8, -33, 20, -21)],
        "AT-PEN": [(-65, -69, -44, -58)],
        "WEDDELL": [(-65, -85, -10, -72)]
    }

PATH_TO_RAW_DATA = '../raw_data/SAt'

def get_tracks(year: int, month: int):
    data_folder = os.path.abspath(PATH_TO_RAW_DATA)  # Absolute path
    month_str = f"{month:02d}"  # Format month as two digits
    fname = f"ff_cyc_SAt_era5_{year}{month_str}.csv"
    file_path = os.path.join(data_folder, fname)

    try:
        tracks = pd.read_csv(file_path, header=None)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except pd.errors.ParserError:
        print(f"Error parsing file: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
    track_columns = ['track_id', 'date', 'lon vor', 'lat vor', 'vor42']
    tracks.columns = track_columns
    tracks['lon vor'] = np.where(tracks['lon vor'] > 180, tracks['lon vor'] - 360, tracks['lon vor'])
    tracks.sort_values(by=['track_id', 'date'], inplace=True, kind='mergesort')
    tracks['date'] = pd.to_datetime(tracks['date'])
    return tracks

def get_season(month):
    if month in [12, 1, 2]:
        return 'DJF'  # December, January, February - Winter
    elif month in [3, 4, 5]:
        return 'MAM'  # March, April, May - Spring
    elif month in [6, 7, 8]:
        return 'JJA'  # June, July, August - Summer
    else:
        return 'SON'  # September, October, November - Autumn
    
def get_region(lon, lat):
    for region_name, bounds in REGIONS.items():
        for bound in bounds:
            lon_min, lat_min, lon_max, lat_max = bound
            if lon_min <= lon <= lon_max and lat_min <= lat <= lat_max:
                return region_name
    return "SAt"  # Return "SAt" if no region matches

def check_first_position_inside_area(cyclone_id, tracks, area_bounds):
    cyclone_track = tracks[tracks['track_id'] == cyclone_id]
    first_position = cyclone_track.head(1)  # Get the first row
    first_lat = first_position['lat vor'].values[0]
    first_lon = first_position['lon vor'].values[0]

    min_lon, min_lat, max_lon, max_lat = area_bounds

    # Check if the first position is inside the specified area
    is_inside_area = (min_lat <= first_lat <= max_lat) and (min_lon <= first_lon <= max_lon)

    return cyclone_id, is_inside_area

def compute_distance(lon1, lat1, lon2, lat2):
    try:
        return great_circle((lat1, lon1), (lat2, lon2)).kilometers
    except ValueError:
        # Return NaN if there is a problem with the coordinates
        return np.nan
    
def compute_distances(tracks):
    tracks.sort_values(by=['track_id', 'date'], inplace=True)

    # Shift latitude and longitude
    tracks[['prev_lon vor', 'prev_lat vor']] = tracks.groupby('track_id')[['lon vor', 'lat vor']].shift(1)

    # Vectorized distance calculation
    valid_rows = tracks['prev_lon vor'].notnull()
    coords1 = tracks.loc[valid_rows, ['prev_lat vor', 'prev_lon vor']]
    coords2 = tracks.loc[valid_rows, ['lat vor', 'lon vor']]
    tracks['Distance (km)'] = tracks.apply(
    lambda row: compute_distance(row['prev_lon vor'], row['prev_lat vor'], row['lon vor'], row['lat vor'])
    if pd.notnull(row['prev_lon vor']) else np.nan,
    axis=1
)
    tracks.drop(['prev_lon vor', 'prev_lat vor'], axis=1, inplace=True)

    return tracks

def compute_speeds(tracks):
    # Calculate the time difference in hours between the current and previous positions
    tracks['time_diff'] = tracks.groupby('track_id')['date'].diff().dt.total_seconds() / 3600  # convert seconds to hours

    # Calculate speed by dividing the distance by the time difference and convert from km/h to m/s
    tracks['Speed (m/s)'] = ((tracks['Distance (km)'] * 1000) / (tracks['time_diff'] * 3600)).replace([np.inf, -np.inf], np.nan)

    # Drop the temporary column used for calculation
    tracks.drop('time_diff', axis=1, inplace=True)

    return tracks

def compute_growth_rate(tracks):
    """
    Computes the rate of change in vorticity ('vor42') between each time step for each track_id.
    For the first occurrence of each track_id, the growth rate is set to NaN.
    Assumes vor42 units are 10^−5 s^−1 and converts growth rate to 10^−5 s^−1 day^-1.
    """
    # Ensure the DataFrame is sorted by track_id and date for correct calculation
    tracks.sort_values(by=['track_id', 'date'], inplace=True)

    # Shift the 'vor42' column and 'date' column to align each row with its predecessor
    tracks['prev_vor42'] = tracks.groupby('track_id')['vor42'].shift(1)
    tracks['prev_date'] = tracks.groupby('track_id')['date'].shift(1)

    # Calculate the time difference in days
    tracks['time_diff_days'] = (tracks['date'] - tracks['prev_date']).dt.total_seconds() / (24 * 3600)

    # Calculate the vorticity growth rate in 10^−5 s^−1 day^-1
    tracks['Growth Rate (10^−5 s^−1 day^-1)'] = (tracks['vor42'] - tracks['prev_vor42']) / tracks['time_diff_days']

    # Drop the temporary columns used for calculation
    tracks.drop(['prev_vor42', 'prev_date', 'time_diff_days'], axis=1, inplace=True)

    return tracks

def process_csv_file(csv_file, tracks):
    try:
        df_phases = pd.read_csv(csv_file, parse_dates=['start', 'end'], index_col=0)
        updates = {}
        for phase, row in df_phases.iterrows():
            mask = (tracks['date'] >= row['start']) & (tracks['date'] <= row['end'])
            updates[phase] = mask
        return updates
    except Exception as e:
        print(f"Error processing file {csv_file}: {e}")
        return {}

def process_phase_data_parallel(tracks):
    print("Reading periods...")
    data_path = "../periods-energetics/all"
    csv_files = glob(os.path.join(data_path, '*.csv'))

    if not csv_files:
        print("No CSV files found in the directory.")
        return tracks
    
    # Convert track IDs to strings for matching
    track_ids = tracks['track_id'].unique()
    track_id_strs = [str(id) for id in track_ids]

    # Filter CSV files based on whether they contain any of the track IDs
    csv_files_filtered = [f for f in csv_files if any(track_id in os.path.basename(f) for track_id in track_id_strs)]

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        future_to_file = {executor.submit(process_csv_file, csv_file, tracks): csv_file for csv_file in csv_files_filtered}
        for future in tqdm(as_completed(future_to_file), total=len(csv_files_filtered), desc='Processing Files'):
            result = future.result()
            if result:
                for phase, mask in result.items():
                    tracks.loc[mask, 'phase'] = phase

    print("Done.")
    return tracks

# Adjust the create_database function to reflect these changes
def create_database(tracks_year):
    # Convert the 'date' column to datetime
    tracks_year['date'] = pd.to_datetime(tracks_year['date'])

    # Get the first appearance of each track_id
    first_appearance = tracks_year.groupby('track_id')['date'].min()

    # Map each track_id to its Genesis Season
    tracks_year['Genesis Season'] = tracks_year['track_id'].map(first_appearance.apply(lambda x: get_season(x.month)))

    # Get the Genesis Region for each track_id based on the first appearance's coordinates
    genesis_regions = tracks_year.groupby('track_id').apply(lambda x: get_region(x.iloc[0]['lon vor'], x.iloc[0]['lat vor']))

    # Map each track_id to its Genesis Region
    tracks_year['Genesis Region'] = tracks_year['track_id'].map(genesis_regions)

    # Apply the function to compute distances
    tracks_year = compute_distances(tracks_year)

    # Now we will calculate the speeds.
    tracks_year = compute_speeds(tracks_year)

    # Apply the function to compute growth rates
    tracks_year = compute_growth_rate(tracks_year)

    # Sanity checks
    tracks_distance_sum = tracks_year.groupby('track_id')['Distance (km)'].sum().reset_index(name='Total Distance (km)')
    global_mean_distance = tracks_distance_sum['Total Distance (km)'].mean()
    tracks_speed_mean = tracks_year.groupby('track_id')['Speed (m/s)'].mean().reset_index(name='Speed (m/s)')
    global_mean_speed = tracks_speed_mean['Speed (m/s)'].mean()
    print(f"Global Mean Speed: {global_mean_speed:.2f} m/s, Global Mean Distance: {global_mean_distance:.2f} km")

    tracks_year = process_phase_data_parallel(tracks_year)

    return tracks_year

def main():
    tracks_pattern = "ff_cyc_SAt_era5_"
    raw_tracks = sorted(glob(os.path.join(PATH_TO_RAW_DATA, f"{tracks_pattern}*.csv")))
    years = np.unique([int(os.path.basename(raw_track).split(tracks_pattern)[1].split(".")[0][:4]) for raw_track in raw_tracks])

    for year in years:
        for month in range(1, 13):
            month_str = f"{month:02d}"
            print(f"Processing year: {year}, month: {month_str}")
            tracks_year = get_tracks(year, month)
            if tracks_year is None:
                print(f"No data available for year {year}, month {month}. Skipping...")
                continue

            # Create database if it doesn't exist
            databse_path = f"../processed_tracks_with_periods/"
            os.makedirs(databse_path, exist_ok=True)
            duration_database = os.path.join(databse_path, f"ff_cyc_SAt_era5_{year}{month_str}.csv")
            try:
                pd.read_csv(duration_database)
                print(f"{duration_database} already exists.")
            except FileNotFoundError:
                print(f"{duration_database} not found, creating it...")
                tracks = create_database(tracks_year)
                tracks.to_csv(duration_database)
                print(f"{duration_database} created.")

if __name__ == "__main__":
    main()