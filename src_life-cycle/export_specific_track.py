import pandas as pd
from glob import glob

input = "RG1-19810690"
desired_id = 19810690
year = str(desired_id)[:4]

results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
                '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
                '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/']

for results_dir in results_directories:
    for track_file in sorted(glob(f'{results_dir}/*')):

        if year not in track_file:
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
        
        tracks.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']
        
        if 'RG1' in track_file:
            RG = 'RG1'
        elif 'RG2' in track_file:
            RG = 'RG2'
        elif 'RG3' in track_file:
            RG = 'RG3'

        desired_cyclone = tracks[tracks['track_id'] == desired_id]

        if not desired_cyclone.empty:
            track = desired_cyclone[desired_cyclone['track_id']==desired_id][['date','vor42']]
            track = track.rename(columns={"date":"time"})
            track['vor42'] = - track['vor42'] * 1e-5
            track = track.rename(columns={'vor42':'min_zeta_850'})
            tmp_file = (f"tmp_{RG}-{desired_id}.csv")
            track.to_csv(tmp_file, index=False, sep=';')
