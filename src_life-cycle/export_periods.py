# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_periods.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 16:45:03 by Danilo            #+#    #+#              #
#    Updated: 2023/09/29 23:34:12 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script exports the cyclone periods to a csv file in '../periods-energetics/BY_RG-all/'
for each cyclone

"""

import glob
import pandas as pd
from cyclophaser import determine_periods
import matplotlib.pyplot as plt
import multiprocessing
import os

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
        "plot": periods_outfile,
        "plot_steps": periods_didatic_outfile,
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
    # Filter the 'tracks' DataFrame to keep only the systems that are west of 70W
    if '70W' in analysis_type:
        RG = 'SAt-70W'
        ids_west_70 = tracks[tracks['lon vor'] < -70]['track_id'].unique()
        tracks = tracks[~tracks['track_id'].isin(ids_west_70)]

    # Filter the 'tracks' DataFrame to keep only the systems that have a duration of at least 48 hours
    if '48h' in analysis_type:
        RG = 'SAt-70W'
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
    
    return tracks, RG

testing = False
# analysis_type = 'BY_RG-all'
# analysis_type = 'all'
# analysis_type = '70W' 
analysis_type = '48h'
# analysis_type = '70W-48h'

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
