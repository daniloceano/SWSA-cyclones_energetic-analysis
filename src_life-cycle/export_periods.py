# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_periods.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 16:45:03 by Danilo            #+#    #+#              #
#    Updated: 2023/09/06 18:13:09 by Danilo           ###   ########.fr        #
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
    tracks.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']
    track = tracks[tracks['track_id']==id_cyclone][['date','vor42']]
    track = track.rename(columns={"date":"time"})
    track['vor42'] = - track['vor42'] * 1e-5
    tmp_file = (f"tmp_{RG}-{id_cyclone}.csv")
    track.to_csv(tmp_file, index=False, sep=';')

    # Check if periods_outfile already exists
    periods_outfile_exists = os.path.exists(periods_outfile)

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

if __name__ == '__main__':


    testing = True

    if testing == True:
        output_directory = './'
        periods_outfile_path = output_directory + './'    
        periods_didatic_outfile_path = output_directory + './'
        periods_csv_outfile_path = './'
        results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/']

    else:
        output_directory = '../figures/'
        periods_outfile_path = output_directory + 'periods/BY_RG-all/'    
        periods_didatic_outfile_path = output_directory + 'periods_didactic/BY_RG-all/'
        periods_csv_outfile_path = '../periods-energetics/BY_RG-all/'

        results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
                        '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
                        '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/']

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
            
            tracks.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']
            
            if 'RG1' in track_file:
                RG = 'RG1'
            elif 'RG2' in track_file:
                RG = 'RG2'
            elif 'RG3' in track_file:
                RG = 'RG3'

            id_cyclones = tracks['track_id'].unique()

            # Create a list of arguments for the process_cyclone function
            arguments_list = [(id_cyclone, track_file, periods_outfile_path, periods_didatic_outfile_path, periods_csv_outfile_path, RG) for id_cyclone in id_cyclones]

            # Use multiprocessing Pool to execute the function in parallel
            with multiprocessing.Pool() as pool:
                pool.map(process_cyclone, arguments_list)
