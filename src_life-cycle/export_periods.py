# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_periods.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 16:45:03 by Danilo            #+#    #+#              #
#    Updated: 2023/08/15 23:57:11 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
This script exports the cyclone periods to a csv file in '../periods-energetics/BY_RG-all/'
for each cyclone

"""

import glob
import pandas as pd
import determine_periods as det
import matplotlib.pyplot as plt
import multiprocessing

def process_cyclone(args):
    id_cyclone, track_file, periods_outfile_path, periods_didatic_outfile_path, periods_csv_outfile_path, RG = args
    plt.close('all') # save memory

    # Set the output file names
    periods_csv_outfile = f"{periods_csv_outfile_path}{RG}_{id_cyclone}.csv"
    periods_outfile = f"{periods_outfile_path}{RG}_{id_cyclone}"
    periods_didatic_outfile = f"{periods_didatic_outfile_path}{RG}_{id_cyclone}"

    try:
        # Read the track file and extract the vorticity data
        tracks = pd.read_csv(track_file)
        tracks.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']
            
        if 'RG1' in track_file:
            RG = 'RG1'
        elif 'RG2' in track_file:
            RG = 'RG2'
        elif 'RG3' in track_file:
            RG = 'RG3'

        track = tracks[tracks['track_id'] == id_cyclone]
        zeta_df = -pd.DataFrame(track['vor42'].rename('zeta'))/1e5  
        zeta_df.index = pd.to_datetime(track['date'].rename('time'))
        vorticity = det.array_vorticity(zeta_df)

        # Determine the periods
        periods_dict, df = det.get_periods(vorticity)

        pd.DataFrame(periods_dict).to_csv(periods_csv_outfile, index=False)
        print(f'{periods_csv_outfile} written.')

        # Create plots
        det.plot_all_periods(periods_dict, df, ax=None, vorticity=vorticity.zeta, periods_outfile_path=periods_outfile)
        det.plot_didactic(df, vorticity, periods_didatic_outfile)

    except Exception as e:
        error_msg = str(e)
        with open("error_log.txt", "a") as file:
            file.write(f"Error processing cyclone: {id_cyclone} - RG: {RG} in track_file: {track_file}\n")
            file.write(f"Error message: {error_msg}\n\n")

testing = False

if testing == True:
    output_directory = './'
    periods_outfile_path = output_directory + './'    
    periods_didatic_outfile_path = output_directory + './'
    periods_csv_outfile_path = './'

else:
    output_directory = '../figures/'
    periods_outfile_path = output_directory + 'periods/BY_RG-all/'    
    periods_didatic_outfile_path = output_directory + 'periods_didactic/BY_RG-all/'
    periods_csv_outfile_path = '../periods-energetics/BY_RG-all/'

results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
                       '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
                       '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/']

# For testing 
# results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/']
####

det.check_create_folder(periods_outfile_path)
det.check_create_folder(periods_didatic_outfile_path)
det.check_create_folder(periods_csv_outfile_path)

if __name__ == '__main__':
    for results_dir in results_directories:
        for track_file in sorted(glob.glob(f'{results_dir}/*')):

            # For testing
            # if '1980' not in track_file:
            #    continue

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
