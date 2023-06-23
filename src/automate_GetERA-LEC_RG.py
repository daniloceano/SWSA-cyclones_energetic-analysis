# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_GetERA-LEC_RG.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/06/21 17:59:14 by Danilo            #+#    #+#              #
#    Updated: 2023/06/22 23:47:24 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import multiprocessing
import subprocess
import fileinput
import shutil
import pandas as pd
import os 
import logging

def download_ERA5(line, prefix, scripts_dir):
    """
    Downloads ERA5 data files for a given list of files and date ranges.
    
    infile (srt):
      Path to the input file containing a list of files and date ranges.
    prefix (str):
      Prefix for the output file names.
    scripts_dir (str):
      Path to the directory where the scripts should be saved.
    data_dir (str):
      Path to the directory where the data should be saved.
    """
            
    file_id, date_start, date_end, south, north, west, east = line.strip().split(',')
    
    shutil.copy('GetERA5-pl.py', f'GetERA5-pl_{file_id}.py')
    
    # Increase area size a little bit for avoinding data too close to boundaries
    north = round(float(north)+8)
    south = round(float(south)-8)
    east = round(float(east)+8)
    west = round(float(west)-8)
    
    # Get dates and hours as distinct variables
    day_start = date_start.split()[0]
    day_start_fmt = pd.to_datetime(day_start).strftime('%Y-%m-%d')
    # day_end = pd.to_datetime(day_start) + pd.Timedelta(days=1)
    day_end = date_end.split()[0]
    day_end_fmt = pd.to_datetime(day_end).strftime('%Y-%m-%d')
    
    # Perform replacements in the script file
    script_file = f'GetERA5-pl_{file_id}.py'
    outfile = f'{prefix}-{file_id}_ERA5.nc'
    for line in fileinput.input(script_file, inplace=True):
        if "'date':" in line:
            print(f"        'date': '{day_start_fmt}/{day_end_fmt}',")
        elif "'area':" in line:
            print(f"        'area': '{north}/{west}/{south}/{east}',")
        elif "'ID_ERA5.nc'" in line:
            print(f"    '{outfile}')")
        else:
            print(line, end='')

    # Check if the outfile already exists
    outfile_path = os.path.join(src_directory, outfile)
    if os.path.exists(outfile_path):
        print(f"Destination path '{outfile_path}' already exists. Skipping download.")
    else:
        # Download the file
        cmd = ['python', script_file, file_id]
        subprocess.call(cmd)

    # Remove existing script file if it exists in the destination directory
    script_dest = os.path.join(scripts_dir, script_file)
    if os.path.exists(script_dest):
        print(f"{script_file} already exists in {script_dest}. Removing it.")
        os.remove(script_dest)
        
    # Move the new script file to the destination directory, if it exists
    if os.path.exists(script_file):
        shutil.move(script_file, scripts_dir)
        print(f"moved {script_file} to {scripts_dir}")

    return outfile

def find_track_file(file_id):

    # Directory containing the tracks_LEC-format
    tracks_lec_dir = os.path.join(main_directory, "tracks_LEC-format")

    # Define the file name pattern to search for
    file_name_pattern = f"track_{file_id}"

    # Search for the file within the directory structure
    matching_file = None

    for root, dirs, files in os.walk(tracks_lec_dir):
        for file in files:
            if file.startswith(file_name_pattern):
                matching_file = os.path.join(root, file)
                break
        if matching_file:
            break
    return matching_file


def run_LEC(infile, main_directory, src_directory):

    lorenz_dir = os.path.join("../../", "lorenz-cycle")
    lorenz_src_dir = os.path.join(lorenz_dir, "src")
    lorenz_input_track = os.path.join(lorenz_dir, "inputs", "track")

    # Extract the ID from the infile name
    infile_name = os.path.basename(infile)
    file_id = infile_name.split("-")[1].split("_")[0]

    track_file = find_track_file(file_id)

    # Copy track_file to lorenz-cycle/inputs/track
    shutil.copy(track_file, lorenz_input_track)

    # Move to the desired directory
    os.chdir(lorenz_src_dir)

    infile_path = os.path.join(src_directory, infile)

    # Run program
    cmd = ["python", "lorenz-cycle.py", infile_path, "-t", "-r", "-g", "> output.txt"]
    subprocess.call(cmd)

    # Move back to the original directory
    os.chdir(main_directory)

# Define a worker function to process each line
def process_line(lines):
    ERA5_file = download_ERA5(lines, prefix, scripts_dir)
    logging.info(f'{ERA5_file} download complete')
    run_LEC(ERA5_file, main_directory, src_directory)
    logging.info('LEC run complete')
    os.remove(ERA5_file)
    logging.info(f'{ERA5_file} deleted')


# Save the current directory (src directory)
src_directory = os.getcwd()

# Get the parent directory of the main directory
main_directory = os.path.abspath(os.path.join(src_directory, os.pardir))

print("src_directory:", src_directory)
print("main_directory:", main_directory)

# Directory containing the input files
infiles_dir = os.path.join(main_directory, "dates_limits")

# Path to the scripts directory
scripts_dir = os.path.join(main_directory, "met_data/ERA5/scripts/APIs/")

# Prefix for the output file names
prefix = "q0.999"

# Get a list of all input files in the directory
infiles = [os.path.join(infiles_dir, f) for f in os.listdir(infiles_dir) if f.startswith("RG") and "-0.999" in f]
print(f"infiles: {infiles}")

# Create a pool of worker processes
pool = multiprocessing.Pool(processes=5)

# Configure logging
logging.basicConfig(filename='logfile-automate.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Iterate over each input file
for infile in infiles:

    print(f"Processing {infile}...")

    with open(infile, 'r') as f:
        next(f)  # Skip the first line

        # Create a list of lines in the file
        lines = list(f)

        # Process each line in parallel
        pool.map(process_line, lines)

