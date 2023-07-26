# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_GetERA-LEC_RG.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/06/21 17:59:14 by Danilo            #+#    #+#              #
#    Updated: 2023/07/26 08:56:44 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from functools import partial

import multiprocessing
import subprocess
import fileinput
import shutil
import pandas as pd
import os 
import logging
import time

testing = False
num_cores = 40

def copy_script_file(file_id):
    script_file = f'GetERA5-pl_{file_id}.py'
    shutil.copy('GetERA5-pl.py', script_file)
    return script_file

def replace_script_variables(script_file, day_start_fmt, day_end_fmt, north, west, south, east, outfile):
    for line in fileinput.input(script_file, inplace=True):
        if "'date':" in line:
            print(f"        'date': '{day_start_fmt}/{day_end_fmt}',")
        elif "'area':" in line:
            print(f"        'area': '{north}/{west}/{south}/{east}',")
        elif "'ID_ERA5.nc'" in line:
            print(f"    '{outfile}')")
        else:
            print(line, end='')

def download_ERA5_file(script_file, file_id, outfile, src_directory):
    cmd = ['python', script_file, file_id]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        logging.info(f'{outfile} download complete')
    else:
        logging.error(f'Error occurred during ERA5 file download: {stderr.decode()}')

def move_script_file(script_file, scripts_dir):
    script_dest = os.path.join(scripts_dir, script_file)
    if os.path.exists(script_dest):
        os.remove(script_dest)
    if os.path.exists(script_file):
        shutil.move(script_file, scripts_dir)
        logging.info(f"Moved {script_file} to {scripts_dir}")

def download_ERA5(args):
    line, prefix, scripts_dir, src_directory, main_directory = args
    file_id, date_start, date_end, south, north, west, east = line.strip().split(',')

    script_file = copy_script_file(file_id)

    # Increase area size a little bit for avoiding data too close to boundaries
    north = round(float(north) + 8)
    south = round(float(south) - 8)
    east = round(float(east) + 8)
    west = round(float(west) - 8)

    day_start_fmt = pd.to_datetime(date_start.split()[0]).strftime('%Y-%m-%d')
    if testing: 
        day_end = pd.to_datetime(date_start) + pd.Timedelta(days=1)
    else:
        day_end = date_end.split()[0]
    day_end_fmt = pd.to_datetime(day_end).strftime('%Y-%m-%d')

    outfile = f'{prefix}-{file_id}_ERA5.nc'

    print(f'Downloading ERA5 file: {outfile}')

    replace_script_variables(script_file, day_start_fmt, day_end_fmt, north, west, south, east, outfile)

    outfile_path = os.path.join(src_directory, outfile)
    if os.path.exists(outfile_path):
        logging.info(f"Destination path '{outfile_path}' already exists. Skipping download.")
    else:
        logging.info(f"Downloading ERA5 data for file ID: {file_id}")
        download_ERA5_file(script_file, file_id, outfile, src_directory)

    move_script_file(script_file, scripts_dir)

    return outfile

def find_track_file(file_id, main_directory, dir_prefix):
    # Directory containing the tracks_LEC-format
    tracks_lec_dir = os.path.join(main_directory, f"tracks_LEC-format/BY_RG/{dir_prefix}")

    # Define the file name pattern to search for
    file_name_pattern = f"track_{file_id}"

    try:
        # Search for the file within the directory structure
        matching_file = None

        for root, dirs, files in os.walk(tracks_lec_dir):
            for file in files:
                if file.startswith(file_name_pattern):
                    matching_file = os.path.join(root, file)
                    break
            if matching_file:
                logging.info(f'{matching_file} found')
                break

        if matching_file is None:
            logging.warning(f'No matching file found for ID: {file_id}')

        return matching_file

    except Exception as e:
        logging.error(f'Error occurred during file search: {e}')
        return None



def run_LEC(infile, main_directory, src_directory):

    lorenz_dir = os.path.join("../../", "lorenz-cycle")
    lorenz_src_dir = os.path.join(lorenz_dir, "src")
    lorenz_input_track = os.path.join(lorenz_dir, "inputs", "track")

    # Extract the ID from the infile name
    infile_name = os.path.basename(infile)
    file_id =  infile_name.split("q")[1].split("-")[1].split("_")[0]
    dir_prefix = infile.split("q")[1].split("-")[0]

    track_file = find_track_file(file_id, main_directory, dir_prefix)

    # Copy track_file to lorenz-cycle/inputs/track
    shutil.copy(track_file, lorenz_input_track)
    logging.info(f'{track_file} copied to {lorenz_input_track}')

    # Move to the desired directory
    os.chdir(lorenz_src_dir)

    infile_path = os.path.join(src_directory, infile)

    # Run program
    logging.info(f"Running LEC for ERA5 file: {infile}")
    cmd = ["python", "lorenz-cycle.py", infile_path, "-t", "-r", "-g"]
    subprocess.call(cmd)

    # Move back to the original directory
    os.chdir(src_directory)

def process_line(args):
    line, prefix, scripts_dir, src_directory, main_directory = args

    # Get the process number from the enumerate function
    process_number, line = line

    # Log the process number
    logging.info(f"Process {process_number}: Started processing line - {line}")

    ERA5_file = download_ERA5(args)

    # Check if ERA5_file was downloaded successfully
    if ERA5_file is not None:
        try:
            while not os.path.exists(ERA5_file):
                time.sleep(1)  # Wait for the file to be downloaded

            if os.path.isfile(ERA5_file):
                print(f'ERA5 file exists: {ERA5_file}')
                run_LEC(ERA5_file, main_directory, src_directory)
                logging.info('LEC run complete')
            else:
                logging.error(f'ERA5 file does not exist: {ERA5_file}')
                
        except FileNotFoundError:
            logging.error(f'ERA5 file not found: {ERA5_file}')
        except Exception as e:
            logging.error(f'Error running LEC: {e}')
        finally:
            os.remove(ERA5_file)
            logging.info(f'{ERA5_file} deleted')
    else:
        logging.error('ERA5 file was not downloaded successfully')
    
if __name__ == '__main__':

    # Configure logging
    logging.basicConfig(filename='logfile-automate.txt', level=logging.INFO,
                         format='%(asctime)s - %(levelname)s - %(message)s',
                         filemode='w')
    
    quantile = 0.99

    try:
        logging.info("Starting the script")

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_cores) as pool:

            # Save the current directory (src directory)
            src_directory = os.getcwd()

            # Get the parent directory of the main directory
            main_directory = os.path.abspath(os.path.join(src_directory, os.pardir))

            logging.info(f"src_directory: {src_directory}")
            logging.info(f"main_directory: {main_directory}")

            # Directory containing the input files
            infiles_dir = os.path.join(main_directory, "dates_limits")

            # Path to the scripts directory
            scripts_dir = os.path.join(main_directory, "met_data/scripts/")

            # Get a list of all input files in the directory
            infiles = [os.path.join(infiles_dir, f) for f in os.listdir(infiles_dir) if f.startswith("RG") and f"-{quantile}" in f]
            print(f"infiles: {infiles}")
            logging.info(f"infile to be processed: {infiles}.")

            # Iterate over each input file
            for infile in infiles:
                print(f"processing infile: {infile}")
                logging.info(f"Processing {infile}...")

                with open(infile, 'r') as f:
                    next(f)  # Skip the first line

                    lines = list(f)[1:3] if testing else list(f)

                    print("Systems which will be analyzed:")
                    for line in lines:
                        print(line)
                    
                    # Extract the pattern from the infile name
                    RG = infile.split("RG")[1].split(f"-{quantile}")[0]

                    # Create the prefix with the pattern
                    prefix = f"test-RG{RG}-q{quantile}" if testing else f"RG{RG}-q{quantile}"
                    
                    process_line_partial = partial(process_line, prefix=prefix)
                    # Process each line in parallel
                    line_args = [(line, prefix, scripts_dir, src_directory, main_directory, process_number)
                                 for process_number, line in enumerate(lines)]

                    logging.info("Starting parallel processing for input file...")
                    pool.map(process_line, line_args)
                    logging.info("Parallel processing completed for input file.")

        logging.info("Script execution completed")

    except Exception as e:
        logging.exception(f"An error occurred in the main block: {e}")
    
    finally:
        pool.close()
        pool.join()
        logging.info("Pool of worker processes closed.")
