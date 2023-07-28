# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    automate_GetERA-LEC_RG.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/06/21 17:59:14 by Danilo            #+#    #+#              #
#    Updated: 2023/07/28 10:19:00 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from functools import partial
import argparse
import multiprocessing
import subprocess
import fileinput
import shutil
import pandas as pd
import os 
import logging
import time
import datetime

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
    
    # Print dots while waiting for the process to finish
    while process.poll() is None:
        print('.', end='', flush=True)
        time.sleep(3)  # Adjust the interval between dots if needed
    
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        print()  # Print a new line after the download is complete
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

def download_ERA5(args, testing):
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

def find_track_file(file_id, main_directory, infile_name):

    quantile = infile_name.split('-')[1]

    # Directory containing the tracks_LEC-format
    tracks_lec_dir = os.path.join(main_directory, f"tracks_LEC-format/BY_RG/{quantile}")

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

    print(f'Initializing LEC for: {infile}')

    lorenz_dir = os.path.join("../../", "lorenz-cycle")
    lorenz_src_dir = os.path.join(lorenz_dir, "src")
    lorenz_input_track = os.path.join(lorenz_dir, "inputs", "track")

    # Extract the ID from the infile name
    print('oi')
    file_id =  infile.split("-")[2].split("_")[0]

    track_file = find_track_file(file_id, main_directory, infile)
    print(f'track file: {track_file}')

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

    ERA5_file = download_ERA5(args, testing)
    print(f'ERA5 file to be processed: {ERA5_file}')

    # Check if ERA5_file was downloaded successfully
    if ERA5_file is not None:
        try:
            if os.path.isfile(ERA5_file):
                print(f'ERA5 file exists: {ERA5_file}')
                logging.info(f'ERA5 file found: {ERA5_file}')
                run_LEC(ERA5_file, main_directory, src_directory)
                logging.info('LEC run complete')
                
        except FileNotFoundError:
            logging.error(f'ERA5 file not found: {ERA5_file}')

        except Exception as e:
            logging.error(f'Error running LEC: {e}')

        finally:
            os.remove(ERA5_file)
            logging.info(f'{ERA5_file} deleted')
            
    else:
        logging.error('ERA5 file was not downloaded successfully')

    logging.info(f'finished processing: {ERA5_file}')

    
def main(infile, num_cores, testing):
    if infile is None:
        # Provide a default input file when not provided
        infile = '../dates_limits/RG2-0.999'
    
    filename = os.path.basename(infile)

    # Create the prefix with the file name
    prefix = f"test_{filename}" if testing else filename
            
    # Set up the log file name with the infile name and current date and time
    if testing:
        log_file_name = "logfile-automate_test"
    else:
        log_file_name = f"logfile-automate-{os.path.basename(infile)}-{datetime.datetime.now().strftime('%Y%m%d')}.txt"
    
    # Configure logging with the updated log file name
    logging.basicConfig(filename=log_file_name, level=logging.INFO,
                         format='%(asctime)s - %(levelname)s - %(message)s',
                         filemode='w')
    
    # Save the current directory (src directory)
    src_directory = os.getcwd()

    # Get the parent directory of the main directory
    main_directory = os.path.abspath(os.path.join(src_directory, os.pardir))

    logging.info(f"src_directory: {src_directory}")
    logging.info(f"main_directory: {main_directory}")

    # Path to the scripts directory
    scripts_dir = os.path.join(main_directory, "met_data/scripts/")

    print(f"processing infile: {infile}")
    logging.info(f"Processing {infile}...")
    
    try:
        logging.info("Starting the script")

        # Create a pool of worker processes
        with multiprocessing.Pool(processes=num_cores) as pool:

            with open(infile, 'r') as f:
                next(f)  # Skip the first line
                
                lines = list(f)[1:3] if testing else list(f)

                print("Systems which will be analyzed:")
                for line in lines:
                    print(line[:7])
                
                # Process each line in parallel
                line_args = [(line, prefix, scripts_dir, src_directory, main_directory) for line in lines]
                
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

if __name__ == '__main__':
    # Create an argument parser
    parser = argparse.ArgumentParser(description='Process ERA5 data for a specific infile.')

    # Add arguments for the infile, number of cores, and testing flag
    parser.add_argument('--infile', type=str, default=None,
                    help='Path to the input file to be processed.')
    parser.add_argument('--num_cores', type=int, default=1,
                         help='Number of CPU cores to use for parallel processing.')

    # Parse the arguments
    args = parser.parse_args()

    # Set the testing flag based on the argument
    testing = False

    # Call the main function with the infile, num_cores, and testing flag
    main(args.infile, args.num_cores, testing)