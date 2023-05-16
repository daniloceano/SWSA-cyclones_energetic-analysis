#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 19:40:27 2023

@author: danilocoutodsouza
"""

import subprocess
import fileinput
import shutil
import pandas as pd
import os 

data_dir = '../met_data/ERA5/DATA/'
scripts_dir = '../met_data/ERA5/scripts/APIs/'

with open('../dates_limits/intense', 'r') as f:
    next(f)  # Skip the first line
    for line in f:
        
        file_id, date_start, date_end, south, north, west, east = line.strip().split(',')
        
        shutil.copy('GetERA5-pl.py', f'GetERA5-pl_{file_id}.py')
        
        # Increase area size a little bit for avoinding data too close to boundaries
        north = round(float(north)+20)
        south = round(float(south)-20)
        east = round(float(east)+20)
        west = round(float(west)-20)
        
        # Get dates and hours as distinct variables
        day_start, hour_start = date_start.split()[0], date_start.split()[1][:2]
        day_start_fmt = pd.to_datetime(day_start).strftime('%Y-%m-%d')
        day_end, hour_end = date_end.split()[0], date_end.split()[1][:2]
        day_end_fmt = pd.to_datetime(day_end).strftime('%Y-%m-%d')
        
        # Perform replacements in the script file
        script_file = f'GetERA5-pl_{file_id}.py'
        outfile = f'{file_id}_ERA5.nc'
        for line in fileinput.input(script_file, inplace=True):
            if "'date':" in line:
                print(f"        'date': '{day_start_fmt}/{day_end_fmt}',")
            elif "'area':" in line:
                print(f"        'area': '{north}/{west}/{south}/{east}',")
            elif "'ID_ERA5.nc'" in line:
                print(f"    '{outfile}')")
            else:
                print(line, end='')

        # Check if outfile already exists
        if not os.path.exists(os.path.join(data_dir, outfile)):
            cmd = ['python', script_file, file_id]
            subprocess.call(cmd)

        # Remove existing script file if it exists in the destination directory
        script_dest = os.path.join(scripts_dir, script_file)
        if os.path.exists(script_dest):
            os.remove(script_dest)

        # Move the new script file to the destination directory
        shutil.move(script_file, scripts_dir)

        # Check if outfile already exists
        outfile_path = os.path.join(data_dir, outfile)
        if os.path.exists(outfile_path):
            print(f"Destination path '{outfile_path}' already exists. Overwriting the file.")
            # Overwrite the outfile
            shutil.move(outfile, outfile_path, copy_function=shutil.copy2)
        else:
            # Move the outfile to the data directory
            shutil.move(outfile, outfile_path)
