#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 19:40:27 2023

@author: danilocoutodsouza
"""

import subprocess
import fileinput
import pandas as pd


data_dir = '../met_data/ERA5/DATA/'
scripts_dir = '../met_data/ERA5/scripts/APIs/'

with open('../dates_limits/intense', 'r') as f:
    for line in f:
        
        id, date_start, date_end, south, north, west, east = line.strip().split(',')
        
        if id == '19920334':
        
            subprocess.call(['cp', 'GetERA5-pl.py', f'GetERA5-pl_{id}.py'])
            
            # Increase area size a little bit for avoinding data too close to boundaries
            north = round(float(north)+10)
            south = round(float(north)-10)
            east = round(float(east)+10)
            west = round(float(west)-10)
            
            # Get dates and hours as distinct variables
            day_start, hour_start = date_start.split()[0], date_start.split()[1][:2]
            day_start_fmt = pd.to_datetime(day_start).strftime('%Y%m%d')
            day_end, hour_end = date_end.split()[0], date_end.split()[1][:2]
            day_end_fmt = pd.to_datetime(day_end).strftime('%Y%m%d')
            
            # Replace the 'date' field in the file with the correct date range
            date_range = f"        'date': '{day_start_fmt}/{day_end_fmt}',"
            for line in fileinput.input(f'GetERA5-pl_{id}.py', inplace=True):
                if "'date':" in line:
                    print(date_range)
                else:
                    print(line, end='')
             
            # Replace the 'area' field in the file with the actual area
            area = f"        'area': '{north}/{west}/{south}/{east}',"
            for line in fileinput.input(f'GetERA5-pl_{id}.py', inplace=True):
                if "'area':" in line:
                    print(area)
                else:
                    print(line, end='')
                    
            # Replace the 'ID' field in the file with the ID
            output_file = f"{data_dir}{id}_ERA5.nc"
            for line in fileinput.input(f'GetERA5-pl_{id}.py', inplace=True):
                if "'ID_ERA5.nc'" in line:
                    print(f"     '{output_file}')")
                else:
                    print(line, end='')
                    
            cmd = [
                'python', f'GetERA5-pl_{id}.py',
                id, area
            ]
            subprocess.call(cmd)
            
            subprocess.call(['mv', f'GetERA5-pl_{id}.py', scripts_dir])
