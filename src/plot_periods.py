#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:17:29 2023

@author: daniloceano
"""

import glob
import pandas as pd
import determine_periods as det
  

data_dir = '../met_data/ERA5/DATA/'
track_dir = '../tracks_LEC-format/intense/'
output_directory = '../periods-energetics/figures/intense/'
periods_outfile_path = output_directory + 'periods/'    
periods_didatic_outfile_path = output_directory + 'periods_didactic/'

for testfile in glob.glob(data_dir+'/*'):  
    
    fname = testfile.split('/')[-1].split('.nc')[0] 
    id_cyclone = fname.split('_')[0]
    track_file = track_dir+'track_'+id_cyclone
    print('Cyclone ID:',id_cyclone)
    print('Track file:',track_file)  

    # Set the output file names
    periods_outfile = f"{periods_outfile_path}{id_cyclone}"
    periods_didatic_outfile = f"{periods_outfile_path}{id_cyclone}"

    det.check_create_folder(periods_outfile_path)
    det.check_create_folder(periods_didatic_outfile_path)

    # Read the track file and extract the vorticity data
    track = pd.read_csv(track_file, parse_dates=[0], delimiter=';', index_col=[0])
    zeta_df = pd.DataFrame(track['min_zeta_850'].rename('zeta'))        
    vorticity = det.array_vorticity(zeta_df)

    # Determine the periods
    periods = det.get_phases(vorticity, output_directory)

    # Create plots
    det.plot_periods(vorticity, periods, periods_outfile_path)
    det.plot_didactic(vorticity, periods, periods_didatic_outfile_path)
        

