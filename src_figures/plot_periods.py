#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:17:29 2023

@author: daniloceano
"""

import glob
import pandas as pd
from cyclophaser import determine_periods
import matplotlib.pyplot as plt
import os

intensities = ['0.99']
output_directory = '../figures/'
periods_outfile_path = output_directory + 'periods/BY_RG/'    
periods_didatic_outfile_path = output_directory + 'periods_didactic/BY_RG/'

# intensities = ['10MostIntense', 'moda']
# output_directory = '../figures/'
# periods_outfile_path = output_directory + 'periods/intensity/'    
# periods_didatic_outfile_path = output_directory + 'periods_didactic/intensity/'

for intensity in intensities:

    print('Processing intensity %s' % intensity)

    results_dir = f'../LEC_results-{intensity}/*ERA5*'

    for result in glob.glob(results_dir):  
        
        plt.close('all') # save memory
        fname = result.split('/')[-1].split('.nc')[0] 
        id_cyclone = fname.split('_')[0]
        track_file = glob.glob(f"{results_dir}/{fname}_track")[0]
        print('Cyclone ID:',id_cyclone)
        print('Track file:',track_file) 

        os.makedirs(periods_outfile_path, exist_ok=True)
        os.makedirs(periods_didatic_outfile_path, exist_ok=True)

        # Set the output file names
        periods_outfile = f"{periods_outfile_path}{id_cyclone}"
        periods_didatic_outfile = f"{periods_didatic_outfile_path}{id_cyclone}"

        options = {
        "vorticity_column": 'min_zeta_850',
        "plot": periods_outfile,
        "plot_steps": periods_didatic_outfile,
        "export_dict": False,
        "process_vorticity_args": {
            "use_filter": False,
            "use_smoothing_twice": "auto"}
        }

        # Determine the periods
        df = determine_periods(track_file, **options)
