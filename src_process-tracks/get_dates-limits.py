#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:10:42 2023

@author: daniloceano

Get start and end date of all systems and their corresponding maximum and minimum latitude and longitude.
Store the results as csv files in the directory 'dates_limits'.

"""

import pandas as pd
import glob

output_dir = f'../dates_limits/'

for quantile in [0.9, 0.95, 0.99, 0.999]:

    for RG in range(1,4):

        prefix = f'../stats_tracks/BY_RG/tracks-RG{RG}_q{quantile}.csv'
        files =  glob.glob(prefix)

        for csv_name in files:
            
            df =  pd.read_csv(csv_name, names = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42',
                                                'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 
                                                'lat 10spd', '10spd'], header=0)
            
            ids = df.groupby('track_id').mean().index.values
            print(csv_name,ids)
            
            starts, ends = [], []
            min_lats, max_lats = [], []
            min_lons, max_lons = [], []
            for idx in ids:
                df_id = df[df.track_id==idx]
            
                periods = len(df_id)
                dates = pd.date_range(start=df_id.date.iloc[0],
                                    periods=periods, freq='1H')
                starts.append(dates[0])
                ends.append(dates[-1])
                min_lats.append(df_id['lat vor'].min())
                max_lats.append(df_id['lat vor'].max())
                min_lons.append(df_id['lon vor'].min())
                max_lons.append(df_id['lon vor'].max())

            df_se = pd.DataFrame([ids, starts, ends,
                                min_lats, max_lats,
                                min_lons, max_lons]).transpose()
            
            df_se.columns = ['track_id','start', 'end',
                            'min_lat','max_lat',
                            'min_lon','max_lon']
            
            outfile_name = f'{output_dir}/RG{RG}-{quantile}'

            df_se.to_csv(outfile_name, sep=',',index=False, header=True)

            print(f"{outfile_name} created successfully.")