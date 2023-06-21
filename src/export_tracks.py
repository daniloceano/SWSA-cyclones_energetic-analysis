#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:42:05 2022

@author: daniloceano
"""

import pandas as pd
import os

for quantile in [0.9, 0.95, 0.99, 0.999]:

    for RG in range(1,4):

        csv_name = f'../stats_tracks/BY_RG/tracks-RG{RG}_q{quantile}.csv'
        output_dir = f'../tracks_LEC-format/BY_RG/{quantile}/RG{RG}/'

        if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        df =  pd.read_csv(csv_name, names = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42',
                                                'lon mslp', 'lat mslp', 'mslp', 'lon 10spd',  'lat 10spd', '10spd'],
                                                header=0)

        ids = df.groupby('track_id').mean().index.values

        print(csv_name,ids)

        for idi in ids:

            track = df[df['track_id']==idi][['date','lat vor', 'lon vor']]
            track['date'] = pd.to_datetime(track.date).dt.strftime('%Y-%m-%d-%H%M')
            track.columns = ['time','Lat','Lon']
            
            track.to_csv(f'{output_dir}/track_{idi}',sep=';',index=False)
