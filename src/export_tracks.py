#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 15:42:05 2022

@author: daniloceano
"""

import pandas as pd
import glob


prefix = '../raw_data/ff_*'
files =  glob.glob(prefix)

for csv_name in files:

    df =  pd.read_csv(csv_name, names = ['track_id', 'dt',
                                         'date', 'lon vor',
                                         'lat vor', 'vor42',
                                         'lon mslp', 'lat mslp',
                                         'mslp', 'lon 10spd',
                                         'lat 10spd', '10spd'])
    
    ids = df.groupby('track_id').mean().index.values
    
    print(csv_name,ids)
    
    intensity = csv_name.split('.')[-2].split('_')[-1]
    
    for idi in ids:
    
        track = df[df['track_id']==idi][['date','lat vor', 'lon vor']]
        track['date'] = pd.to_datetime(track.date).dt.strftime('%Y-%m-%d-%H%M')
        track.columns = ['time','Lat','Lon']
        
        track.to_csv('../tracks_LEC-format/'+intensity+\
                     '/track_'+str(idi),sep=';',index=False)
