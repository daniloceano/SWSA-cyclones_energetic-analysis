#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 15:10:42 2023

@author: daniloceano
"""

import pandas as pd
import glob
import os

prefix = '../raw_data/ff_*'
files =  glob.glob(prefix)

for csv_name in files:
    
    intensity = csv_name.split('.')[-2].split('_')[-1]

    df =  pd.read_csv(csv_name, names = ['track_id', 'dt',
                                         'date', 'lon vor',
                                         'lat vor', 'vor42',
                                         'lon mslp', 'lat mslp',
                                         'mslp', 'lon 10spd',
                                         'lat 10spd', '10spd'])
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
    df_se.to_csv('../dates_limits/'+intensity,
                 sep=',',index=False, header=False)