#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:31:00 2023

@author: daniloceano
"""
import pandas as pd
import glob
from determine_periods import check_create_folder


intensities = ['10MostIntense', 'moda']

for intensity in intensities:

    results = glob.glob(f'../LEC_results-{intensity}/*ERA5*')
    outdir = f'../periods-energetics/{intensity}/'
    check_create_folder(outdir) 

    for result in results:

        id = result.split('/')[-1].split('_')[0]

        periods_file = result+'/periods.csv'
        results_file = glob.glob(result+'/*_ERA5_track-15x15.csv')[0]
        
        df = pd.read_csv(results_file, index_col=[0])
        df['Datetime'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')
        
        periods = pd.read_csv(periods_file, index_col=[0])
        periods = periods.dropna()
        for i in range(len(periods)):
            start,end = periods.iloc[i]['start'],periods.iloc[i]['end']
            selected_dates = df[(df['Datetime'] >= start) & (df['Datetime'] <= end)]
            if i == 0:
                period = selected_dates.drop(['Datetime','Date','Hour'],axis=1).mean()
                period = period.to_frame(name=periods.iloc[i].name).transpose()
            else:
                tmp = selected_dates.drop(['Datetime','Date','Hour'],axis=1).mean()
                tmp = tmp.to_frame(name=periods.iloc[i].name).transpose()
                period = pd.concat([period,tmp]) 
            
        fname = f'{outdir}/{id}_ERA5.csv'
        period.to_csv(fname)
        print(f'{fname} created')