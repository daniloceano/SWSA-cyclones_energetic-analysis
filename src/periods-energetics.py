#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:31:00 2023

@author: daniloceano
"""
import pandas as pd

id_n = 19830422
id = str(19830422)

testfile = '../LEC_results/'+id+'_ERA5_track-15x15/'+id+'_ERA5_track-15x15.csv'
periods_file = '../LEC_results/'+id+'_ERA5_track-15x15/periods.csv'

df = pd.read_csv(testfile, index_col=[0])
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
    
    