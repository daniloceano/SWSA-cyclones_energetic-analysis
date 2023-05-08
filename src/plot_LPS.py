#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:05:29 2022

@author: danilocoutodsouza
"""


import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cmocean
import glob
import numpy as np

from LPS import LorenzPhaseSpace
             

def get_ids(intensity):
        list = glob.glob('../raw_data/CycloneList_*'+intensity+'*')[0]
        df =  pd.read_csv(list, names = ['track_id', 'dt',
                                             'date', 'lon vor',
                                             'lat vor', 'vor42',
                                             'lon mslp', 'lat mslp',
                                             'mslp', 'lon 10spd',
                                             'lat 10spd', '10spd']) 
        ids = pd.unique(df['track_id'])
        return ids

def get_id_data(id):
        outfile = glob.glob('../LEC_results/'+str(id)+'*ERA*/'+str(id)+'*ERA5*.csv')[0]
        df = pd.read_csv(outfile, index_col=[0])
        df['Datetime'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')
        return df
      

if __name__ == "__main__":
    
        datasource = 'ERA5'
        intensity = '10MostIntense'
        ids = get_ids(intensity)

        for method in ['1H', '6H', '12H', '24H', '48H']:

                kwargs = {'terms':[], 'title':intensity,'datasource': datasource}

                plt.close('all')
                plt.figure(figsize=(10,10))
                ax = plt.gca()
                
                for id in ids:

                        df = get_id_data(id)

                        smoothed = df.groupby(pd.Grouper(key="Datetime", freq=method)).mean(numeric_only=True)
                        # Set datetime to the date range
                        starts = pd.Series(smoothed.index).dt.strftime('%Y-%m-%d %H:%M')
                        ends = pd.Series(pd.DatetimeIndex(starts) + \
                                        pd.Timedelta(hours=12)).dt.strftime('%Y-%m-%d %H:%M')
                        smoothed['Datetime'] = pd.DataFrame(starts.astype(str)+' - '+\
                                                        ends.astype(str)).values
                        
                        terms = {'Ca': smoothed['Ca'], 'Ck': smoothed['Ck'],
                                'Ge': smoothed['Ge'], 'Ke': smoothed['Ke']}

                        kwargs['terms'].append(terms)

                LorenzPhaseSpace(ax, **kwargs)
                fname = '../Figures/LPS/LPS_'+intensity+'_'+method+'.png'
                plt.savefig(fname,dpi=500)
                print(fname+' created!')

                LorenzPhaseSpace(ax, zoom=True, **kwargs)
                fname = '../Figures/LPS/LPS_'+intensity+'_'+method+'_zoom.png'
                plt.savefig(fname,dpi=500)
                print(fname+' created!')
 