# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    plot_LPS-RG.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/06/26 13:20:05 by Danilo            #+#    #+#              #
#    Updated: 2023/06/26 13:40:56 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from LPS import LorenzPhaseSpace

def check_create_folder(DirName, verbosity=False):
    if not os.path.exists(DirName):
                os.makedirs(DirName)
                print(DirName+' created')
    else:
        if verbosity:
            print(DirName+' directory exists')

def get_ids(prefix, RG):
        list = glob.glob(f'../stats_tracks/BY_RG/tracks-RG{RG}_{prefix}.csv')[0]
        df =  pd.read_csv(list, names = ['track_id', 'dt',
                                             'date', 'lon vor',
                                             'lat vor', 'vor42',
                                             'lon mslp', 'lat mslp',
                                             'mslp', 'lon 10spd',
                                             'lat 10spd', '10spd'],
                                             header=0) 
        ids = pd.unique(df['track_id'])
        return ids

def get_id_data(id, prefix):
        outfile = glob.glob(f'../LEC_results-{prefix}/*{prefix}-{id}*ERA*/*{prefix}-{id}*ERA5*.csv')[0]
        df = pd.read_csv(outfile, index_col=[0])
        df['Datetime'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')
        return df

def smooth_data(df, period):
        smoothed = df.groupby(pd.Grouper(key="Datetime", freq=period)).mean(numeric_only=True)
        # Set datetime to the date range
        starts = pd.Series(smoothed.index).dt.strftime('%Y-%m-%d %H:%M')
        ends = pd.Series(pd.DatetimeIndex(starts) + \
                        pd.Timedelta(hours=12)).dt.strftime('%Y-%m-%d %H:%M')
        smoothed['Datetime'] = pd.DataFrame(starts.astype(str)+' - '+\
                                        ends.astype(str)).values
        return smoothed

def period_data(id, prefix, first=False):
        df = get_id_data(id, prefix)
        periods_file = glob.glob(f'../LEC_results-{prefix}/*{prefix}-{id}*ERA*/periods.csv')[0]
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
        # Set datetime to the period date range
        period['Datetime'] = (periods['start'].astype(str)+' - '+\
                                                periods['end'].astype(str)).values
        if first == True:
                period = period.loc[['intensification', 'mature', 'decay']]
        return period 

def create_LPS_plots(fig_title, figsdir, zoom=False, **kwargs):
        plt.close('all')
        plt.figure(figsize=(10,10))
        ax = plt.gca()
        LorenzPhaseSpace(ax, zoom=zoom, **kwargs)
        zoom_suffix = "_zoom" if zoom else ""
        fname = f"{figsdir}/{fig_title}{zoom_suffix}.png"
        with plt.rc_context({'savefig.dpi': 500}):
                plt.savefig(fname)
        print(f"{fname} created!")

if __name__ == "__main__":
    
        datasource = 'ERA5'
        prefix = 'q0.999'

        for RG in range(1,4):

            ids = get_ids(prefix, RG)

            figsdir = f'../figures/LPS/{prefix}/RG{RG}'
            check_create_folder(figsdir)

            for period in ['1H', '6H', '12H', '24H', '48H']:
                    kwargs = {'terms':[], 'title':f'RG{RG} {prefix}+ (+period+ means)','datasource': datasource,
                            'start':1979, 'end': '2020'}

                    plt.close('all')
                    plt.figure(figsize=(10,10))
                    ax = plt.gca()
                    
                    for id in ids:
                            df = get_id_data(id, prefix)
                            smoothed = smooth_data(df, period)
                            terms = {'Ca': smoothed['Ca'], 'Ck': smoothed['Ck'],
                                    'Ge': smoothed['Ge'], 'Ke': smoothed['Ke']}
                            kwargs['terms'].append(terms)

                    create_LPS_plots(f"RG{RG}-{prefix}_{period}", figsdir, zoom=False, **kwargs)
                    create_LPS_plots(f"RG{RG}-{prefix}_{period}", figsdir, zoom=True, **kwargs)
            
            # Plot all periods
            kwargs = {'terms':[], 'title':f"RG{RG} {prefix} (periods mean)",
                    'datasource': datasource, 'start':1979, 'end': '2020'}
            for first in [False, True]:
                    first_suffix = "_first" if first else ""
                    for id in ids:
                            period = period_data(id, prefix, first=first)
                            terms = {'Ca': period['Ca'], 'Ck': period['Ck'],
                                    'Ge': period['Ge'], 'Ke': period['Ke']}
                            kwargs['terms'].append(terms)

            
                    create_LPS_plots(f"RG{RG}-{prefix}_periods{first_suffix}", figsdir, zoom=False, **kwargs)
                    create_LPS_plots(f"RG{RG}-{prefix}_periods{first_suffix}", figsdir, zoom=True, **kwargs)