# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    periods_energetics-RG.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/06/26 15:02:10 by Danilo            #+#    #+#              #
#    Updated: 2023/07/31 17:13:51 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import glob
from determine_periods import check_create_folder


qauntile = 0.99

for RG in range(1,4):

    results = glob.glob(f'../LEC_results-q{qauntile}/RG{RG}-{qauntile}*ERA5*')
    outdir = f'../periods-energetics/{qauntile}/RG{RG}'
    check_create_folder(outdir) 

    for result in results:

        id = result.split('/')[-1].split('_')[0]

        periods_file = result+'/periods.csv'
        try:
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

        except:
            print(result)
            pass
            
        fname = f'{outdir}/{id}_ERA5.csv'
        period.to_csv(fname)
        print(f'{fname} created')