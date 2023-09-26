# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    periods_energetics-RG.py                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/06/26 15:02:10 by Danilo            #+#    #+#              #
#    Updated: 2023/09/26 19:34:55 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import glob
import os

from cyclophaser import determine_periods

qauntile = 0.99

figures_outdir = f'../figures/periods/quantile/{qauntile}'
didatic_outdir = f'../figures/periods_didactic/quantile/{qauntile}'
csv_outdir = f'../periods-energetics/{qauntile}/'
os.makedirs(figures_outdir, exist_ok=True)
os.makedirs(didatic_outdir, exist_ok=True)
os.makedirs(csv_outdir, exist_ok=True)

for RG in range(1,4):

    results = glob.glob(f'../LEC_results-{qauntile}/RG{RG}-{qauntile}*ERA5*')

    for result in results:

        cyclone_id = result.split('/')[-1].split('_')[0].split('-')[-1]

        track_file = glob.glob(result+'/*track')[0]

        #pd.read_csv(track_file, sep=';')

        periods_fig_outfile = os.path.join(figures_outdir, f'RG{RG}_{cyclone_id}.png')
        didatic_fig_outfile = os.path.join(didatic_outdir, f'RG{RG}_{cyclone_id}.png')
        periods_csv_outfile = os.path.join(csv_outdir, f'RG{RG}_{cyclone_id}.csv')

        options = {
        "vorticity_column": 'min_zeta_850',
        "plot": periods_fig_outfile,
        "plot_steps": didatic_fig_outfile,
        "export_dict": periods_csv_outfile,
        "process_vorticity_args": {
            "use_filter": "auto",
            "use_smoothing_twice": "auto"}
        }

        df = determine_periods(track_file, **options)

        # periods_file = result+'/periods.csv'
        # try:
        #     results_file = glob.glob(result+'/*_ERA5_track-15x15.csv')[0]
        
        #     df = pd.read_csv(results_file, index_col=[0])
        #     df['Datetime'] = pd.to_datetime(df.Date) + pd.to_timedelta(df.Hour, unit='h')
            
        #     periods = pd.read_csv(periods_file, index_col=[0])
        #     periods = periods.dropna()
        #     for i in range(len(periods)):
        #         start,end = periods.iloc[i]['start'],periods.iloc[i]['end']
        #         selected_dates = df[(df['Datetime'] >= start) & (df['Datetime'] <= end)]
        #         if i == 0:
        #             period = selected_dates.drop(['Datetime','Date','Hour'],axis=1).mean()
        #             period = period.to_frame(name=periods.iloc[i].name).transpose()
        #         else:
        #             tmp = selected_dates.drop(['Datetime','Date','Hour'],axis=1).mean()
        #             tmp = tmp.to_frame(name=periods.iloc[i].name).transpose()
        #             period = pd.concat([period,tmp]) 

        # except:
        #     print(result)
        #     pass
            
        # fname = f'{outdir}/{id}_ERA5.csv'
        # period.to_csv(fname)
        # print(f'{fname} created')