# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_periods.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/04 09:20:43 by Danilo            #+#    #+#              #
#    Updated: 2023/08/04 11:13:26 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cartopy.crs as ccrs
import cartopy
import matplotlib.pyplot as plt
import pandas as pd
import cmocean
import matplotlib.colors as colors
import numpy as np
import glob

periods_directory = '../periods-energetics/BY_RG-all_raw/'

results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
                       '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
                       '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/']

def gridlines(ax):
    ax.add_feature(cartopy.feature.LAND)
    ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
    gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',alpha=0.8,
                 color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.bottom_labels = None
    gl.right_labels = None


fig = plt.figure(figsize=(12, 9))
datacrs = ccrs.PlateCarree()
ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=datacrs, frameon=True)
ax.set_extent([-70, 20, 10, -70], crs=datacrs)
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
ax.coastlines(zorder = 1)
gridlines(ax)

for results_dir in results_directories:
    for track_file in glob.glob(f'{results_dir}/*')[:1]:
        tracks = pd.read_csv(track_file)
        
        tracks.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor',
                          'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd',
                          'lat 10spd', '10spd']
        
        cyclone_ids = tracks['track_id'].unique()
        
        for cyclone_id in cyclone_ids[:2]:
            track = tracks[tracks['track_id'] == cyclone_id].copy()
            track['date'] = pd.to_datetime(track['date'])
            track['lon vor'] = (track['lon vor'] + 180) % 360 - 180

            periods = pd.read_csv(glob.glob(f'{periods_directory}*{cyclone_id}*')[0])
            
            dt = track['date'].iloc[1] - track['date'].iloc[0]

            # Convert period timestamps to datetime objects
            corresponding_periods = pd.Series(range(len(track)), index=track['date'])
            for phase in list(periods.columns):
                periods[phase] = pd.to_datetime(periods[phase])
                period_dates = pd.date_range(start=periods[phase][0], end=periods[phase][1], freq=dt)

                corresponding_periods.loc[period_dates] = phase

            # Add a new column 'period' to the track DataFrame
            track['period'] = corresponding_periods.values
            
            plt.plot(track['lon vor'], track['lat vor'], marker='o')

plt.show()