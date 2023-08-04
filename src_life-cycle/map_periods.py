# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_periods.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/04 09:20:43 by Danilo            #+#    #+#              #
#    Updated: 2023/08/04 17:39:19 by Danilo           ###   ########.fr        #
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
import os

def gridlines(ax):
    # ax.add_feature(cartopy.feature.LAND)
    # ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
    gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',alpha=0.8,
                 color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.bottom_labels = None
    gl.right_labels = None

def get_track(cyclone_id, tracks, filter=False):

    track = tracks[tracks['track_id'] == cyclone_id].copy()
    track['date'] = pd.to_datetime(track['date'])
    track['lon vor'] = (track['lon vor'] + 180) % 360 - 180

    periods = pd.read_csv(glob.glob(f'{periods_directory}*{cyclone_id}*')[0])
    
    dt = track['date'].iloc[1] - track['date'].iloc[0]

    # Convert period timestamps to datetime objects
    corresponding_periods = pd.DataFrame([], columns=['period', 'color'], index=track['date'])
    for phase in list(periods.columns):
        periods[phase] = pd.to_datetime(periods[phase])
        period_dates = pd.date_range(start=periods[phase][0], end=periods[phase][1], freq=dt)

        corresponding_periods['period'].loc[period_dates] = phase
        corresponding_periods['color'].loc[period_dates] = colors_phases_all[phase]

    # Add a new column 'period' to the track DataFrame
    track['period'] = corresponding_periods['period'].values
    track['color'] = corresponding_periods['color'].values

    if filter:
        if 'residual' in track['period'].values:
            track = track[track['period'] != 'residual']

    return track

def plot_tracks(results_directories, suffix, filter=False):

    if filter:
        suffix += '_filtered'
        ncol=7
        colors_phases = colors_phases_filtered
    else:
        colors_phases = colors_phases_all
        ncol=5

    print(f'Filter: {filter}')

    fig = plt.figure(figsize=(15, 10))
    datacrs = ccrs.PlateCarree()
    ax = fig.add_axes([0.05, 0.05, 0.9, 0.9], projection=datacrs, frameon=True)
    ax.set_extent([-70, 180, 10, -90], crs=datacrs)
    ax.coastlines(zorder = 1)
    gridlines(ax)

    for results_dir in results_directories:
        for track_file in glob.glob(f'{results_dir}/*'):
            try:
                tracks = pd.read_csv(track_file)
            except:
                print('not able to open:', track_file)
                continue
            
            tracks.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor',
                            'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd',
                            'lat 10spd', '10spd']
            
            cyclone_ids = tracks['track_id'].unique()
            
            for cyclone_id in cyclone_ids:
                
                track = get_track(cyclone_id, tracks, filter)

                if filter:
                    track_phases = track['period'].unique()
                    if track_phases not in filtered_periods:
                        continue
                
                for date in track['date'].unique():
                    idata = track[track['date'] == date]
                    if (list(idata['period'])[0]) not in colors_phases:
                        continue
                    plt.plot(idata['lon vor'], idata['lat vor'], marker='o',
                            color=list(idata['color'])[0], alpha=0.5)
                    
    plt.text(-59.5, 0.5, suffix, fontsize=20, fontweight='bold')

    # Create the legend patches and labels
    legend_patches = [plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=color, label=phase)
                    for phase, color in colors_phases.items()]

    # Create a legend for the phases
    legend = plt.legend(handles=legend_patches, loc='upper left',
                        bbox_to_anchor=(-0.03, -0.03),ncol=ncol, fontsize=14)
    plt.gca().add_artist(legend)  # Add legend to the existing axes

    fname = f'{output_directory}/map_periods_{suffix}.png'
    plt.savefig(fname,dpi=500)
    print(f"{fname} written.")

def main():

    for filter in [True, False]:
        if filter == False:
            continue
        results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
                        '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
                        '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/']
        
        for i in range(len(results_directories)):
            plot_tracks([results_directories[i]], f'RG{i+1}', filter)
            print()
        
        plot_tracks(results_directories, 'all', filter)

if __name__ == '__main__':

    periods_directory = '../periods-energetics/BY_RG-all_raw/'
    output_directory = '../figures/periods_statistics/maps/'
    os.makedirs(output_directory, exist_ok=True)

    colors_phases_all = {'incipient': '#65a1e6',
                    'incipient 2': '#206dc5',
                    'intensification': '#f7b538',
                    'intensification 2': '#c48208',
                    'mature': '#d62828',
                    'mature 2': '#971c1c', 
                    'decay': '#9aa981', 
                    'decay 2': '#74b474',
                    'residual': '#b3b3b3',
                    'residual 2': '#666666'} 
    
    colors_phases_filtered = {'incipient': '#65a1e6',
                    'intensification': '#f7b538',
                    'intensification 2': '#c48208',
                    'mature': '#d62828',
                    'mature 2': '#971c1c', 
                    'decay': '#9aa981', 
                    'decay 2': '#74b474'} 
    
    filtered_periods = pd.read_csv('filtered_periods.csv').values

    main()