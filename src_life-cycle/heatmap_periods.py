# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    heatmap_periods.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/08 10:54:18 by Danilo            #+#    #+#              #
#    Updated: 2023/08/15 17:18:54 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.colors as mcolors
from dateutil.relativedelta import relativedelta

"""
Reads the periods exported by export_periods.py and produce heatmaps
"""

def gridlines(ax):
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

    periods = pd.read_csv(glob(f'{periods_directory}*{cyclone_id}*')[0])
    
    dt = track['date'].iloc[1] - track['date'].iloc[0]

    # Convert period timestamps to datetime objects
    corresponding_periods = pd.DataFrame([], columns=['period'], index=track['date'])
    for phase in list(periods.columns):
        periods[phase] = pd.to_datetime(periods[phase])
        period_dates = pd.date_range(start=periods[phase][0], end=periods[phase][1], freq=dt)

        corresponding_periods['period'].loc[period_dates] = phase

    # Add a new column 'period' to the track DataFrame
    track['period'] = corresponding_periods['period'].values

    if filter:
        if 'residual' in track['period'].values:
            track = track[track['period'] != 'residual']

    return track

def plot_phase_count(ds, phase, num_months):
    plt.close('all')

    colors = ['white', '#F1F5F9', '#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', 
            '#FEEC9F', '#FDB567', '#F06744',  '#C1274A']
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

    count = ds[phase]/num_months

    levels = np.linspace(0, round(float(count.max()),2), 21)

    fig = plt.figure(figsize=(15, 10))
    datacrs = ccrs.PlateCarree()
    ax = fig.add_axes([-0.01, -0.05, 0.9, 0.7], projection=datacrs, frameon=True)
    ax.set_extent([-90, 180, 0, -90], crs=datacrs)
    
    cf = plt.contourf(ds['longitude'], ds['latitude'], count, cmap=cmap, levels=levels)

    # Create a separate axis for the colorbar
    cax = fig.add_axes([0.92, 0.078, 0.02, 0.45])  # Adjust the position and size as needed
    cbar = plt.colorbar(cf, cax=cax, ticks=levels)  # Set ticks to rounded levels

    ax.text(-80, 10, phase, ha='center', va='center', fontsize=14, fontweight='bold')

    ax.coastlines(zorder=1)
    gridlines(ax)
    plt.tight_layout()

    fname = f'heatmap_{phase}.png'
    plt.savefig(os.path.join(output_directory, fname), bbox_inches='tight')

if __name__ == '__main__':

    periods_directory = '../periods-energetics/BY_RG-all/'
    output_directory = '../figures/periods_statistics/heatmaps/'
    os.makedirs(output_directory, exist_ok=True)

    results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
                        '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
                        '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/']
    
    delta = relativedelta(pd.to_datetime("2019-11-30"), pd.to_datetime("1989-01-01"))
    num_months = delta.years * 12 + delta.months

    # Create latitude and longitude grid
    lats = np.arange(-90, 0, 1)
    lons = np.arange(-180, 180, 1)

    # Create an empty xarray Dataset with latitude and longitude dimensions
    ds = xr.Dataset(coords={"latitude": lats, "longitude": lons})

    # List of phases
    phases = ['incipient', 'intensification', 'mature', 'decay',
              'intensification 2', 'mature 2', 'decay 2', 'residual']
    
    # Iterate through phases and count occurrences for each grid point
    for phase in phases:
        ds[phase] = xr.DataArray(
            np.zeros((len(lats), len(lons)), dtype=int), dims=("latitude", "longitude"))
        
        for results_directory in results_directories:  
            files = glob(f'{results_directory}*')
            for file in files:  
                try:
                    tracks = pd.read_csv(file)
                except:
                    continue
                tracks.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor',
                                'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd',
                                'lat 10spd', '10spd']

                cyclone_ids = tracks['track_id'].unique()

                for cyclone_id in cyclone_ids:
                    track = get_track(cyclone_id, tracks, filter=False)
                    track['lon vor'] = (track['lon vor'] + 180) % 360 - 180
                    
                    track_phase = track[track['period'] == phase]
                    
                    for lat_bin, lon_bin in zip(track_phase['lat vor'], track_phase['lon vor']):
                        ds[phase].sel(latitude=lat_bin, longitude=lon_bin, method='nearest').values += 1

        
        plot_phase_count(ds, phase, num_months)
    
    print(ds)
