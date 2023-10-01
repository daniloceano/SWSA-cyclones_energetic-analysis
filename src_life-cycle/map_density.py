# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_density.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/08 20:33:08 by Danilo            #+#    #+#              #
#    Updated: 2023/10/01 10:49:54 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Adapted from Carolina B. Gramcianinov (cbgramcianinov@gmail.com) - Feb/2021

Script to plot cyclone density with KDE method from sklearn package
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import os

def gridlines(ax):
    gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',alpha=0.8,
                 color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.top_labels = None
    gl.right_labels = None

def plot_density(ax, density, phase):

    ax.set_extent([-90, 180, 0, -90], crs=datacrs)

    lon, lat = density.lon, density.lat

    levels = np.linspace(0, round(float(density.max()),2), 21)

    cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels)

    # Create a separate axis for the colorbar
    cax = fig.add_axes([ax.get_position().x1 + 0.02,
                        ax.get_position().y0, 0.02,
                        ax.get_position().height]) 
    # Define the ticks for the colorbar with every 2nd value
    ticks = np.round(levels[::3], decimals=2)
    plt.colorbar(cf, cax=cax, ticks=ticks)

    props = dict(boxstyle='round', facecolor='white')
    ax.text(160, -18, phase, ha='right', va='bottom', fontsize=12, fontweight='bold', bbox=props)

    ax.coastlines(zorder=1)
    gridlines(ax)

#####################################    

# analysis_type = 'BY_RG-all'
# analysis_type = 'all'
# analysis_type = '70W'
analysis_type = '48h'

output_directory = f'../figures/periods_statistics/{analysis_type}/density_maps/'
infile_directory = f'../periods_species_statistics/{analysis_type}/track_density/'

phases = ['incipient', 'intensification', 'mature', 'decay', 'residual',
                    'intensification 2', 'mature 2', 'decay 2']

# List of season names
seasons = ['JJA', 'MAM', 'SON', 'DJF', False]

# List of RGs
RGs = ['1', '2', '3', 'all'] if analysis_type == 'BY_RG-all' else ['all']

# Colormap for plotting
colors = ['white', '#F1F5F9', '#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', 
            '#FEEC9F', '#FDB567', '#F06744',  '#C1274A']
cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

os.makedirs(output_directory, exist_ok=True)

for RG in RGs:
    if analysis_type == 'BY_RG-all':
        RG_str = f'_RG{RG}' if RG != 'all' else '_all-RG'
    else:
        RG_str = ''
    print(f'RG: {RG}...')

    for season in seasons:
        season_str = f'_{season}' if season else ''
        print(f'season: {season}') if season else print('all seasons') 

        infile = f'{infile_directory}/track_density{RG_str}{season_str}.nc'
        ds = xr.open_dataset(infile)

        if analysis_type == 'BY_RG-all':
            fname = os.path.join(output_directory, RG_str+season_str)
        else:
            fname = os.path.join(output_directory, 'all'+season_str)

        fig = plt.figure(figsize=(15, 10))
        datacrs = ccrs.PlateCarree()
        plt.subplots_adjust(wspace=0.35)

        for i, phase in enumerate(phases):
            ax = fig.add_subplot(4, 2, i+1, projection=datacrs, )

            density = ds[phase]
            plot_density(ax, density, phase)

        plt.savefig(fname, bbox_inches='tight')
        print(f'Density map saved in {fname}')