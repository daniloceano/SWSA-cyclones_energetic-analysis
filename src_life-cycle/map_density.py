# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_density.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/08 20:33:08 by Danilo            #+#    #+#              #
#    Updated: 2023/10/11 13:04:56 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Adapted from Carolina B. Gramcianinov (cbgramcianinov@gmail.com) - Feb/2021

Script to plot cyclone density with KDE method from sklearn package
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import os

def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=100, linestyle='solid', alpha=0.8,
                 color='#383838', lw=0.5)
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.top_labels = None
    gl.right_labels = None

def plot_density(ax, density, phase, i, region=False):

    ax.set_extent([-90, 180, -15, -90], crs=datacrs)

    lon, lat = density.lon, density.lat

    levels = [0, 0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 130]
    if region == 'SE-BR':
        levels = [0, 0.1, 1, 2, 5, 8, 10, 13, 15, 18, 20, 25, 30, 35, 40, 45, 50]

    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm)
    plt.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838',
                 linewidths=0.35, linestyles='dashed')

    if i == 7:
        # Create a grid for colorbars below the subplots
        cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        # Define the ticks for the colorbar with every 2nd value
        ticks = np.round(levels, decimals=2)
        colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g',  orientation='horizontal')
        colorbar.ax.tick_params(labelsize=12)

    props = dict(boxstyle='round', facecolor='white')
    ax.text(160, -40, phase, ha='right', va='bottom', fontsize=14, fontweight='bold', bbox=props)

    ax.coastlines(zorder=1)
    gridlines(ax)

    return cf

#####################################    

# Choose an analysis type
# analysis_type = 'BY_RG-all'
# analysis_type = 'all'
# analysis_type = '70W'
# analysis_type = '48h'
# analysis_type = '70W-48h'
# analysis_type = "70W-1000km"
# analysis_type = "70W-1500km"
# analysis_type = "70W-decayC"
analysis_type = "70W-no-continental"

# Choose a region
region = "SE-BR"
## region = False

output_directory = f'../figures/periods_statistics/{analysis_type}/density_maps/'
infile_directory = f'../periods_species_statistics/{analysis_type}/track_density/'

phases = ['incipient', 'intensification', 'mature', 'decay', 'residual',
                    'intensification 2', 'mature 2', 'decay 2']

# List of season names
seasons = ['JJA', 'MAM', 'SON', 'DJF', False]

# Colormap for plotting
colors_linear = ['white', '#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', 
            '#FEEC9F', '#FDB567', '#F06744',  '#C1274A']
cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)


os.makedirs(output_directory, exist_ok=True)

for season in seasons:
    print(f'Season: {season}') if season else print('Season: all') 
    print(f'Region: {region}') if region else print('Region: SAt') 

    region_str = f"{region}_" if region else "SAt_"
    season_str = f"_{season}" if season else ""
    
    if region:
        infile = f'{infile_directory}/{region_str}track_density{season_str}.nc'
    else:
        infile = f'{infile_directory}/track_density{season_str}.nc'
        
    ds = xr.open_dataset(infile)

    fig = plt.figure(figsize=(14.5, 10))
    datacrs = ccrs.PlateCarree()

    for i, phase in enumerate(phases):
        ax = fig.add_subplot(4, 2, i+1, projection=datacrs, )

        density = ds[phase]
        plot_density(ax, density, phase, i, region)

    fname = os.path.join(output_directory, f"{region_str}density_{analysis_type}{season_str}.png")
    plt.savefig(fname, bbox_inches='tight')
    print(f'Density map saved in {fname}')
