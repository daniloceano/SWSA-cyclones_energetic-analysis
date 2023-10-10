# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_density.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/08 20:33:08 by Danilo            #+#    #+#              #
#    Updated: 2023/10/10 00:43:22 by Danilo           ###   ########.fr        #
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

def plot_density(ax, density, phase, i):

    ax.set_extent([-90, 180, 0, -90], crs=datacrs)

    lon, lat = density.lon, density.lat

    # levels = np.linspace(0, round(float(density.max()),2), 21)
    levels = [0, 0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 130]

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
    ax.text(160, -18, phase, ha='right', va='bottom', fontsize=14, fontweight='bold', bbox=props)

    ax.coastlines(zorder=1)
    gridlines(ax)

    return cf

#####################################    

# analysis_type = 'BY_RG-all'
# analysis_type = 'all'
# analysis_type = '70W'
# analysis_type = '48h'
# analysis_type = '70W-48h'
# analysis_type = "70W-1000km"
# analysis_type = "70W-1500km"
# analysis_type = "70W-decayC"
analysis_type = "70W-no-continental"

output_directory = f'../figures/periods_statistics/{analysis_type}/density_maps/'
infile_directory = f'../periods_species_statistics/{analysis_type}/track_density/'

phases = ['incipient', 'intensification', 'mature', 'decay', 'residual',
                    'intensification 2', 'mature 2', 'decay 2']

# List of season names
seasons = ['JJA', 'MAM', 'SON', 'DJF', False]

# Colormap for plotting
colors_linear = ['white', '#F1F5F9', '#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', 
            '#FEEC9F', '#FDB567', '#F06744',  '#C1274A']
colors_linear = ['white', '#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', 
            '#FEEC9F', '#FDB567', '#F06744',  '#C1274A']
cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)

# colors_listed = ['white', '#4471b2', '#008bc1', '#00a3c2', '#1fb8ba', '#66cbae', '#80cc99',
#           '#9ecc85', '#bdc975', '#cab658', '#d99f43', '#e6853c', '#ef6d42', '#E1471F']
# cmap = mcolors.ListedColormap(colors_listed)

os.makedirs(output_directory, exist_ok=True)

for season in seasons:
    season_str = f'_{season}' if season else ''
    print(f'season: {season}') if season else print('all seasons') 

    infile = f'{infile_directory}/track_density{season_str}.nc'
    ds = xr.open_dataset(infile)

    fname = os.path.join(output_directory, f"density_{analysis_type}{season_str}")

    fig = plt.figure(figsize=(12, 10))
    datacrs = ccrs.PlateCarree()

    for i, phase in enumerate(phases):
        ax = fig.add_subplot(4, 2, i+1, projection=datacrs, )

        density = ds[phase]
        plot_density(ax, density, phase, i)

    plt.savefig(fname, bbox_inches='tight')
    print(f'Density map saved in {fname}')
