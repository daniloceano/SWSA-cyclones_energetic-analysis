# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_density.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/08 20:33:08 by Danilo            #+#    #+#              #
#    Updated: 2023/08/09 16:00:21 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# Adapted from Carolina B. Gramcianinov (cbgramcianinov@gmail.com)

# Carolina B. Gramcianinov: cbgramcianinov@gmail.com
# Feb/2021
#
# script to plot cyclone genesis density with KDE method
# using sklearn package
# - options:
# * to save the density in a netcdf file set savenc key to 'y'
# * to plot (with cartopy + matplotlib) set plot ket to 'y'
# --------------


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
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

def plot_density(density, fname):

    lon, lat = density.lon, density.lat

    levels = np.linspace(0, round(float(density.max()),2), 21)

    cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels)

    # Create a separate axis for the colorbar
    cax = fig.add_axes([0.92, 0.078, 0.02, 0.45]) 
    cbar = plt.colorbar(cf, cax=cax, ticks=levels) 

    ax.text(-90, -91, phase, ha='left', va='top', fontsize=20, fontweight='bold')

    ax.coastlines(zorder=1)
    gridlines(ax)
    plt.tight_layout()

    plt.savefig(os.path.join(output_directory, fname), bbox_inches='tight')
    print(f'Density map saved in {os.path.join(output_directory, fname)}')

output_directory = '../figures/periods_statistics/density_maps/'

#####################################

colors = ['white', '#F1F5F9', '#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', 
            '#FEEC9F', '#FDB567', '#F06744',  '#C1274A']
cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

for RG in ['1', '2', '3', 'all']:

    for phase in ['incipient', 'intensification', 'mature', 'decay', 'residual',
                  'intensification 2', 'mature 2', 'decay 2']:
        
        RG_str = f'RG{RG}' if RG != 'all' else 'all'

        fig = plt.figure(figsize=(15, 10))
        datacrs = ccrs.PlateCarree()
        ax = fig.add_axes([-0.01, -0.05, 0.9, 0.7], projection=datacrs, frameon=True)
        ax.set_extent([-90, 180, 0, -90], crs=datacrs)

        ds = xr.open_dataset(f'./track_density_RG1.nc')

        density = ds[phase]
        fname = f'density_map_{RG_str}_{phase}.png'

        plot_density(density, fname)

