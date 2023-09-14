# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    phases_density.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/08 20:33:08 by Danilo            #+#    #+#              #
#    Updated: 2023/09/14 20:23:03 by Danilo           ###   ########.fr        #
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

def plot_density(ax, i, density, label):

    ax.set_extent([-90, 180, 0, -90], crs=datacrs)

    lon, lat = density.lon, density.lat

    if i < 4:
        levels = np.linspace(0, 50, 21)
    else:
        levels = np.linspace(0, 3, 21)

    cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels, extend='max')

    if i == 0:
        # Create a separate axis for the colorbar
        cax = fig.add_axes([0.92, 0.52, 0.02, 0.35]) 
        plt.colorbar(cf, cax=cax, ticks=levels)
    if i == 4:
        cax = fig.add_axes([0.92, 0.12, 0.02, 0.35]) 
        plt.colorbar(cf, cax=cax, ticks=levels)

    props = dict(boxstyle='round', facecolor='white')
    ax.text(160, -18, label, ha='left', va='bottom', fontsize=16, fontweight='bold', bbox=props)

    ax.coastlines(zorder=1)
    gridlines(ax)

#####################################

colors = ['white', '#F1F5F9', '#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', 
            '#FEEC9F', '#FDB567', '#F06744',  '#C1274A']
cmap = mcolors.LinearSegmentedColormap.from_list("", colors)

fig = plt.figure(figsize=(15, 10))
datacrs = ccrs.PlateCarree()

ds = xr.open_dataset('../src_life-cycle/track_density_RGall.nc')

phases = ['incipient', 'intensification', 'mature', 'decay',
            'intensification 2', 'mature 2', 'decay 2', 'residual']

letters = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)']

for i, phase in enumerate(phases):
    
    print(f'Plotting {phase}')
    
    # Add a subplot
    ax = fig.add_subplot(4, 2, i+1, projection=datacrs)
    
    density = ds[phase]
    label = f'{letters[i]}'
    plot_density(ax, i, density, label)

fname = '../figures/manuscript_life-cycle/density_map.png'
plt.savefig(fname, bbox_inches='tight')
print(f'Density map saved in {fname}')
