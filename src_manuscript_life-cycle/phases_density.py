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

    levels = np.linspace(0, round(float(np.amax(density))), 21)

    cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels, extend='max')

    cax = fig.add_axes([ax.get_position().x1 + 0.02,
                        ax.get_position().y0, 0.02,
                        ax.get_position().height]) 
    # Define the ticks for the colorbar with every 2nd value
    ticks = np.round(levels[::3], decimals=2)
    plt.colorbar(cf, cax=cax, ticks=ticks)

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

plt.subplots_adjust(wspace=0.35)

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
