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
import matplotlib as mpl

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

    if phase == 'incipient':
        levels = [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 28, 30]
    elif phase in ['intensification', 'decay']:
        levels = [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 130]
    elif phase == 'mature':
        levels = [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 25, 30, 35, 40]
    else:
        levels = [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 28, 30]

    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm)
    plt.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838',
                 linewidths=0.35, linestyles='dashed')
    
    ax.coastlines()

    if i == 7:
        cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        ticks = np.round(levels, decimals=2)
        colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')
        colorbar.ax.tick_params(labelsize=12)

    props = dict(boxstyle='round', facecolor='white')
    ax.text(160, -18, label, ha='left', va='bottom', fontsize=16, fontweight='bold', bbox=props)

    gridlines(ax)

#####################################

phases = ['incipient', 'intensification', 'mature', 'decay',
            'intensification 2', 'mature 2', 'decay 2', 'residual']
regions = [False, "SE-BR", "LA-PLATA", "ARG", "SE-SAO", "SA-NAM", "AT-PEN", "WEDDELL"]
seasons = [False, 'JJA', 'DJF']
letters = ['A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)']



colors_linear = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6',
                '#FEEC9F', '#FDB567', '#F06744', '#C1274A']
cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)

results_directory = '../periods_species_statistics/70W-no-continental/track_density'

for phase in phases:
    print(f'Plotting phase: {phase}')

    for season in seasons:
        print(f'Plotting season: {season}')

        fig = plt.figure(figsize=(15, 10))
        datacrs = ccrs.PlateCarree()

        for i, region in enumerate(regions):
            ax = fig.add_subplot(4, 2, i+1, projection=datacrs)

            region_str = region if region else "SAt"
            season_str = f"_{season}" if season else ""

            infile = os.path.join(results_directory, f'{region_str}_track_density{season_str}.nc')
            ds = xr.open_dataset(infile)
            
            density = ds[phase]
            label = f'{letters[i]}'
            plot_density(ax, i, density, label)

        plt.subplots_adjust(wspace=0.35)
        fname = f'../figures/manuscript_life-cycle/density_map_{phase}{season_str}.png'
        plt.savefig(fname, bbox_inches='tight')
        print(f'Density map saved in {fname}')
