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

PHASES = ['incipient', 'intensification', 'mature', 'decay',
            'intensification 2', 'mature 2', 'decay 2', 'residual']
REGIONS = [False, "ARG", "LA-PLATA", "SE-BR", "SE-SAO", "AT-PEN", "WEDDELL" , "SA-NAM"]
SEASONS = [False, 'JJA', 'DJF']
LABELS = ['(A)', '(B)', '(C)', '(D)', '(E)', '(F)', '(G)', '(H)']
COLORS = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6',
                '#FEEC9F', '#FDB567', '#F06744', '#C1274A']
INFILES_DIRECTORY = '../periods_species_statistics/70W-no-continental/track_density'
OUTPUT_DIRECTORY = '../figures/manuscript_life-cycle/density_maps/'
datacrs = ccrs.PlateCarree()

def gridlines(ax):
    gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',alpha=0.8,
                 color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.top_labels = None
    gl.right_labels = None

def plot_density(fig, ax, i, phase, density, label):

    ax.set_extent([-90, 180, 0, -90], crs=datacrs)

    cmap = mcolors.LinearSegmentedColormap.from_list("", COLORS)

    if phase in ['intensification', 'decay']:
        levels = [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 130]
    elif phase == 'mature':
        levels = [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 25, 30, 35, 40]
    elif phase in ['intensification 2', 'mature 2', 'decay 2', 'residual']:
        levels = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]
    else:
        levels = [0.1, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20, 22, 25, 28, 30]

    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    lon, lat = density.lon, density.lat
    cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm)
    plt.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838',
                 linewidths=0.35, linestyles='dashed')

    props = dict(boxstyle='round', facecolor='white')
    ax.text(150, -22, label, ha='left', va='bottom', fontsize=16, fontweight='bold', bbox=props)
    ax.coastlines()
    gridlines(ax)
    return levels, cf

def plot_each_phase():
    
    for phase in PHASES:
        print(f'Plotting phase: {phase}')

        for season in SEASONS:
            print(f'Plotting season: {season}')

            fig = plt.figure(figsize=(12, 10))
            
            for i, region in enumerate(REGIONS):
                ax = fig.add_subplot(4, 2, i+1, projection=datacrs)

                region_str = region if region else "SAt"
                season_str = f"_{season}" if season else ""

                infile = os.path.join(INFILES_DIRECTORY, f'{region_str}_track_density{season_str}.nc')
                ds = xr.open_dataset(infile)
                
                density = ds[phase]
                label = f'{LABELS[i]}'
                levels, cf = plot_density(fig, ax, i, phase, density, label)

                if i == 7:
                    cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.04])
                    ticks = np.round(levels, decimals=2)
                    colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')
                    colorbar.ax.tick_params(labelsize=12)

            plt.subplots_adjust(wspace=0.15)
            fname = os.path.join(OUTPUT_DIRECTORY, f'density_map_{phase}{season_str}.png')
            plt.savefig(fname, bbox_inches='tight')
            print(f'Density map saved in {fname}')

def plot_secondary_development():
    # Plot just summer and winter
    for i, season in enumerate(SEASONS):
        print(f'Plotting secondary development for season: {season}')
        fig = plt.figure(figsize=(12, 5))
        datacrs = ccrs.PlateCarree()
        
        for i, phase in enumerate(['intensification 2', 'mature 2', 'decay 2', 'residual']):
            ax = fig.add_subplot(2, 2, i+1, projection=datacrs)

            season_str = f"_{season}" if season else ""

            infile = os.path.join(INFILES_DIRECTORY, f'SAt_track_density{season_str}.nc')
            ds = xr.open_dataset(infile)
            
            density = ds[phase]
            label = f'{LABELS[i]}'
            levels, cf = plot_density(fig, ax, i, phase, density, label)

            if i == 3:
                cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.04])
                ticks = np.round(levels, decimals=2)
                colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')
                colorbar.ax.tick_params(labelsize=12)

        plt.subplots_adjust(wspace=0.15)
        fname = os.path.join(OUTPUT_DIRECTORY, f'density_map_secondary_development{season_str}.png')
        plt.savefig(fname, bbox_inches='tight')
        #plt.show()
        print(f'Density map saved in {fname}')


# plot_each_phase()
plot_secondary_development()
