# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_density.py                                     :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/08 20:33:08 by Danilo            #+#    #+#              #
#    Updated: 2023/10/11 17:32:28 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Adapted from Carolina B. Gramcianinov (cbgramcianinov@gmail.com) - Feb/2021

Script to plot cyclone density with KDE method from sklearn package
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import matplotlib.colors as mcolors
import numpy as np
import os

def gridlines(ax):
    gl = ax.gridlines(draw_labels=True, zorder=100, linestyle='dashed', alpha=0.5,
                     color='#383838', lw=0.25)
    gl.xlocator = mpl.ticker.FixedLocator(range(-90, 181, 20))  # Add latitude grid lines every 30 degrees
    gl.ylocator = mpl.ticker.FixedLocator(range(-90, 91, 10))  # Add longitude grid lines every 20 degrees
    gl.right_labels = False  # Display longitude labels on the left side
    gl.top_labels = False  # Display latitude labels at the bottom
    gl.xlabel_style = {'size': 12, 'color': '#383838'}
    gl.ylabel_style = {'size': 12, 'color': '#383838', 'rotation': 45}

def plot_density(ax, fig, density, phase, i, region=False):
    datacrs = ccrs.PlateCarree()
    
    ax.set_extent([-90, 180, -15, -90], crs=datacrs)
    lon, lat = density.lon, density.lat

    levels = [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50, 70, 100, 130]
    if region in ['SE-BR', 'AT-PEN', 'SA-NAM']:
        levels = [0.1, 1, 2, 5, 8, 10, 13, 15, 18, 20, 25, 30, 35, 40, 45, 50]
    elif region in ['LA-PLATA', 'SA-NAM']:
        levels = [0.1, 1, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60]
    elif region == 'WEDDELL':
        levels = [0.1, 1, 2, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70]

    colors_linear = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6',
                    '#FEEC9F', '#FDB567', '#F06744', '#C1274A']
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)

    cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm)
    plt.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838',
                 linewidths=0.35, linestyles='dashed')

    if i == 7:
        cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        ticks = np.round(levels, decimals=2)
        colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')
        colorbar.ax.tick_params(labelsize=12)

    props = dict(boxstyle='round', facecolor='white')
    ax.text(160, -40, phase, ha='right', va='bottom', fontsize=14, fontweight='bold',
             bbox=props, zorder=101)

    ax.coastlines(zorder=1)
    ax.add_feature(cfeature.LAND, color='#595959', alpha=0.1)  # Add grey shading to continents
    gridlines(ax)


def generate_density_map(analysis_type, region, season):
    phases = ['incipient', 'intensification', 'mature', 'decay', 'residual',
              'intensification 2', 'mature 2', 'decay 2']

    output_directory = f'../figures/periods_statistics/{analysis_type}/density_maps/'
    infile_directory = f'../periods_species_statistics/{analysis_type}/track_density/'

    os.makedirs(output_directory, exist_ok=True)

    region_str = f"{region}_" if region else "SAt_"
    season_str = f"_{season}" if season else ""

    infile = f'{infile_directory}/{region_str}track_density{season_str}.nc'

    ds = xr.open_dataset(infile)

    fig = plt.figure(figsize=(14.5, 10))
    datacrs = ccrs.PlateCarree()

    for i, phase in enumerate(phases):
        ax = fig.add_subplot(4, 2, i + 1, projection=datacrs)
        density = ds[phase]
        plot_density(ax, fig, density, phase, i, region)

    fname = os.path.join(output_directory, f"{region_str}density_{analysis_type}{season_str}.png")
    plt.savefig(fname, bbox_inches='tight')
    print(f'Density map saved in {fname}')


if __name__ == "__main__":
    """
    analysis_types = ['all', '70W',  '48h', '70W-48h', '70W-1000km',
                        '70W-1500km', '70W-decayC', '70W-no-continental']

    regions = ["SE-BR", "LA-PLATA","ARG", "SE-SAO", "SA-NAM",
                "AT-PEN", "WEDDELL", False]
    """
    analysis_type = "70W-no-continental"
    regions = ["SE-BR", "LA-PLATA", "ARG", "SE-SAO", "SA-NAM", "AT-PEN", "WEDDELL", False]
    seasons = ['JJA', 'MAM', 'SON', 'DJF', False]

    for region in regions:
        for season in seasons:
            generate_density_map(analysis_type, region, season)