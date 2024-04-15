# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    map_peak_intensity_density.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/04/12 by Danilo                #+#    #+#              #
#    Updated: 2024/04/12 by Danilo               ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Script to plot cyclone density for peak intensity and mature phases
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
    gl.xlocator = mpl.ticker.FixedLocator(range(-90, 181, 20))
    gl.ylocator = mpl.ticker.FixedLocator(range(-90, 91, 10))
    gl.right_labels = False
    gl.top_labels = False
    gl.xlabel_style = {'size': 12, 'color': '#383838'}
    gl.ylabel_style = {'size': 12, 'color': '#383838', 'rotation': 45}

def plot_density(ax, fig, density, phase, season, norm, levels, cmap):
    datacrs = ccrs.PlateCarree()
    ax.set_extent([-90, 180, -15, -90], crs=datacrs)
    lon, lat = density.lon, density.lat

    cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm)
    plt.contour(lon, lat, density, levels=levels, norm=norm, colors='#383838', linewidths=0.35, linestyles='dashed')

    ax.coastlines(zorder=1)
    ax.add_feature(cfeature.LAND, color='#595959', alpha=0.1)
    gridlines(ax)
    props = dict(boxstyle='round', facecolor='white')
    ax.text(160, -40, f'{phase} - {season}', ha='right', va='bottom', fontsize=14, fontweight='bold', bbox=props, zorder=101)

    return cf

def generate_simplified_density_map(infile_directory, output_directory, seasons, phases):
    fig = plt.figure(figsize=(12, 6))
    datacrs = ccrs.PlateCarree()

    level_maps = {
        'peak_intensity': np.linspace(0,4,20),
        'mature': [0.1, 1, 2, 5, 8, 10, 12, 14, 16, 18, 20, 25, 30, 35, 45]
    }

    for j, phase in enumerate(phases):
        colors_linear = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6',
                    '#FEEC9F', '#FDB567', '#F06744', '#C1274A']
        cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)
        levels = level_maps[phase]
        norm = mpl.colors.BoundaryNorm(levels, cmap.N)

        for i, season in enumerate(seasons):
            ax = fig.add_subplot(2, 2, j*2 + i + 1, projection=datacrs)
            infile = f'{infile_directory}/SAt_track_density_{season}.nc'
            ds = xr.open_dataset(infile)
            density = ds[phase]
            cf = plot_density(ax, fig, density, phase, season, norm, levels, cmap)

        cbar_ax = fig.add_axes([0.15, 0.5 if j == 0 else 0.08, 0.7, 0.03])  # Adjust the vertical position based on the row
        plt.colorbar(cf, cax=cbar_ax, orientation='horizontal')

    fname = os.path.join(output_directory, 'peak_intensity_density_map.png')
    plt.savefig(fname, bbox_inches='tight')
    print(f'Density map saved in {fname}')


if __name__ == "__main__":
    analysis_type = "70W-no-continental"
    phases = ['mature', 'peak_intensity']
    seasons = ['DJF', 'JJA']

    output_directory = f'../figures/periods_statistics/{analysis_type}/peak_intensity/'
    infile_directory = f'../periods_species_statistics/{analysis_type}/track_density/'

    os.makedirs(output_directory, exist_ok=True)
    generate_simplified_density_map(infile_directory, output_directory, seasons, phases)
