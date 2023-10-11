import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import xarray as xr
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches

def plot_region_box(ax, coords, edgecolor, label=None):
    for coord in coords:
        lon_corners = np.array([coord[0], coord[2], coord[2], coord[0]])
        lat_corners = np.array([coord[1], coord[1], coord[3], coord[3]])

        poly_corners = np.column_stack((lon_corners, lat_corners))
        polygon = mpatches.Polygon(poly_corners, closed=True, ec=edgecolor,
                                    fill=False, lw=2, alpha=0.5,
                                      transform=ccrs.Geodetic())
        ax.add_patch(polygon)

        if label:
            text_lat = coord[3]
            text_lon = coord[2] - 1
            if label == 'ARG':
                text_lat = coord[3] - 5
                text_lon = coord[2] + 10
            elif label == 'AT-PEN':
                text_lon = coord[2] + 15
            elif label == 'SA-NAM':
                text_lat = coord[3] + 5
            elif label == 'SE-SAO':
                text_lon = coord[2] - 27
                text_lat = coord[3] - 5
                
            ax.text(text_lon, text_lat, label, transform=ccrs.Geodetic(),
                     fontsize=16, color=edgecolor, fontweight='bold', ha='right', va='bottom')

def create_map_and_axes():
    fig = plt.figure(figsize=(10, 10))
    proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -50.0))
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent([-80, 50, -15, -90], crs=ccrs.PlateCarree())
    return fig, ax

def add_gridlines_and_continents(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0.8, lw=0.35, color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.top_labels = None
    gl.right_labels = None
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='lightgray'), zorder=1)

def plot_genesis_density(ax):
    colors_linear = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', 
            '#FEEC9F', '#FDB567', '#F06744',  '#C1274A']
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)
    infile = '../periods_species_statistics/70W-no-continental/track_density/track_density.nc'
    ds = xr.open_dataset(infile)
    density = ds['incipient']
    levels = [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50]
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    lon, lat = density.lon, density.lat
    cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm, transform=ccrs.PlateCarree(), alpha=0.7)
    ax.contour(lon, lat, density, colors='#383838', linewidths=0.35, levels=levels, norm=norm, linestyles='dashed', transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cf, format='%g', orientation='horizontal', ticks=levels, pad=0.06)
    cbar.ax.tick_params(labelsize=12)
    return cf

def main():
    fig, ax = create_map_and_axes()
    add_gridlines_and_continents(ax)
    cf = plot_genesis_density(ax)

    regions = {
        "SE-BR": [(-52, -38, -37, -23)],
        "LA-PLATA": [(-69, -38, -52, -23)],
        "ARG": [(-70, -55, -50, -39)],
        "SE-SAO": [(-15, -55, 30, -37)],
        "SA-NAM": [(8, -33, 20, -21)],
        "AT-PEN": [(-65, -69, -44, -58)],
        "WEDDELL": [(-65, -85, -10, -72)]
    }

    for region, coords in regions.items():
        plot_region_box(ax, coords, edgecolor='#383838', label=region)

    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', facecolor='None')
    ax.coastlines()

    plt.tight_layout()

    fname = '../figures/manuscript_life-cycle/genesis_regions.png'
    plt.savefig(fname, bbox_inches='tight')
    print(f'Genesis regions saved in {fname}')

if __name__ == "__main__":
    main()
