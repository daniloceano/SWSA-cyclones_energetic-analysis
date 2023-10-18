import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import xarray as xr
import matplotlib as mpl
import numpy as np
import matplotlib.patches as mpatches

def load_seasonal_data(season):
    file_path = f'../periods_species_statistics/70W-no-continental/track_density/SAt_track_density_{season}.nc'
    ds = xr.open_dataset(file_path)
    return ds['incipient']

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
                text_lon = coord[2] + 17
            elif label == 'SE-BR':
                text_lon = coord[2] + 5
            elif label == 'AT-PEN':
                text_lon = coord[2] + 30
            elif label == 'SA-NAM':
                text_lat = coord[3] + 5
            elif label == 'SE-SAO':
                text_lon = coord[0] + 13
                text_lat = coord[3] + 2
            elif label == 'WEDDELL':
                text_lon = coord[2] + 25
                text_lat = coord[3] + 5
                
            ax.text(text_lon, text_lat, label, transform=ccrs.Geodetic(),
                     fontsize=16, color=edgecolor, fontweight='bold', ha='right', va='bottom')
            
def create_map_and_axes():
    fig = plt.figure(figsize=(15, 10))  # Adjust the figure size to accommodate subplots
    proj = ccrs.AlbersEqualArea(central_longitude=-30, central_latitude=-35, standard_parallels=(-20.0, -50.0))

    # Create a grid of subplots for seasons (2 rows, 2 columns)
    # axs = fig.subplots(2, 2, subplot_kw={'projection': proj})
    axs = fig.subplots(1, 2, subplot_kw={'projection': proj})
    for ax in axs.flat:
        ax.set_extent([-80, 50, -15, -90], crs=ccrs.PlateCarree())
    
    return fig, axs

def add_gridlines_and_continents(ax):
    gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0.8, lw=0.35, color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.top_labels = None
    gl.right_labels = None
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', edgecolor='face', facecolor='lightgray'), zorder=1)

def plot_genesis_density(fig, ax, seasonal_data, i):
    colors_linear = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', 
            '#FEEC9F', '#FDB567', '#F06744',  '#C1274A']
    cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)
    labels = ["(A)", "(B)", "(C)", "(D)"]
    levels = [0.1, 1, 2, 5, 8, 10, 12, 15, 17, 20, 25, 30, 35, 40]
    norm = mpl.colors.BoundaryNorm(levels, cmap.N)
    lon, lat = seasonal_data.lon, seasonal_data.lat
    cf = ax.contourf(lon, lat, seasonal_data, cmap=cmap, levels=levels, norm=norm, transform=ccrs.PlateCarree(), alpha=0.7)
    ax.contour(lon, lat, seasonal_data, colors='#383838', linewidths=0.35, levels=levels, norm=norm, linestyles='dashed', transform=ccrs.PlateCarree())
    props = dict(boxstyle='round', facecolor='white')
    ax.text(27, -3, labels[i], transform=ccrs.PlateCarree(), ha='left', va='top', fontsize=16, fontweight='bold', bbox=props)
    if i == 3:
        cbar_axes = fig.add_axes([0.15, 0.01, 0.7, 0.04])
        ticks = np.round(levels, decimals=2)
        colorbar = plt.colorbar(cf, cax=cbar_axes, ticks=ticks, format='%g', orientation='horizontal')
        colorbar.ax.tick_params(labelsize=12)
    return cf

def main():
    fig, axs = create_map_and_axes()
    #add_gridlines_and_continents(axs[0, 0])  # Apply gridlines and continents to the first subplot
    add_gridlines_and_continents(axs[0])

    # Define seasons and corresponding subplot positions
    # seasons = ['DJF', 'MAM', 'JJA', 'SON']
    # subplot_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    seasons = ['DJF','JJA']
    subplot_positions = [0, 1]

    i = 0
    for season, position in zip(seasons, subplot_positions):
        ax = axs[position]
        seasonal_data = load_seasonal_data(season)
        add_gridlines_and_continents(ax)
        plot_genesis_density(fig, ax, seasonal_data, i)

        regions = {
            "SE-BR": [(-52, -38, -37, -23)],
            "LA-PLATA": [(-69, -38, -52, -23)],
            "ARG": [(-70, -55, -50, -39)],
            "SE-SAO": [(-10, -55, 30, -35)],
            "SA-NAM": [(8, -33, 20, -21)],
            "AT-PEN": [(-65, -69, -44, -58)],
            "WEDDELL": [(-65, -85, -10, -72)]
        }

        for region, coords in regions.items():
            plot_region_box(ax, coords, edgecolor='#383838', label=region)

        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', facecolor='None')
        ax.coastlines()

        i+=1
    
    fname = '../figures/manuscript_life-cycle/genesis_regions.png'
    plt.savefig(fname, bbox_inches='tight')
    print(f'Genesis regions saved in {fname}')

if __name__ == "__main__":
    main()