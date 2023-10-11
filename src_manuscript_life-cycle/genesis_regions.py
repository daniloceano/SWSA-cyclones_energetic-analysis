import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import shapely.geometry as sgeom
import matplotlib.colors as mcolors
import xarray as xr
import matplotlib as mpl

def plot_region_box(ax, coords, edgecolor, label=None):
    for coord in coords:
        box = sgeom.box(*coord)
        ax.add_geometries([box], ccrs.PlateCarree(), edgecolor=edgecolor, facecolor='none', alpha=0.8)
        if label:
            # Adjust the text coordinates
            text_lat = coord[1]  # Subtract 2 degrees
            text_lon = coord[0]  # Add 1 degree
            ax.text(text_lon, text_lat, label, transform=ccrs.PlateCarree(), 
                    fontsize=14, color=edgecolor, fontweight='bold', ha='left', va='bottom')

colors_linear = ['#AFC4DA', '#4471B2', '#B1DFA3', '#EFF9A6', 
            '#FEEC9F', '#FDB567', '#F06744',  '#C1274A']
cmap = mcolors.LinearSegmentedColormap.from_list("", colors_linear)

fig = plt.figure(figsize=(10, 10))
proj = ccrs.AzimuthalEquidistant(central_longitude=-30, central_latitude=-35)
ax = fig.add_subplot(1, 1, 1, projection=proj)

# Set the extent
ax.set_extent([-80, 30, -15, -90], crs=ccrs.PlateCarree())

# Add gridlines
gl = ax.gridlines(draw_labels=True, zorder=2, linestyle='dashed', alpha=0.8, lw=0.35, color='#383838')
gl.xlabel_style = {'size': 14, 'color': '#383838'}
gl.ylabel_style = {'size': 14, 'color': '#383838'}
gl.top_labels = None
gl.right_labels = None

# Add gray shaded background to continents
ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m',
                                             edgecolor='face', facecolor='lightgray'), zorder=1)

# Plot genesis density
infile = '../periods_species_statistics/70W-no-continental/track_density/track_density.nc'
ds = xr.open_dataset(infile)
density = ds['incipient']
levels = [0.1, 1, 2, 5, 8, 10, 15, 20, 30, 40, 50]
norm = mpl.colors.BoundaryNorm(levels, cmap.N)
lon, lat = density.lon, density.lat
cf = plt.contourf(lon, lat, density, cmap=cmap, levels=levels, norm=norm,
                   transform=ccrs.PlateCarree(), alpha=0.7)
ax.contour(lon, lat, density, colors='#383838', linewidths=0.35,  levels=levels, norm=norm,
            linestyles='dashed', transform=ccrs.PlateCarree())

# Region coordinates and labels
regions = {
    "SE-BR": [(-52, -23, -37, -38)],
    "LA-PLATA": [(-69, -23, -52, -38)],
    "ARG": [(-70, -39, -50, -55)],
    "SE-SAO": [(-15, -37, 30, -55)],
    "SA-NAM": [(8, -21, 20, -33)],
    "AT-PEN": [(-65, -58, -44, -69)],
    "WEDDELL": [(-65, -72, -10, -85)]
}

# Add boxes for regions
for region, coords in regions.items():
    plot_region_box(ax, coords, edgecolor='#383838', label=region)

# Add country borders
ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray', facecolor='None')

ax.coastlines()

cbar_axes = fig.add_axes([0.15, 0.05, 0.7, 0.02])
colorbar = plt.colorbar(cf, cax=cbar_axes, format='%g',  orientation='horizontal')
colorbar.ax.tick_params(labelsize=12)

fname = '../figures/manuscript_life-cycle/genesis_regions.png'
plt.savefig(fname, bbox_inches='tight')
print(f'Genesis regions saved in {fname}')
