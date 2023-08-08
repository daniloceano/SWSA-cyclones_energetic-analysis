# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    periods_density.py                                 :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/08 20:33:08 by Danilo            #+#    #+#              #
#    Updated: 2023/08/08 20:58:59 by Danilo           ###   ########.fr        #
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

import sys, os
import numpy as np
import pandas as pd
import time as tm
import datetime
from netCDF4 import Dataset
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage.filters import gaussian_filter
from glob import glob

def get_month(i):
        switcher={
        'DJF':np.array([1, 2, 12]),
        'MAM':np.array([3, 4, 5]),
        'JJA':np.array([6, 7, 8]),
        'SON':np.array([9, 10, 11]),
        'ALL':np.arange(1,13)
        }
        return switcher.get(i,"Invalid input")

def compute_density(tracks):
    #####
    # Computing gen/track density using KDE folowing the idea of K. Hodges (e.g., Hoskins and Hodges, 2005)
    # ----
    # (1) Creating a global grid with 128 x 64 (lon, lat): 2.5 degree
    k = 64
    longrd = np.linspace(-180, 180, 2 * k)
    latgrd = np.linspace(-87.863, 87.863 , k)
    tx, ty = np.meshgrid(longrd, latgrd)
    mesh = np.vstack((ty.ravel(), tx.ravel())).T
    mesh *= np.pi / 180.

    pos = tracks[['lat vor', 'lon vor']]
    x = pos['lon vor'].values
    y = pos['lat vor'].values

    # KDE ------
    # (2) Building the KDE for the positions
    h = np.vstack([y, x]).T
    h *= np.pi / 180.  # Convert lat/long to radians
    bdw = 0.05
    kde = KernelDensity(bandwidth=bdw, metric='haversine',
        kernel='gaussian', algorithm='ball_tree').fit(h)

    # We evaluate the kde() function on the grid.
    v = np.exp(kde.score_samples(mesh)).reshape((k, 2 * k))

    # Converting KDE values to scalled density
    # (a) cyclone number: multiply by total number of genesis (= pos.shape[0])
    # (b) area: divide by R ** 2 (R = Earth Radius)
    # (c) area: scalle by 1.e6
    # (d) time: divide by the number the time unit that you wish (month or year, as you wish)
    #
    # --> The final unit is genesis/area/time
    # - here, area is 10^6 km^2 and time is months (within the dataset)
    #
    # --> obs: the absolute values obtained here for the track density are not comparable with the
    # ones obtained by Hoskin and Hodges (2005) due to a diffent normalization. (this need to be improved)
    # 

    R = 6369345.0 * 1e-3 # Earth radius in meters at 40ºS (WGS 84 reference ellipsoid)
    factor = (1 / (R * R) ) * 1.e6
    ntime = num_years * num_months # months in data set

    density = v * pos.shape[0] * factor /  ntime 

    return density

# keys
savenc = True
plot = False

# set your data information
model = 'era5'
initial_year = 1979
final_year = 2020
seas = 'DJF'
nlabel= '24h_1000km'
add_fld = 1
# Define plot area
minlon  = -75. 
minlat  = -60  
maxlon  = -20  
maxlat  = -15

num_years = final_year - initial_year
num_months = 12

periods_directory = '../periods-energetics/BY_RG-all_raw/'
output_directory = '../figures/periods_statistics/heatmaps/'
os.makedirs(output_directory, exist_ok=True)

results_directories = ['../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
                    '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
                    '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/']

columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor',
                                'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd',
                                'lat 10spd', '10spd']

tracks = pd.DataFrame(columns = columns)
for results_directory in results_directories:  
            files = glob(f'{results_directory}*')
            for file in files:  
                try:
                    tmp = pd.read_csv(file)
                except:
                    continue
                tmp.columns = columns
                tracks = pd.concat([tracks,tmp])
                
x = tracks['lon vor'].values 
tracks['lon vor'] = np.where(x > 180, x - 360, x)

# no. of tracks
ntracks = len(tracks.groupby('track_id').first())

print(f'Number of tracks: {ntracks}')

density = compute_density(tracks)

# if savenc:
# 	# -----------------
# 	# save netcdf 
# 	print('writing netcdf')
# 	fout = '{}/tden_kde_{}_{}.nc'.format(dirout, model, period)
# 	os.system('rm -f {}'.format(fout))	
# 	fnetcdf="NETCDF4"
# 	# open a new netCDF file for writing.
# 	ncfile = Dataset(fout, "w", format=fnetcdf) 
# 	ncfile.description='Genesis density computed by KDE (sklearn package)'
# 	ncfile.author = "script by cbgramcianinov@gmail.com"
# 	ncfile.history = 'created ' + tm.ctime(tm.time())
# 	ncfile.source = '{}'.format(model)
# 	# create the lat and lon dimensions.
# 	Latitude  = ncfile.createDimension( 'lat' , k ) 
# 	Longitude  = ncfile.createDimension( 'lon' , 2 * k ) 
# 	lats = ncfile.createVariable('lat', np.float32, ('lat',)) 
# 	lons = ncfile.createVariable('lon', np.float32, ('lon',)) 
# 	# Assign units attributes to coordinate var data. This attaches a text attribute to each of the coordinate variables, containing the units.
# 	lats.units = 'degrees_north'
# 	lons.units = 'degrees_east'

# 	# write data to coordinate vars.
# 	lats[:] = latgrd
# 	lons[:] = longrd
# 	# create  variable
# 	density_f  = ncfile.createVariable('density', np.float32, ('lat','lon'),fill_value=1.e+20)

# 	# write data to variables.
# 	density_f[:,:] = density[:,:]

# 	# close the file
# 	ncfile.close()
# 	print('netcdf ok')

# if plot:
# 	print('plotting')
# 	#density = np.where(density < 1, np.nan, density)
# 	minlon  = -75. 
# 	minlat  = -60  
# 	maxlon  = -20  
# 	maxlat  = -10
# 	# ---- PLOT
# 	sl=20
# 	matplotlib.rcParams.update({'font.size': sl}); plt.rc('font', size=sl) 
# 	matplotlib.rc('xtick', labelsize=sl); matplotlib.rc('ytick', labelsize=sl); 	
	
# 	crs = ccrs.PlateCarree()			
# 	#cmap = plt.cm.get_cmap('magma_r')
# 	cmap = plt.cm.YlOrRd
# 	#clev = np.arange(0, np.nanmax(density), 1)
# 	#llev = np.arange(0, np.nanmax(density), 1); #llev2 = np.arange(5, np.nanmax(density), 10)
# 	clev = np.linspace(0, np.nanmax(density), 20)
# 	llev = np.linspace(0, np.nanmax(density), 20);

# 	fig = plt.figure(1, figsize=(20,20))

# 	ax = plt.subplot(projection=crs)

# 	#ax.set_title("Genesis density")
# 	ax.set_extent([minlon, maxlon, minlat, maxlat], crs)
# 	ax.outline_patch.set_linewidth(2) 
# 	ax.outline_patch.set_linestyle('solid')
# 	ax.add_feature(cfeature.BORDERS, linewidth=1)
# 	ax.add_feature(cfeature.COASTLINE, linewidth=1)
# 	ax.add_feature(cfeature.STATES, linewidth=1)
# 	ax.add_feature(cfeature.LAND, facecolor='grey', alpha = 0.5)
# 	gl = ax.gridlines(crs=crs, draw_labels=True, linewidth=0.75, color='k', linestyle=':')
# 	gl.bottom_labels = False
# 	gl.right_labels = False
# 	gl.xlabel_style = {'size': sl, 'color': 'k'}
# 	gl.ylabel_style = {'size': sl, 'color': 'k'}

# 	#sig = ax.plot(x, y, marker='.', linestyle='None', color='gray', markersize=2, alpha=0.5)

# 	cf = ax.contourf(tx, ty, density, clev, cmap=cmap, extend='both', transform=crs, alpha=0.85)
# 	cbar = plt.colorbar(cf, orientation='horizontal', pad=0.05, aspect=30, extendrect=False, extend='both')
# 	cbar.set_label('track density'); cbar.ax.tick_params(labelsize=sl)


# 	cs = ax.contour(tx, ty, density, llev, colors='k',
# 			linewidths=2, linestyles='-', transform=crs)
# 	plt.clabel(cs, fontsize=sl, inline=1, inline_spacing=3, fmt='%i',
# 		   rightside_up=True, use_clabeltext=True)
# 	#cs2 = ax.contour(tx, ty, density, llev2, colors='k',
# 	#		linewidths=2, linestyles='-', transform=crs)

# 	plt.savefig('{}/tden_kde_{}_{}.png'.format(dirout, model, period), dpi=300, facecolor='w', edgecolor='w',orientation='portrait', format='png',transparent=False, bbox_inches='tight', pad_inches=0.1)
# 	plt.close()