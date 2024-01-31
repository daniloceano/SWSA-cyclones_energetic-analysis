# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    residual_study_case.py                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/31 08:54:11 by daniloceano       #+#    #+#              #
#    Updated: 2024/01/31 09:23:07 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import cdsapi
import math
import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import cmocean.cm as cmo
import numpy as np
from glob import glob
import matplotlib.colors as colors
from metpy.units import units
from metpy.calc import vorticity
from metpy.constants import g

TRACKS_DIRECTORY = "../processed_tracks_with_periods/"
STUDY_CASE = 19920876
CRS = ccrs.PlateCarree() 

def get_cdsapi_data(track, infile) -> xr.Dataset:

    # Extract bounding box (lat/lon limits) from track
    min_lat, max_lat = track['lat vor'].min(), track['lat vor'].max()
    min_lon, max_lon = track['lon vor'].min(), track['lon vor'].max()

    # Apply a 15-degree buffer and round to nearest integer
    buffered_max_lat = math.ceil(max_lat + 15)
    buffered_min_lon = math.floor(min_lon - 15)
    buffered_min_lat = math.floor(min_lat - 15)
    buffered_max_lon = math.ceil(max_lon + 15)

    # Define the area for the request
    area = f"{buffered_max_lat}/{buffered_min_lon}/{buffered_min_lat}/{buffered_max_lon}" # North, West, South, East. Nort/West/Sout/East

    pressure_levels = ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70',
                       '100', '125', '150', '175', '200', '225', '250', '300', '350',
                       '400', '450', '500', '550', '600', '650', '700', '750', '775',
                       '800', '825', '850', '875', '900', '925', '950', '975', '1000']
    
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature",
                 "vertical_velocity", "geopotential"]
    

    # Convert unique dates to string format for the request
    dates = pd.to_datetime(track['date'].tolist())
    start_date = dates[0].strftime("%Y%m%d")
    end_date = dates[-1].strftime("%Y%m%d")
    time_range = f"{start_date}/{end_date}"
    time_step = '3'

    # Log track file bounds and requested data bounds
    print(f"Track File Limits: max_lon (east):  min_lon (west): {min_lon}, max_lon (west): {max_lon}, min_lat (south): {min_lat}, max_lat (north): {max_lat}")
    print(f"Buffered Data Bounds: min_lon (west): {buffered_min_lon}, max_lon (east): {buffered_max_lon}, min_lat (south): {buffered_min_lat}, max_lat (north): {buffered_max_lat}")
    print(f"Requesting data for time range: {time_range}, and time step: {time_step}...")

    # Load ERA5 data
    print("Retrieving data from CDS API...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "pressure_level": pressure_levels,
            "date": time_range,
            "area": area,
            'time': f'00/to/23/by/{time_step}',
            "variable": variables,
        }, infile # save file as passed in arguments
    )

    if not os.path.exists(infile):
        raise FileNotFoundError("CDS API file not created.")
    
    ds = xr.open_dataset(infile)

    return ds

def map_decorators(ax):
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',alpha=0.7,
                 linewidth=0.5, color='#383838')
    gl.xlabel_style = {'size': 14, 'color': '#383838'}
    gl.ylabel_style = {'size': 14, 'color': '#383838'}
    gl.top_labels = None
    gl.right_labels = None

def plot_zeta(ax, zeta, lat, lon, hgt):
    cmap = cmo.balance
    # plot contours
    cf1 = ax.contourf(lon, lat, zeta, cmap=cmap,norm=norm,levels=51,
                      transform=CRS) 
    plt.colorbar(cf1, orientation='vertical', shrink=0.5)
    cs = ax.contour(lon, lat, hgt, levels=11, colors='#344e41', 
                    linestyles='dashed',linewidths=1.3,
                    transform=CRS)
    ax.clabel(cs, cs.levels, inline=True, fontsize=10)

def draw_box_map(u, v, zeta, hgt, lat, lon):
    plt.close('all')
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=CRS)
    fig.add_axes(ax)
    
    plot_zeta(ax, zeta, lat, lon, hgt)
    ax.streamplot(lon.values, lat.values, u.values, v.values, color='#2A1D21',
              transform=CRS)
    map_decorators(ax)

track_file = glob(f"{TRACKS_DIRECTORY}/*{str(STUDY_CASE)[:4]}*.csv")
tracks = pd.concat([pd.read_csv(f) for f in track_file])
track = tracks[tracks['track_id'] == STUDY_CASE]

infile = f"{STUDY_CASE}.nc"

if os.path.exists(infile):
    ds = xr.open_dataset(infile)
else:
    ds = get_cdsapi_data(track, infile)

# Slice the dataset to match the track
ds = ds.sel(time=slice(track['date'].min(), track['date'].max()))

# Get lat and lon arrays
lat, lon = ds['latitude'], ds['longitude']

# Slice for 850 hPa
u_850, v_850 = ds['u'].sel(level=850), ds['v'].sel(level=850)
zeta_850 = vorticity(u_850, v_850).metpy.dequantify() 

# Set colorbar limits
norm = colors.TwoSlopeNorm(vmin=-zeta_850.max(), vcenter=0,vmax=zeta_850.max())

for time in ds.time.values[:4]:

    # Get the current time and format it
    time  = pd.to_datetime(time)

    # Get the data for the current time
    iu_850, iv_850 = u_850.sel(time=time), v_850.sel(time=time)
    izeta_850 = zeta_850.sel(time=time)
    ihgt_850 = ds['z'].sel(level=850).sel(time=time) / g

    # Plot the data
    draw_box_map(iu_850, iv_850, izeta_850, ihgt_850, lat, lon)

    # Add the system position from the track 
    itrack = track[track['date'] == str(time)]
    ax = plt.gca()
    ax.scatter(itrack['lon vor'], itrack['lat vor'], c='r', marker='o', s=50, zorder=3)

    # Title: Current time
    timestr = time.strftime("%Y-%m-%d %H:%M")
    plt.title(timestr)

    print()
