#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:17:29 2023

@author: daniloceano
"""

import xarray as xr
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

import cmocean.cm as cmo

from metpy.calc import vorticity
from metpy.units import units

from scipy.signal import savgol_filter    
from scipy.interpolate import interp1d

import cartopy.crs as ccrs
import cartopy

def convert_lon(xr,LonIndexer):
    xr.coords[LonIndexer] = (xr.coords[LonIndexer] + 180) % 360 - 180
    xr = xr.sortby(xr[LonIndexer])
    return xr

def plot_timeseries(ax, x, *args, **kwargs):
    colors = ['#1d3557', '#d62828', '#606c38', '#f77f00']
    ls = [2.5, 2, 2, 2]
    for i in range(len(args)):
        ax.plot(x,args[i],c=colors[i],linewidth=ls[i],
                label=kwargs['labels'][i])
    plt.legend()
    plt.grid(linewidth=0.5, alpha=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()

testfile = '/home/daniloceano/Documents/Programs_and_scripts/data_etc/netCDF_files/19820684_NCEP-R2.nc'
testtrack = '/home/daniloceano/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/tracks_LEC-format/intense/track_19820684' 
   
da = convert_lon(xr.open_dataset(testfile),'lon_2')
da850 = da.sel(lv_ISBL3=850)
u850 = da850.U_GRD_2_ISBL * units('m/s')
v850 = da850.V_GRD_2_ISBL * units('m/s')
zeta850 = vorticity(u850,v850)
dZdt = zeta850.differentiate('initial_time0_hours',datetime_unit='h') 

track = pd.read_csv(testtrack, parse_dates=[0],delimiter=';',index_col='time')

min_lat, max_lat = track.Lat.min(), track.Lat.max()
min_lon, max_lon = track.Lon.min(), track.Lon.max()

min_zeta = []
dzdt = []
lats, lons = [], []
times = []
for t in zeta850.initial_time0_hours:
    datestr = pd.to_datetime(t.values)
    if datestr in track.index:
        lat,lon = track.loc[datestr]
        lats.append(lat), lons.append(lon)
        
        min_zeta.append(float(zeta850.sel(initial_time0_hours=t, method='nearest'
                                    ).sel(lat_2=lat,lon_2=lon,method='nearest')))
        
        dzdt.append(float(dZdt.sel(initial_time0_hours=t, method='nearest'
                                    ).sel(lat_2=lat,lon_2=lon,method='nearest')))                                 
        
        
        times.append(datestr)
    
plt.close('all')
datacrs = ccrs.PlateCarree()
fig = plt.figure(figsize=(10, 15))
gs = gridspec.GridSpec(2, 2)

# Track
ax = fig.add_subplot(gs[0, :], projection=datacrs,frameon=True)
ax.set_extent([min_lon-10, max_lon+10, max_lat+10, min_lat-10], crs=datacrs) 
ax.coastlines(zorder = 1)
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN,facecolor=("lightblue"))
gl = ax.gridlines(draw_labels=True,zorder=2,linestyle='dashed',alpha=0.8,
             color='#383838')
gl.xlabel_style = {'size': 14, 'color': '#383838'}
gl.ylabel_style = {'size': 14, 'color': '#383838'}
gl.bottom_labels = None
gl.right_labels = None
ax.plot(lons,lats,'-',c='k')
scatter = ax.scatter(lons,lats,zorder=100,cmap=cmo.deep_r,c=min_zeta)
plt.colorbar(scatter, pad=0.07, orientation='vertical', shrink=0.5,
             label= 'ζ')
ax.title.set_text('Track')

# Zeta 850
ax = fig.add_subplot(gs[1, 0],frameon=True)
kwargs={'title':'min_Zeta', 'labels':['ζ','ζ (f)', 'ζ (f2)']}
# zeta_mean = pd.Series(min_zeta).rolling(5).mean()
zeta_spline = interp1d(range(len(min_zeta)), min_zeta, kind='cubic')
zeta_fil = savgol_filter(min_zeta, 39, 2, mode="nearest")
zeta_fil2 = savgol_filter(zeta_fil, 39, 2, mode="nearest")
variables = [min_zeta, zeta_fil, zeta_fil2]
plot_timeseries(ax, times, *variables, **kwargs)

# dZdt
ax = fig.add_subplot(gs[1, 1],frameon=True)
kwargs={'title':'dzdt','labels':['∂ζ/∂t','∂ζ/∂t (f)',
                                 '∂ζ(f2)/∂t','∂ζ(f)2/∂t (f)']}
# dzdt_mean = pd.Series(dzdt).rolling(5).mean()
dzdt_fil = savgol_filter(dzdt, 39, 3, mode="nearest")
dzetafil_dt = pd.Series(zeta_fil2).diff()
dzetafil_dt_fil = savgol_filter(dzetafil_dt, 39, 3, mode="nearest")
variables = [dzdt, dzdt_fil, dzetafil_dt, dzetafil_dt_fil]
plot_timeseries(ax, times, *variables, **kwargs)

# plt.tight_layout()
fig.align_labels()