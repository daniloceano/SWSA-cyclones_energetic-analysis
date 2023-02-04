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
from sklearn import preprocessing

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

#testfile = '/home/daniloceano/Documents/Programs_and_scripts/data_etc/netCDF_files/19820684_NCEP-R2.nc'
#testtrack = '/home/daniloceano/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/tracks_LEC-format/intense/track_19820684' 
  
testfile = '/Users/danilocoutodsouza/Documents/USP/Programs_and_scripts/data_etc/netCDF_data/Reg1-Representative_ERA5.nc'
testtrack = '/Users/danilocoutodsouza/Documents/USP/Programs_and_scripts/lorenz-cycle/inputs/track_Reg1-Representative'
varlist = '/Users/danilocoutodsouza/Documents/USP/Programs_and_scripts/lorenz-cycle/inputs/fvars_ERA5'

dfVars = pd.read_csv(varlist,sep= ';',index_col=0,header=0)
LonIndexer = dfVars.loc['Longitude']['Variable']
LatIndexer = dfVars.loc['Latitude']['Variable']
TimeIndexer = dfVars.loc['Time']['Variable']
LevelIndexer = dfVars.loc['Vertical Level']['Variable']
                                            
da = convert_lon(xr.open_dataset(testfile),LonIndexer)
da850 = da.sel({LevelIndexer:850})
u850 = da850[dfVars.loc['Eastward Wind Component']['Variable']] * units('m/s')
v850 = da850[dfVars.loc['Northward Wind Component']['Variable']] * units('m/s')
zeta850 = vorticity(u850,v850)
dZdt = zeta850.differentiate(TimeIndexer,datetime_unit='h') 

track = pd.read_csv(testtrack, parse_dates=[0],delimiter=';',index_col='time')

min_lat, max_lat = track.Lat.min(), track.Lat.max()
min_lon, max_lon = track.Lon.min(), track.Lon.max()

min_zeta = []
dzdt = []
lats, lons = [], []
times = []
for t in zeta850[TimeIndexer]:
    datestr = pd.to_datetime(t.values)
    if datestr in track.index:
        lat,lon = track.loc[datestr]
        lats.append(lat), lons.append(lon)
        
        min_zeta.append(float(zeta850.sel({TimeIndexer:t}, method='nearest'
                                    ).sel({LatIndexer:lat,LonIndexer:lon},
                                          method='nearest')))
        
        dzdt.append(float(dZdt.sel({TimeIndexer:t}, method='nearest'
                                    ).sel({LatIndexer:lat,LonIndexer:lon},
                                          method='nearest')))                            
        
        
        times.append(datestr)
    
plt.close('all')
datacrs = ccrs.PlateCarree()
fig = plt.figure(figsize=(10, 15))
gs = gridspec.GridSpec(2, 3)

# Track
ax = fig.add_subplot(gs[0, :], projection=datacrs,frameon=True)
ax.set_extent([-80, 20, 0, -50], crs=datacrs) 
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
cb = plt.colorbar(scatter, pad=0.07, orientation='vertical',
                  label=' 850 hPa vorticity (ζ)')
ax.title.set_text('Track')

# Zeta 850
ax = fig.add_subplot(gs[1, 0],frameon=True)
kwargs={'title':'min_Zeta', 'labels':['ζ', r'$ζ_f$']}
zeta_fil = savgol_filter(min_zeta, 39, 2, mode="nearest")
plot_timeseries(ax, times, *[min_zeta, zeta_fil], **kwargs)

# dZdt
ax = fig.add_subplot(gs[1, 1],frameon=True)
kwargs={'title':'dzdt','labels':[r'$\frac{∂ζ_f}{∂t}$']}
zeta_fil = xr.DataArray(np.array(zeta_fil),coords={'time':times})
dzfil_dt = zeta_fil.differentiate('time',datetime_unit='h') 
dzdt_fil = savgol_filter(dzdt, 39, 3, mode="nearest")
plot_timeseries(ax, times, *[dzfil_dt], **kwargs)

# dZdt2
ax = fig.add_subplot(gs[1, 2],frameon=True)
kwargs={'title':'dzdt','labels':[r'$\frac{∂^{3}ζ_f}{∂t^{3}}$',
                                 r'$\frac{∂^{3}ζ_f}{∂t^{3}}$ f' ]}
dzfil_dt2 = dzfil_dt.differentiate('time',datetime_unit='h') 
dzfil_dt3 = dzfil_dt2.differentiate('time',datetime_unit='h')
plot_timeseries(ax, times, *[dzfil_dt3], **kwargs)
plt.tight_layout()
fig.align_labels()

plt.savefig('test_periods.png',dpi=500)

## Periods
intensification = dzfil_dt.time.where(dzfil_dt < 0, drop=True).values
intensification = pd.to_datetime(intensification)
decay = dzfil_dt.time.where(dzfil_dt > 0, drop=True).values
decay = pd.to_datetime(decay)

dzdt3_abs = np.abs(dzfil_dt3)
dzdt3_norm = (dzdt3_abs-dzdt3_abs.min())/(dzdt3_abs.max()-dzdt3_abs.min())
da_dzdt3_norm = xr.DataArray(np.array(dzdt3_norm),coords={'time':times})

# For the third derivative, get the 10% smaller values
dz10th =  da_dzdt3_norm[da_dzdt3_norm < da_dzdt3_norm.quantile(.2)][TimeIndexer].values
# Remove not continuous values
dt_original = dzfil_dt3[TimeIndexer][1] - dzfil_dt3[TimeIndexer][0]
mature = []
for i in range(len(dz10th)-1):
    dt = dz10th[i+1] - dz10th[i]
    if dt != dt_original:
        pass
    else:
        mature.append(dz10th[i])
        mature.append(dz10th[i+1])
mature = pd.to_datetime(mature)

# Now, remove mature periods from intensification and decay
intensification = pd.to_datetime(
    [x for x in intensification if(x <= mature.min())])
decay = pd.to_datetime([x for x in decay if (x >= mature.max())])

# plot periods
plt.close('all')
fig = plt.figure(figsize=(10, 15))
gs = gridspec.GridSpec(1, 1)
ax = fig.add_subplot(gs[0, 0],frameon=True)
kwargs={'title':'min_Zeta', 'labels':['ζ', r'$ζ_f$']}
plot_timeseries(ax, times, *[min_zeta, zeta_fil], **kwargs)

for period,str in zip([intensification, mature, decay],
                      ['int.', 'mature', 'decay']):
    ax.axvline(period[0], c='gray', linestyle='dashed')
    ax.axvline(period[-1], c='gray', linestyle='dashed')
    
    pos = round(len(period)/2)
    ax.text(period[pos], max(min_zeta)*1.3, str, weight='bold',
            horizontalalignment='right')

plt.savefig('test_periods2.png',dpi=500)
