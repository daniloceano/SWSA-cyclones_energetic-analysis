#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 17:17:29 2023

@author: daniloceano
"""
import glob
import matplotlib 

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

def plot_track(lons, lats, min_zeta, fname):
    plt.close('all')
    datacrs = ccrs.PlateCarree()
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(1, 1)
    ax = fig.add_subplot(gs[:, :], projection=datacrs,frameon=True)
    ax.set_extent([-80, 40, 0, -70], crs=datacrs) 
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
    plt.colorbar(scatter, pad=0.07, orientation='vertical',fraction=0.026,
                      label=' 850 hPa vorticity (ζ)')
    plt.savefig('../periods/'+fname+'_track.png',dpi=500)

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
    
def plot_vorticity(min_zeta, dzdt, fname):
    font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}

    matplotlib.rc('font', **font)
    
    window_lenght = round(len(min_zeta)/2)
    if (window_lenght % 2) == 0:
        window_lenght += 1
        
    wl = 21

    # min_zeta = min_zeta.resample('3H').mean()    
    times = min_zeta.index

    plt.close('all')
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)
    
    # Zeta 850
    ds_min_zeta = xr.DataArray(np.array(min_zeta[0]),coords={'time':times})
    ax = fig.add_subplot(gs[0, 0],frameon=True)
    kwargs={'title':'min_Zeta', 'labels':['ζ', r'$ζ_f$']}
    zeta_fil = savgol_filter(min_zeta[0], window_lenght, 3, mode="nearest")
    plot_timeseries(ax, times, *[min_zeta, zeta_fil], **kwargs)
    
    # dZdt
    ax = fig.add_subplot(gs[0, 1],frameon=True)
    kwargs={'title':'dzdt','labels':[r'$\frac{∂ζ_f}{∂t}$',r'$\frac{∂ζ_f}{∂t}_f$']}
    zeta_fil = xr.DataArray(np.array(zeta_fil),coords={'time':times})
    dzfil_dt = zeta_fil.differentiate('time',datetime_unit='h') 
    dzfildt_fil = savgol_filter(dzfil_dt, window_lenght, 2, mode="nearest")
    plot_timeseries(ax, times, *[dzfil_dt, dzfildt_fil], **kwargs)
    
    # dZdt2
    ax = fig.add_subplot(gs[1, 0],frameon=True)
    kwargs={'title':'dzdt','labels':[r'$\frac{∂^{2}ζ_f}{∂t^{2}}$',
                                      r'$\frac{∂^{3}ζ_f}{∂t^{3}}_f$']}
    dzfildt_fil = xr.DataArray(np.array(dzfildt_fil),coords={'time':times})
    dzfil_dt2 = dzfil_dt.differentiate('time',datetime_unit='h') 
    dzfildt2_fil = savgol_filter(dzfil_dt2, window_lenght, 2, mode="nearest")
    plot_timeseries(ax, times, *[dzfil_dt2, dzfildt2_fil], **kwargs)
    
    # dZdt3
    ax = fig.add_subplot(gs[1, 1],frameon=True)
    kwargs={'title':'dzdt','labels':[r'$\frac{∂^{3}ζ_f}{∂t^{3}}$',
                                      r'$\frac{∂^{3}ζ_f}{∂t^{3}}_f$']}
    dzfil_dt3 = dzfil_dt.differentiate('time',datetime_unit='h'
                                    ).differentiate('time',datetime_unit='h')
    dzfil_dt3_fil = savgol_filter(dzfil_dt3, wl, 2, mode="nearest")
    plot_timeseries(ax, times, *[dzfil_dt3, dzfil_dt3_fil], **kwargs)

    plt.tight_layout()
    fig.align_labels()
    
    plt.savefig('../periods/'+fname+'_zeta.png',dpi=500)
    
def get_periods(min_zeta):
    
    # Set filter window_lenght to be half the of timeseries lenght
    window_lenght = round(len(min_zeta)/2)
    if (window_lenght % 2) == 0:
        window_lenght += 1
        
    # Transform Vorticity into DataArray and get derivatives
    zeta_fil = xr.DataArray(np.array(
        savgol_filter(min_zeta[0], window_lenght, 3, mode="nearest")),
        coords={'time':times})
    dzfil_dt = zeta_fil.differentiate('time',datetime_unit='h') 
    dzfil_dt2 = dzfil_dt.differentiate('time',datetime_unit='h')
    dzfil_dt3 = dzfil_dt2.differentiate('time',datetime_unit='h')
    
    # Get the peiod of minimum vorticity. For splititng data in half.
    # The timesteps before it cannot be decaying stages. 
    minimum = zeta_fil.min()
    tmin_zeta = zeta_fil[zeta_fil==minimum].time.values
    zeta_fill_first_half = zeta_fil.where(zeta_fil.time < tmin_zeta
                                          ).dropna(dim='time')
    # The same goes for everything after the local minimum: it cannot be
    # intensification or inicipient periods
    zeta_fill_second_half = zeta_fil.where(zeta_fil.time > tmin_zeta
                                          ).dropna(dim='time')
    
    # Intensification is given by the period between local minima and maxima
    # of the vorticity third derivative, but for the first half of the
    # vorticity time series
    dzfil_dt3_fh = dzfil_dt3.sel(time=zeta_fill_first_half.time)
    intensification_end = dzfil_dt3_fh.idxmin().values
    intensification_start = dzfil_dt3_fh.idxmax().values
    intensification = zeta_fil.time.sel(
        time=slice(intensification_start,intensification_end))
    
    # Everything before the intensification start is incipient phase
    incipient = zeta_fill_first_half.time.where(
        zeta_fill_first_half.time < intensification_start).dropna(dim='time')
    incipient_start = zeta_fil.time[0].values
    incipient_end = incipient[-1].values
    incipient = zeta_fil.time.sel(time=slice(incipient_start,incipient_end))
    
    # For decaying phase, it will be followed the same procedure as for the
    # intensification, but inverted: the local minima starts the intensfication
    # period, which will follow until the end of data
    dzfil_dt3_sh = dzfil_dt3.sel(time=zeta_fill_second_half.time)
    decay_start = dzfil_dt3_sh.idxmin().values
    decay_end = zeta_fil.time[-1].values
    decay = zeta_fil.time.sel(time=slice(decay_start,decay_end))
    
    # Mature stage is everything between end of intensification and start of
    # the decaying stage
    dt = zeta_fil.time[1] - zeta_fil.time[0]
    mature_start = zeta_fil.sel(time=intensification_end+dt).time.values
    mature_end = zeta_fil.sel(time=decay_start-dt).time.values
    mature = zeta_fil.time.sel(time=slice(mature_start,mature_end))
    
    # Plot results
    plt.close('all')
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111,frameon=True)
    colors = ['k', '#134074', '#d62828', '#f7b538', '#5b8e7d',]
    
    # Plot filtered vorticity and vorticity on background
    ax.plot(min_zeta.index, min_zeta, c='gray', linewidth=1,label=r'$ζ$',
            zorder=98)
    
    ax2 = ax.twinx()
    ax2.plot(dzfil_dt3.time, dzfil_dt3, c='#283618', linewidth=2, alpha=0.8,
             label=r'$\frac{∂^{3}ζ_f}{∂t^{3}}$')
    
    ax.plot(zeta_fil.time, zeta_fil,c=colors[0], linewidth=4,label=r'$ζ_f$',
            zorder=101)    
    
    # Plot periods
    y = np.arange(min_zeta[0].min(),min_zeta[0].max()+1e-5,1e-5)
    ax.fill_betweenx(y, incipient_start, intensification_start, 
                     facecolor=colors[1], alpha=0.2, label='incipient')
                
    ax.fill_betweenx(y, intensification_start, mature_start,
                     facecolor=colors[2], alpha=0.2, label='intensification')

    ax.fill_betweenx(y, mature_start, decay_start,
                     facecolor=colors[3], alpha=0.2, label='mature')
    
    ax.fill_betweenx(y, decay_start, decay_end,
                     facecolor=colors[4], alpha=0.2, label='decay')
    
    plt.xlim(zeta_fil.time[0].values, zeta_fil.time[-1].values)
    ax.set_ylim(y[0],y[-1])
    ax.legend()
    ax2.legend()
    plt.grid(linewidth=0.5, alpha=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()
    
    plt.savefig('../periods/'+fname+'_preiods.png',dpi=500)
    
    return incipient, intensification, mature, decay
    
# def get_periods(min_zeta):
    
#     # min_zeta = min_zeta.resample('3H').mean() 
#     times = min_zeta.index
    
#     window_lenght = round(len(min_zeta)/2)
#     if (window_lenght % 2) == 0:
#         window_lenght += 1
    
#     zeta_fil = savgol_filter(min_zeta[0], window_lenght, 3, mode="nearest")
#     zeta_fil = xr.DataArray(np.array(zeta_fil),coords={'time':times})
#     # dzfil_dt = zeta_fil.differentiate('time',datetime_unit='h') 
#     ds_min_zeta = xr.DataArray(np.array(min_zeta[0]),coords={'time':times})
#     dz_dt = ds_min_zeta.differentiate('time',datetime_unit='h') 
    
#     # dzfil_dt2 = dzfil_dt.differentiate('time',datetime_unit='h') 
#     # dzfil_dt3 = dzfil_dt2.differentiate('time',datetime_unit='h')
#     dz_dt2 = dz_dt.differentiate('time',datetime_unit='h') 
#     dz_dt3 = dz_dt2.differentiate('time',datetime_unit='h') 
    
#     # intensification = dzfil_dt.time.where(dzfil_dt < 0, drop=True).values
#     # intensification = pd.to_datetime(intensification)
#     # decay = dzfil_dt.time.where(dzfil_dt > 0, drop=True).values
#     # decay = pd.to_datetime(decay)
    
#     intensification = dz_dt.time.where(dz_dt < 0, drop=True).values
#     intensification = pd.to_datetime(intensification)
#     decay = dz_dt.time.where(dz_dt > 0, drop=True).values
#     decay = pd.to_datetime(decay)
    
#     # dzdt3_abs = np.abs(dzfil_dt3)
#     # dzdt3_norm = (dzdt3_abs-dzdt3_abs.min())/(dzdt3_abs.max()-dzdt3_abs.min())
#     # da_dzdt3_norm = xr.DataArray(np.array(dzdt3_norm),coords={'time':times})
    
#     # # For the third derivative, get the 10% smaller values
#     # dz10th =  da_dzdt3_norm[da_dzdt3_norm < da_dzdt3_norm.quantile(.2)][TimeIndexer].values
#     # # Remove not continuous values
#     # dt_original = dzfil_dt3[TimeIndexer][1] - dzfil_dt3[TimeIndexer][0]
#     # mature = []
#     # for i in range(len(dz10th)-1):
#     #     dt = dz10th[i+1] - dz10th[i]
#     #     if dt != dt_original:
#     #         pass
#     #     else:
#     #         mature.append(dz10th[i])
#     #         mature.append(dz10th[i+1])
#     # mature = pd.to_datetime(mature)
    
#     # Now, remove mature periods from intensification and decay
#     intensification = pd.to_datetime(
#         [x for x in intensification if(x <= mature.min())])
#     decay = pd.to_datetime([x for x in decay if (x >= mature.max())])
    
#     # plot periods
#     plt.close('all')
#     fig = plt.figure(figsize=(10, 10))
#     gs = gridspec.GridSpec(1, 1)
#     ax = fig.add_subplot(gs[0, 0],frameon=True)
#     kwargs={'title':'min_Zeta', 'labels':['ζ', r'$ζ_f$']}
#     plot_timeseries(ax, times, *[min_zeta, zeta_fil], **kwargs)
    
#     for period,str in zip([intensification, mature, decay],
#                           ['int.', 'mature', 'decay']):
#         try:
#             ax.axvline(period[0], c='gray', linestyle='dashed')
#             ax.axvline(period[-1], c='gray', linestyle='dashed')
#             pos = round(len(period)/2)
#             ax.text(period[pos], max(min_zeta)*1.3, str, weight='bold',
#                     horizontalalignment='right')
#             plt.savefig('../periods/'+fname+'_periods.png',dpi=500)
#         except:
#             print('could not determine periods.')
#     return intensification, mature, decay

data_dir = '/home/daniloceano/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/met_data/ERA5/DATA/'
testtrack = '../tracks_LEC-format/intense/track_20180722'
varlist = '/home/daniloceano/Documents/Programs_and_scripts/lorenz-cycle/inputs/fvars_ERA5-cdsapi'
testfile = '/home/daniloceano/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/met_data/ERA5/DATA/20180722_ERA5.nc'

# testfile = '/Users/danilocoutodsouza/Documents/USP/Programs_and_scripts/data_etc/netCDF_data/Reg1-Representative_ERA5.nc'
# testtrack = '/Users/danilocoutodsouza/Documents/USP/Programs_and_scripts/lorenz-cycle/inputs/track_Reg1-Representative'
# varlist = '/Users/danilocoutodsouza/Documents/USP/Programs_and_scripts/lorenz-cycle/inputs/fvars_ERA5'

dfVars = pd.read_csv(varlist,sep= ';',index_col=0,header=0)
LonIndexer = dfVars.loc['Longitude']['Variable']
LatIndexer = dfVars.loc['Latitude']['Variable']
TimeIndexer = dfVars.loc['Time']['Variable']
LevelIndexer = dfVars.loc['Vertical Level']['Variable']
               

# for testfile in glob.glob(data_dir+'/*')[1:]:   
    
fname = testfile.split('/')[-1].split('.nc')[0]   
print(fname)
                  
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
    

min_zeta = pd.DataFrame(min_zeta, index=times)
plot_track(lons, lats, min_zeta, fname)
plot_vorticity(min_zeta, dzdt, fname)
incipient, intensification, mature, decay  = get_periods(min_zeta)

    
