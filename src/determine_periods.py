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
from sklearn.preprocessing import normalize   
from sklearn import preprocessing

import cartopy.crs as ccrs
import cartopy

def convert_lon(xr,LonIndexer):
    xr.coords[LonIndexer] = (xr.coords[LonIndexer] + 180) % 360 - 180
    xr = xr.sortby(xr[LonIndexer])
    return xr

def get_min_zeta(track, zeta850):
    
    min_zeta, times = [], []
    lats, lons = [], []
    for t in zeta850[TimeIndexer]:
        datestr = pd.to_datetime(t.values)
        
        if datestr in track.index:
            times.append(t.values)
            lat,lon = track.loc[datestr]
            lats.append(lat), lons.append(lon)
            min_zeta.append(float(zeta850.sel({TimeIndexer:t}
                    ).sel({LatIndexer:lat,LonIndexer:lon},method='nearest'))) 
            
    df = pd.DataFrame(min_zeta, index=times).rename(columns={0:'zeta'})
    df['lat'] = lats 
    df['lon'] = lons
    df.index.names = [TimeIndexer]
    
    return df      

def filter_var(variable):
    window_lenght = round(len(variable)/2)
    if (window_lenght % 2) == 0:
        window_lenght += 1
    return savgol_filter(variable, window_lenght, 3, mode="nearest")

def normalise_var(variable):
    return (variable - variable.min()) / (variable.max() - variable.min()) 

def array_vorticity(df):
    
    da = df.to_xarray()
    
    # Filter vorticity twice
    zeta_fil = xr.DataArray(filter_var(da.zeta), coords={'time':df.index})
    da = da.assign(variables={'zeta_fil':zeta_fil})
    zeta_fil2 = xr.DataArray(filter_var(zeta_fil),
                                 coords={'time':df.index})
    da = da.assign(variables={'zeta_fil2':zeta_fil2})
    
    # derivatives of the double-filtered vorticity
    da = da.assign(variables={'dzfil2_dt':
                da.zeta_fil2.differentiate(TimeIndexer,datetime_unit='h')})
        
    da = da.assign(variables={'dzfil2_dt2':
                da.dzfil2_dt.differentiate(TimeIndexer,datetime_unit='h')})
        
    da = da.assign(variables={'dzfil2_dt3':
                da.dzfil2_dt2.differentiate(TimeIndexer,datetime_unit='h')})
        
    # filter derivatives
    da = da.assign(variables={'dz_dt_fil2':
        xr.DataArray(filter_var(da.dzfil2_dt), coords={'time':df.index})})
    da = da.assign(variables={'dz_dt2_fil2':
        xr.DataArray(filter_var(da.dzfil2_dt2), coords={'time':df.index})})
    da = da.assign(variables={'dz_dt3_fil2':
        xr.DataArray(filter_var(da.dzfil2_dt3), coords={'time':df.index})})

    return da

def plot_track(da, fname):
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
    ax.plot(da.lon,da.lat,'-',c='k')
    scatter = ax.scatter(da.lon,da.lat,zorder=100,cmap=cmo.deep_r,c=da.zeta)
    plt.colorbar(scatter, pad=0.07, orientation='vertical',fraction=0.026,
                      label=' 850 hPa vorticity (ζ)')
    outname = '../vorticity_analysis/tracks/'+fname+'_track.png'
    plt.savefig(outname,dpi=500)
    print(outname,'saved')

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
    
def plot_vorticity(da, fname, norm=False, MegaFilter=False):
    font = { 'weight' : 'normal', 'size'   : 16}
    
    if norm  == True:
        flag = '_norm'
    else:
        flag = ''

    matplotlib.rc('font', **font)

    # min_zeta = min_zeta.resample('3H').mean()    
    times = da.time

    plt.close('all')
    fig = plt.figure(figsize=(5, 15))
    gs = gridspec.GridSpec(4, 1)
    
    # Zeta 850
    ax = fig.add_subplot(gs[0, 0],frameon=True)
    kwargs={'title':'min_Zeta', 'labels':['ζ', r'$ζ_f$']}
    plot_timeseries(ax, times,
                    *[da['zeta'+flag], da['zeta_fil'+flag]], **kwargs)
    
    # dZdt
    ax = fig.add_subplot(gs[1, 0],frameon=True)
    kwargs={'title':'dzdt','labels':[r'$\frac{∂ζ}{∂t}$',r'$\frac{∂ζ}{∂t}_f$']}
    plot_timeseries(ax, times,
                    *[da['dz_dt'+flag], da['dz_dt_fil'+flag]], **kwargs)
    
    # dZdt2
    ax = fig.add_subplot(gs[2, 0],frameon=True)
    kwargs={'title':'dzdt','labels':[r'$\frac{∂^{2}ζ}{∂t^{2}}$',
                                      r'$\frac{∂^{3}ζ}{∂t^{3}}_f$']}
    plot_timeseries(ax, times,
                    *[da['dz_dt2'+flag], da['dz_dt2_fil'+flag]], **kwargs)
    
    # dZdt3
    ax = fig.add_subplot(gs[3, 0],frameon=True)
    kwargs={'title':'dzdt','labels':[r'$\frac{∂^{3}ζ}{∂t^{3}}$',
                                      r'$\frac{∂^{3}ζ}{∂t^{3}}_f$']}
    plot_timeseries(ax, times,
                    *[da['dz_dt3'+flag], da['dz_dt3_fil'+flag]], **kwargs)

    plt.tight_layout()
    fig.align_labels()
    
    if MegaFilter == False:
        outname = '../vorticity_analysis/derivatives/'+fname+'_zeta'+flag+'.png'
    else:
        outname = '../vorticity_analysis/derivatives/'+fname+'_zeta_filter'+flag+'.png'
    plt.savefig(outname,dpi=500)
    print(outname, 'saved')
    
def get_periods(da, MegaFilter=False):
    
    zeta_fil = da.zeta_fil
    dt = pd.Timedelta((zeta_fil.time[1] - zeta_fil.time[0]).values)
    
    # Get the peiod of minimum vorticity. For splititng data in half.
    # The timesteps before it cannot be decaying stages. 
    minimum = zeta_fil.min()
    tmin_zeta = zeta_fil[zeta_fil==minimum].time.values
    zeta_fill_first_half = zeta_fil.where(zeta_fil.time < tmin_zeta
                                          ).dropna(dim=TimeIndexer)
    # The same goes for everything after the local minimum: it cannot be
    # intensification or inicipient periods
    zeta_fill_second_half = zeta_fil.where(zeta_fil.time > tmin_zeta
                                          ).dropna(dim=TimeIndexer)
    
    # Intensification is given by the period between local minima and maxima
    # of the vorticity third derivative, but for the first half of the
    # vorticity time series
    dzfil_dt3_fh = da.dz_dt3_fil2.sel(time=zeta_fill_first_half.time)
    intensification_start = dzfil_dt3_fh.idxmin().values
    intensification_end = dzfil_dt3_fh.idxmax().values
    if intensification_start > intensification_end:
        tmp = intensification_start
        intensification_start = intensification_end
        intensification_end = tmp
    intensification = pd.date_range(intensification_start,intensification_end,
                                    freq=dt)
    
    # Everything before the intensification start is incipient phase
    incipient = zeta_fill_first_half.time.where(
        zeta_fill_first_half.time < intensification_start).dropna(dim=TimeIndexer)
    incipient_start = zeta_fil.time[0].values
    if len(incipient) != 0:
        incipient_end = incipient[-1].values
    else:
        incipient_end = incipient_start
    incipient = zeta_fil.time.sel(time=slice(incipient_start,incipient_end))
    # Remove form incipient series, periods defined as intensification
    df_incip = pd.DataFrame(incipient.values)
    df_int = pd.DataFrame(intensification)
    for period in df_int:
        df_incip.drop(period, inplace=True)
    incipient = xr.DataArray(df_incip[0], coords={'time':df_incip[0]})
    
    
    # For decaying phase, it will be followed the same procedure as for the
    # intensification, but inverted: the local minima starts the intensfication
    # period, which will follow until the end of data
    dzfil_dt3_sh = da.dz_dt3_fil2.sel(time=zeta_fill_second_half.time)
    decay_start = dzfil_dt3_sh.idxmin().values
    decay_end = zeta_fil.time[-1].values
    decay = zeta_fil.time.sel(time=slice(decay_start,decay_end))
    
    # Mature stage is everything between end of intensification and start of
    # the decaying stage
    dt = zeta_fil.time[1] - zeta_fil.time[0]
    mature_start = zeta_fil.sel(time=intensification_end+dt).time.values
    mature_end = zeta_fil.sel(time=decay_start-dt).time.values
    mature = zeta_fil.time.sel(time=slice(mature_start,mature_end))
    
    periods = {'incipient': incipient,
               'intensification': intensification,
               'mature': mature,
               'decay': decay}
    
    return periods
    
def plot_periods(da, periods, fname, derivatives=False, filters=False):
    
    incipient = periods['incipient']
    intensification = periods['intensification']
    mature = periods['mature']
    decay = periods['decay']
    
    plt.close('all')
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111,frameon=True)
    colors = ['k', '#134074', '#d62828', '#f7b538', '#5b8e7d',]
    
    # Plot periods
    
    ax2 = ax.twinx()
    if len(incipient) != 0:
        ax2.fill_betweenx((0,1), incipient[0], intensification[0], 
                     facecolor=colors[1], alpha=0.2, label='incipient')
                
    ax2.fill_betweenx((0,1), intensification[0], mature[0],
                     facecolor=colors[2], alpha=0.2, label='intensification')

    ax2.fill_betweenx((0,1), mature[0], decay[0],
                     facecolor=colors[3], alpha=0.2, label='mature')
    
    ax2.fill_betweenx((0,1), decay[0], decay[-1],
                     facecolor=colors[4], alpha=0.2, label='decay')
    
    if derivatives==True:
        ax.plot(da.time, da.dz_dt3_fil2*10000, c='#fca311', linewidth=2,
                 label=r'$\frac{∂^{3}ζ}{∂t^{3}}$', alpha=0.8)  
        ax.plot(da.time, da.dz_dt2_fil2*1000, c='#e76f51', linewidth=2,
                 label=r'$\frac{∂^{2}ζ}{∂t^{3}}$', alpha=0.8)
        ax.plot(da.time, da.dz_dt_fil2*100, c='#219ebc', linewidth=2,
                 label=r'$\frac{∂ζ}{∂t}$', alpha=0.8)
    if filters==True:
        ax.plot(da.zeta_fil2.time, da.zeta_fil2,c=colors[0],
        linewidth=4,label=r'$ζ$')
    else:
        ax.plot(da.zeta.time, da.zeta,c=colors[0],
        linewidth=4,label=r'$ζ$')
    # y = np.arange(da.zeta_fil2.min(),da.zeta_fil2.max()+1e-5,1e-5)
    # ax.set_ylim(y[0],y[-1])
    
    plt.xlim(da.zeta_fil.time[0].values, da.zeta_fil.time[-1].values)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=4)
    plt.grid(linewidth=0.5, alpha=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    outname = '../vorticity_analysis/periods/'+fname
    if derivatives==True:
        outname+='-filter'
    if derivatives==True:
        outname+='-derivaitves'
    outname+='.png'
    plt.savefig(outname,dpi=500)
    print(outname,'saved')
    

data_dir = '../met_data/ERA5/DATA/'
track_dir = '../tracks_LEC-format/intense/'
varlist = '../fvars/fvars_ERA5-cdsapi'

dfVars = pd.read_csv(varlist,sep= ';',index_col=0,header=0)
LonIndexer = dfVars.loc['Longitude']['Variable']
LatIndexer = dfVars.loc['Latitude']['Variable']
TimeIndexer = dfVars.loc['Time']['Variable']
LevelIndexer = dfVars.loc['Vertical Level']['Variable']
               

for testfile in glob.glob(data_dir+'/*'):   
        
    fname = testfile.split('/')[-1].split('.nc')[0]   
    id_cyclone = fname.split('_')[0]
    track_file = track_dir+'track_'+id_cyclone
    print('Cyclone ID:',id_cyclone)
    print('Track file:',track_file)  
                    
    da = convert_lon(xr.open_dataset(testfile),LonIndexer)
    da850 = da.sel({LevelIndexer:850})
    u850 = da850[dfVars.loc['Eastward Wind Component']['Variable']] * units('m/s')
    v850 = da850[dfVars.loc['Northward Wind Component']['Variable']] * units('m/s')
    zeta850 = vorticity(u850,v850)
    
    track = pd.read_csv(track_file, parse_dates=[0],delimiter=';',index_col=TimeIndexer)

    df = get_min_zeta(track, zeta850) 
    
    da = array_vorticity(df)
    
    # plot_track(da, fname)
    # plot_vorticity(da, fname)    
    periods  = get_periods(da)
    plot_periods(da, periods, fname, derivatives=False, filters=False)
    plot_periods(da, periods, fname, derivatives=True, filters=True)
