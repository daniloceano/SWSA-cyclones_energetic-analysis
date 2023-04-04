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
from scipy.ndimage.measurements import label, find_objects

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
    
    
def plot_didatic(da, fname):
    
    colors = ['k', '#134074', '#d62828', '#f7b538', '#5b8e7d',]
    
    z = da.zeta_fil2
    dz = da.dz_dt_fil2*50
    dz2 = da.dz_dt2_fil2*500
    dz3 = da.dz_dt3_fil2*5000
    
    plt.close('all')
    fig = plt.figure(figsize=(15, 12))
    
# =============================================================================
#   First figure: all series
# =============================================================================
    ax = fig.add_subplot(221,frameon=True)
    ax.plot(da.time,dz3,c=colors[3], linewidth=0.75,
              label=r'$\frac{∂^{3}ζ}{∂t^{3}}$')
    ax.plot(da.time,dz2,c=colors[2], linewidth=0.75,
              label=r'$\frac{∂^{2}ζ}{∂t^{2}}$')
    ax.plot(da.time,dz,c=colors[1], linewidth=0.75,
             label=r'$\frac{∂ζ}{∂t}$')
    ax.plot(da.time,z,c=colors[0], linewidth=2, label='ζ')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),ncol=4)
    
# =============================================================================
#   Get peaks and vallys of the derivatives
# =============================================================================
    ax2 = fig.add_subplot(222,frameon=True)

    c = 1
    for derivative in [dz, dz2, dz3]: 
        maxs_dz = derivative.where(derivative > derivative.quantile(0.75))
        mins_dz = derivative.where(derivative < derivative.quantile(0.25))
        # get indices of continuous data
        labels_peak, num_features_peak = label(~np.isnan(maxs_dz))
        labels_valley, num_features_valley = label(~np.isnan(mins_dz))
        slices_peak = find_objects(labels_peak)
        slices_valley = find_objects(labels_valley)
        # Get continuous series for derivatives values higher than the 0.75
        # quantile and lower than the 0.15 quantile
        continuous_max_dz = [maxs_dz[sl] for sl in slices_peak if sl is not None]
        continuous_min_dz = [mins_dz[sl] for sl in slices_valley if sl is not None]
        # Now, get maximum and mininum values (peaks and valleys)
        peaks, valleys = [], []
        times_peaks, times_valleys = [], []
        for data_peak in continuous_max_dz:
            peaks.append(data_peak.max())
            times_peaks.append(data_peak.idxmax())
        for data_valley in continuous_min_dz:
            valleys.append(data_valley.min())
            times_valleys.append(data_valley.idxmin())
        for i in range(len(peaks)):
            ax2.scatter(times_peaks[i],peaks[i],facecolor='#ab791b',
                        edgecolor='k', linewidth=2)
        for i in range(len(valleys)):
            ax2.scatter(times_valleys[i],valleys[i],facecolor='#ab791b',
                        linewidth=2)
    
        negative_dz = derivative.where(derivative < 0)
        positive_dz = derivative.where(derivative > 0)
        ax2.plot(negative_dz.time,negative_dz,c=colors[c], linewidth=1,
                  linestyle='dashed')
        ax2.plot(positive_dz.time,positive_dz,c=colors[c], linewidth=1,
                  linestyle='-')
        c+=1
    
    ax2.plot(da.time,z,c=colors[0], linewidth=2)
    ax2.legend(loc='center right', bbox_to_anchor=(1.35, 0.5),ncol=1)
    
# =============================================================================
#   Separete data between peaks and valleys of the third derivative
# =============================================================================
    ax3 = fig.add_subplot(223,frameon=True)
    
    maxs_dz3 = dz3.where(dz3 > dz3.quantile(0.75))
    mins_dz3 = dz3.where(dz3 < dz3.quantile(0.25))
    # get indices of continuous data
    labels_peak, num_features_peak = label(~np.isnan(maxs_dz3))
    labels_valley, num_features_valley = label(~np.isnan(mins_dz3))
    slices_peak = find_objects(labels_peak)
    slices_valley = find_objects(labels_valley)
    # Get continuous series for dz_dt3 values higher than the 0.85 quantile
    # and lower than the 0.15 quantile
    continuous_max_dz = [maxs_dz3[sl] for sl in slices_peak if sl is not None]
    continuous_min_dz = [mins_dz3[sl] for sl in slices_valley if sl is not None]
    # Now, get maximum and mininum values (peaks and valleys)
    peaks, valleys = [], []
    times_peaks, times_valleys = [], []
    for data_peak in continuous_max_dz:
        peaks.append(data_peak.max())
        times_peaks.append(data_peak.idxmax())
    for data_valley in continuous_min_dz:
        valleys.append(data_valley.min())
        times_valleys.append(data_valley.idxmin())
    
    # Remove peaks and valleys that occured very close to each other
    times_peaks_fil, times_valleys_fil = [], []
    for i in range(len(times_peaks)):
        tpeak = times_peaks[i]
        if i == 0:
            times_peaks_fil.append(tpeak)
        else:
            if times_peaks[i]-times_peaks[i-1] < pd.Timedelta('1 day'):
                pass
            else:
                times_peaks_fil.append(tpeak)
    
    for i in range(len(times_valleys)):
        tvalley = times_valleys[i]
        if i == 0:
            times_valleys_fil.append(tvalley)
        else:
            if times_valleys[i]-times_valleys[i-1] < pd.Timedelta('1 day'):
                pass
            else:
                times_valleys_fil.append(tvalley)
    
    ax3.plot(da.time,z,c=colors[0], linewidth=2, label='ζ')
    for tpeak in times_peaks_fil:
        ax3.scatter(tpeak, z.where(z.time==tpeak).dropna(dim='time'),
                    c=colors[3], edgecolor='k')
    for tvalley in times_valleys_fil:
        ax3.scatter(tvalley, z.where(z.time==tvalley).dropna(dim='time'),
                    c=colors[3],)
    
    c = 1
    for derivative in [dz, dz2, dz3]:
        negative_dz = derivative.where(derivative < 0)
        positive_dz = derivative.where(derivative > 0)
        
        negative_z = z.where(z.time == negative_dz.dropna(dim='time').time)
        positive_z = z.where(z.time == positive_dz.dropna(dim='time').time)
        
        ax3.scatter(negative_z[::4].time,negative_z[::4] + (np.abs(z.min()/8)*c),
                    c='None', edgecolor=colors[c])
        ax3.scatter(positive_z[::4].time,positive_z[::4] + (np.abs(z.min()/8)*c),
                    c=colors[c])
        c+=1
    
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gcf().autofmt_xdate()
    plt.subplots_adjust(right=0.85)
    
    outname = '../vorticity_analysis/analysis/'+fname
    outname+='.png'
    plt.savefig(outname,dpi=500)
    print(outname,'saved')

def mature_phase(da):
    
    z = da.zeta_fil2
    dz = da.dz_dt_fil2
    dz2 = da.dz_dt2_fil2
    dz3 = da.dz_dt3_fil2
    
    # timestep
    dt = pd.Timedelta((z.time[1] - z.time[0]).values)
    
    # mature phase will be defined where the filtered vorticity is bellow the
    # 0.25 standard deviations threshold
    throughs = z.where(z < z.mean()-z.std()/4)
    
    ## Determine the first mature phase (maybe the only one)
    # Get the peiod of minimum vorticity. For splititng data in half.
    # The timesteps before it cannot be decaying stages. 
    minimum = z.min()
    tmin_zeta = z[z==minimum].time.values
    z_first_half = z.where(z.time < tmin_zeta).dropna(dim=TimeIndexer)
    # Intensification is given by the period between local minima and maxima
    # of the vorticity third derivative, but for the first half of the
    # vorticity time series
    dz3_fh = dz3.sel(time=z_first_half.time)
    intensification_start = dz3_fh.idxmin().values
    intensification_end = dz3_fh.idxmax().values
    if intensification_start > intensification_end:
        tmp = intensification_start
        intensification_start = intensification_end
        intensification_end = tmp
    intensification = pd.date_range(intensification_start,intensification_end,
                                    freq=dt)
    

    
def get_periods(da):
    
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
    # intensification, but inverted: the local minima starts the decaying
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
    
    outname = '../vorticity_analysis/periods/'
    if filters==True:
        outname+='filter/'
    else:
        outname+='non-filterfilter/'
    outname+=fname
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
    # periods  = get_periods(da)
    
    # plot_periods(da, periods, fname, derivatives=False, filters=False)
    # plot_periods(da, periods, fname, derivatives=True, filters=True)
    
    plot_didatic(da, fname)
