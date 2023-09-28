#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 00:46:26 2023

@author: danilocoutodsouza
"""
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import cmocean
import glob
import os
from LPS import LorenzPhaseSpace, plot_legend

def plot_LPS_one_fig(intensity, figsdir):

        fig_title = f"LPS_{LPS_type}_all-RG_all-PC_{intensity}"

        plt.close('all')
        plt.figure(figsize=(10,10))

        files = glob.glob(f'..//periods-energetics/quantile/{intensity}/PCA_{mode}/*.csv')

        kwargs = {'terms': [], 'title':'PCs - RG3', 'datasource': 'ERA5',
                        'start': '1979', 'end': '2020'} 

        i = 0
        for file in files:
                df =  pd.read_csv(file, header=[0]) 

                if mode == 'all-species':
                        PC_id = os.path.basename(file.split('-')[-1].split('.csv')[0])
                else:
                        PC_id = os.path.basename(file).split('-')[0].split('_')[-1]
                
                terms = {'y_axis': df['Ca'], 'x_axis': df['Ck'],
                        'circles_colors': df['Ge'], 'circles_size': df['Ke']}
                kwargs['terms'].append(terms)
                
                ax = plt.gca()
                ax.text(df['Ck'][0], df['Ca'][0]+0.15, PC_id, fontsize=10, fontweight='bold')

                if i == 0:
                        LorenzPhaseSpace(ax, LPS_type, zoom=zoom, **kwargs)
                else:
                        LorenzPhaseSpace(ax, LPS_type, zoom=zoom, legend=False,
                                                cbar=False, **kwargs)
                i += 1

        zoom_suffix = "_zoom" if zoom else ""
        fname = f"{figsdir}/{fig_title}{zoom_suffix}.png"
        with plt.rc_context({'savefig.dpi': 500}):
                plt.savefig(fname)
        print(f"{fname} created!")

def plot_LPS_multiple_figs(intensity, figsdir):

        for PC in range(1,3):
                fig_title = f"LPS_{LPS_type}_PC{PC}_RG3_{intensity}"

                files = glob.glob(f'..//periods-energetics/quantile/{intensity}/PCA_{mode}/*PC{PC}*.csv')

                LPS_type = 'mixed'

                plt.close('all')
                plt.figure(figsize=(10,10))

                zoom = True

                kwargs = {'terms': [], 'title':'PCs - RG3', 'datasource': 'ERA5',
                                'start': '1979', 'end': '2020'} 

                for file in files:
                        
                        df =  pd.read_csv(file, header=[0]) 

                        if mode == 'all-species':
                                PC_id = os.path.basename(file.split('-')[-1].split('.csv')[0])
                        else:
                                PC_id = os.path.basename(file).split('-')[0].split('_')[-1]
                        
                        terms = {'y_axis': df['Ca'], 'x_axis': df['Ck'],
                                'circles_colors': df['Ge'], 'circles_size': df['Ke']}
                        kwargs['terms'].append(terms)
                        
                        ax = plt.gca()

                        ax.text(df['Ck'][0], df['Ca'][0], PC_id, fontsize=10, fontweight='bold')

                        LorenzPhaseSpace(ax, LPS_type, zoom=zoom, **kwargs)

                zoom_suffix = "_zoom" if zoom else ""
                fname = f"{figsdir}/{fig_title}{zoom_suffix}.png"
                with plt.rc_context({'savefig.dpi': 500}):
                        plt.savefig(fname)
                print(f"{fname} created!")


# mode = 'all-species'
mode = 'IntMatureDecay'
intensity = '0.99'

LPS_type = 'mixed'
zoom = True

figsdir = f'../figures/LPS/{intensity}/RG3/' if mode == 'all-species' else f'../figures/LPS/{intensity}/'

if mode == 'all-species':
        plot_LPS_multiple_figs(intensity, figsdir)
        
else:
        plot_LPS_one_fig(intensity, figsdir)