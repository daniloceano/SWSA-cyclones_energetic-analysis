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

# def create_LPS_plots(fig_title, LPS_type, figsdir, zoom=False, **kwargs):
#         plt.close('all')
#         plt.figure(figsize=(10,10))
#         ax = plt.gca()
#         LorenzPhaseSpace(ax, LPS_type, zoom=zoom, **kwargs)
#         zoom_suffix = "_zoom" if zoom else ""
#         fname = f"{figsdir}/{fig_title}{zoom_suffix}.png"
#         with plt.rc_context({'savefig.dpi': 500}):
#                 plt.savefig(fname)
#         print(f"{fname} created!")

if __name__ == "__main__":
    
    intensities = ['0.99']

    for intensity in intensities:

        figsdir = f'../figures/LPS/{intensity}/RG3/'

        for PC in range(1,3):
    
                files = glob.glob(f'..//periods-energetics/{intensity}/RG3/PCA/*PC{PC}*.csv')

                LPS_type = 'mixed'

                plt.close('all')
                plt.figure(figsize=(10,10))
                fig_title = f"LPS_{LPS_type}_PC{PC}_RG3_{intensity}"

                zoom = True

                kwargs = {'terms': [], 'title':'PCs - RG3', 'datasource': 'ERA5',
                                'start': '1979', 'end': '2020'} 

                for file in files:
                        
                        df =  pd.read_csv(file, header=[0]) 
                        PC_id = os.path.basename(file.split('-')[-1].split('.csv')[0])
                        
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