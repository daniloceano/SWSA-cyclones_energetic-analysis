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
import numpy as np

from LPS import LorenzPhaseSpace

def create_LPS_plots(fig_title, zoom=False, **kwargs):
        plt.close('all')
        plt.figure(figsize=(10,10))
        ax = plt.gca()
        LorenzPhaseSpace(ax, zoom=zoom, **kwargs)
        zoom_suffix = "_zoom" if zoom else ""
        fname = f"../Figures/LPS/LPS_{fig_title}{zoom_suffix}.png"
        with plt.rc_context({'savefig.dpi': 500}):
                plt.savefig(fname)
        print(f"{fname} created!")

if __name__ == "__main__":
    
    files = glob.glob('..//periods-energetics/intense/PCA/*3pcs.csv')

    for file in files:
        
        df =  pd.read_csv(file, header=[0], index_col=[0]) 
        PC = file.split('/')[-1].split('.csv')[-0]
        intensity = file.split('/')[-3]

        kwargs = {'terms':[{'Ca': df['Ca'], 'Ck': df['Ck'],
                    'Ge': df['Ge'], 'Ke': df['Ke']}],
                      'title':PC,'datasource': 'ERA5'}
    
        create_LPS_plots(f"{intensity}_{PC}", zoom=False, **kwargs)
        create_LPS_plots(f"{intensity}_{PC}", zoom=True, **kwargs)