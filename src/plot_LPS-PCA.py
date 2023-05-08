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


if __name__ == "__main__":
    
    files = glob.glob('..//periods-energetics/intense/PCA/*dn.csv')

    for file in files:
        
        df =  pd.read_csv(file, header=[0], index_col=[0]) 
        PC = file.split('/')[-1].split('.csv')[-0]
        intensity = file.split('/')[-3]

        kwargs = {'terms':[{'Ca': df['Ca'], 'Ck': df['Ck'],
                    'Ge': df['Ge'], 'Ke': df['Ke']}],
                      'title':PC,'datasource': 'ERA5'}
    
        plt.close('all')
        plt.figure(figsize=(10,10))
        ax = plt.gca()
        LorenzPhaseSpace(ax, **kwargs)
        fname = '../figures/LPS/LPS-PCA_'+PC+'_'+intensity+'.png'
        plt.savefig(fname,dpi=500)
        print(fname+' created!')
        
        plt.close('all')
        plt.figure(figsize=(10,10))
        ax = plt.gca()
        LorenzPhaseSpace(ax, zoom=True, **kwargs)
        fname = '../figures/LPS/LPS-PCA_'+PC+'_zoom_'+intensity+'.png'
        plt.savefig(fname,dpi=500)
        print(fname+' created!')