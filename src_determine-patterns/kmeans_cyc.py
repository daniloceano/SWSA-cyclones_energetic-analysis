import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import NMF
import sys
import matplotlib.pyplot as plt
import numpy as np 
import math as mat
import scipy.io 
import xarray as xr     
import matplotlib
import sys, glob
from sklearn.linear_model import LinearRegression
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from eofs.xarray import Eof
from eofs.multivariate.standard import MultivariateEof


# Directory where the files are saved
path = '../periods-energetics/intense/'

# Getting all files in the directory
files_output = os.listdir(path)

# Selecting only the files with cyclone phases
files_use = [file for file in files_output if "_ERA5.csv" in file]

# Creating a list to save all dataframes
cyclist1= []

# Reading all files and saving in a list of dataframes
for case in files_use:
  columns_to_read = ['Ck','Ca', 'Ke', 'Ge']
  dfcyc = pd.read_csv(path+case,header=0,index_col=0)
  dfcyc = dfcyc[columns_to_read]
  cyclist1.append(dfcyc)

# List of cyclone phases
new_id = ['incipient','intensification','mature','decay','intensification 2', 'mature 2', 'decay 2']
id_sel = ['intensification','mature','decay']
cyclist2 = cyclist1.copy() 

# Reindexing all dataframes to the same cyclone phases
for i, df in enumerate(cyclist2):
    cyclist2[i] = df.reindex(new_id)
    cyclist2[i] = cyclist2[i].loc[id_sel]

combined_df = pd.concat(cyclist2, axis=1)

Ck1 = combined_df['Ck'].values.T
Ca1 = combined_df['Ca'].values.T
Ke1	= combined_df['Ke'].values.T
Ge1 = combined_df['Ge'].values.T

dsk_means = np.concatenate((Ck1,Ca1,Ke1,Ge1),axis=1)
mk = KMeans(n_clusters=4,n_init=10).fit(dsk_means)

slcenter = 3
centers_Ck = mk.cluster_centers_[:,0:slcenter]
centers_Ca = mk.cluster_centers_[:,slcenter:slcenter*2]
centers_Ke = mk.cluster_centers_[:,slcenter*2:slcenter*3]
centers_Ge = mk.cluster_centers_[:,slcenter*3:slcenter*4]

pc1aCk = centers_Ck[0,:]
pc2aCk = centers_Ck[1,:]
pc3aCk = centers_Ck[2,:]
pc4aCk = centers_Ck[3,:]

pc1aCa = centers_Ca[0,:]
pc2aCa = centers_Ca[1,:]
pc3aCa = centers_Ca[2,:]
pc4aCa = centers_Ca[3,:]

pc1aKe = centers_Ke[0,:]
pc2aKe = centers_Ke[1,:]
pc3aKe = centers_Ke[2,:]
pc4aKe = centers_Ke[3,:]

pc1aGe = centers_Ge[0,:]
pc2aGe = centers_Ge[1,:]
pc3aGe = centers_Ge[2,:]
pc4aGe = centers_Ge[3,:]


# Creating two dataframes to save the PCs
df_pc1 = cyclist2[0].copy()
df_pc1[:] = np.nan
df_pc2 = df_pc1.copy()
df_pc3 = df_pc1.copy()
df_pc4 = df_pc1.copy()


df_pc1['Ck'] = pc1aCk
df_pc1['Ca'] = pc1aCa
df_pc1['Ke'] = pc1aKe
df_pc1['Ge'] = pc1aGe

df_pc2['Ck'] = pc2aCk
df_pc2['Ca'] = pc2aCa
df_pc2['Ke'] = pc2aKe
df_pc2['Ge'] = pc2aGe

df_pc3['Ck'] = pc3aCk
df_pc3['Ca'] = pc3aCa
df_pc3['Ke'] = pc3aKe
df_pc3['Ge'] = pc3aGe

df_pc4['Ck'] = pc4aCk
df_pc4['Ca'] = pc4aCa
df_pc4['Ke'] = pc4aKe
df_pc4['Ge'] = pc4aGe




PCA_DIR = os.path.join(path, 'PCA')

if not os.path.exists(PCA_DIR):
    os.makedirs(PCA_DIR)

# Saving the dataframes in csv files
df_pc1.to_csv(os.path.join(PCA_DIR, 'Pc1_kmeans.csv'))
df_pc2.to_csv(os.path.join(PCA_DIR, 'Pc2_kmeans.csv'))
df_pc3.to_csv(os.path.join(PCA_DIR, 'Pc3_kmeans.csv'))
df_pc4.to_csv(os.path.join(PCA_DIR, 'Pc4_kmeans.csv'))




sys.exit()
PCA_DIR = os.path.join(path, 'PCA')

if not os.path.exists(PCA_DIR):
    os.makedirs(PCA_DIR)

# Saving the dataframes in csv files
df_PC1.to_csv(os.path.join(PCA_DIR, 'Pc1_3p_dn.csv'))
df_PC2.to_csv(os.path.join(PCA_DIR, 'Pc2_3p_dn.csv'))