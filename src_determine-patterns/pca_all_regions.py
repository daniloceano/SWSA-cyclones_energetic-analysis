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
import glob 

percentile = '0.99'
region = 'all'

if region == 'all':
    regions = ['RG1', 'RG2', 'RG3']

path_save = os.path.join('..', 'periods-energetics', percentile)

all_files = []

# Iterar através de cada região e adicionar arquivos à lista
for region in regions:
    path = os.path.join('..', 'periods-energetics', percentile, region)
    # Usando glob para obter todos os arquivos .csv
    files = glob.glob(os.path.join(path, "*_ERA5.csv"))
    all_files.extend(files)

# Creating a list to save all dataframes
cyclist1= []

# Reading all files and saving in a list of dataframes
for case in all_files:
  columns_to_read = ['Ck','Ca', 'Ke', 'Ge']
  dfcyc = pd.read_csv(case,header=0,index_col=0)
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

# Getting all cyclone parameters (columns names)
parameters = cyclist2[0].keys()

# Creating a dictionary where the keys are the cyclone parameters (to save all Az, Ae... at the same time)
variaveis = {param: [] for param in parameters}

# Iterating in the list of dataframes, getting term by term of each case and saving in the dictionary
for i in range(len(cyclist2)):
    for param in parameters:

        # getting the value of the term 'termo' in the dataframe 'cyclist[i]'
        value = cyclist2[i][param]
        
        # saving the value in the dictionary
        variaveis[param].append(value)

# Creating two dataframes to save the PCs
df_PC1 = cyclist2[0].copy()
df_PC1[:] = np.nan
df_PC2 = df_PC1.copy()

## PCA
for tr in variaveis.keys():

  # Concatening all dataframes keeping the 7 phases (in this case it becomes a df with 7 lines and 240 columns [24 parameters * 10 cases])
  combined_df = pd.concat(variaveis[tr], axis=1)

  # Normalizing the data
  scaler = StandardScaler()
  normalized_data = scaler.fit_transform(combined_df)
  #normalized_data = combined_df
  # Imputer (NaN = 0)
  imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

  # The normalized data is updated by the imputer (NaN = 0)
  final_data = imputer.fit_transform(normalized_data)

  # PCA (each iteration the 'final_data' is a parameter)
  pca = PCA(n_components=2)
  pca_final = pca.fit_transform(final_data)
  # Invertendo a transformação do PCA
  pca_inv = pca.inverse_transform(pca_final)

  # Desnormalizando os dados
  denormalized_data = scaler.inverse_transform(pca_inv) 

  # Getting the first and second principal components
  pc1 = denormalized_data[:,0]
  pc2 = denormalized_data[:,1]

  # Saving the PCs in the dataframes
  df_PC1[tr] = pc1
  df_PC2[tr] = pc2

PCA_DIR = os.path.join(path_save, 'PCA')

if not os.path.exists(PCA_DIR):
    os.makedirs(PCA_DIR)

# Saving the dataframes in csv files
df_PC1.to_csv(os.path.join(PCA_DIR, 'AllRegions_PC1-LPS.csv'))
df_PC2.to_csv(os.path.join(PCA_DIR, 'AllRegions_PC2-LPS.csv'))

