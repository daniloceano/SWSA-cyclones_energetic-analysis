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

percentile = '0.999'
region = 'RG3'

# Directory where the files are saved
path = f'../periods-energetics/{percentile}/{region}/'

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


for i, df in enumerate(cyclist1):
    fases_presentes = list(df.index)
    decay_ocorreu = False
    for fase in fases_presentes:
        if fase == 'decay':
            decay_ocorreu = True
        elif fase == 'incipient' and decay_ocorreu:
            # remove a fase incipient que ta bugada
            df.drop(fase, inplace=True)

            cyclist1[i] = df
        elif fase == 'incipient 2' and decay_ocorreu:
            df.drop(fase, inplace=True)

            cyclist1[i] = df



fases_possiveis = ['incipient', 'intensification', 'mature', 'decay','intensification 2', 'mature 2', 'decay 2']
grupos = {}

for df in cyclist1:
    
    fases_presentes = list(df.index)

    chave_grupo = ''.join(['1' if fase in fases_presentes else '0' for fase in fases_possiveis])
    
    if chave_grupo not in grupos:
        grupos[chave_grupo] = [df]
    else:
        grupos[chave_grupo].append(df)




sys.exit()
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

PCA_DIR = os.path.join(path, 'PCA')

if not os.path.exists(PCA_DIR):
    os.makedirs(PCA_DIR)

# Saving the dataframes in csv files
df_PC1.to_csv(os.path.join(PCA_DIR, 'TESTE_PC1-LPS.csv'))
df_PC2.to_csv(os.path.join(PCA_DIR, 'TESTE_PC2-LPS.csv'))