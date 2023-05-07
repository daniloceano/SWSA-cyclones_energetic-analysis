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
import pathlib

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
    cyclist2[i] = cyclist2[i].stack()
    cyclist2[i] = pd.DataFrame(cyclist2[i], columns=['Valor'])
    cyclist2[i].index.names = ['Fase', 'Parametro']
    cyclist2[i] = cyclist2[i].sort_values(by=['Parametro'])
    
combined_df = pd.concat(cyclist2, axis=1)

df_PCm1 = combined_df.copy() # Copiando o DataFrame para não alterar o original
df_PCm1 = df_PCm1.iloc[:, :1] # Selecionando apenas a primeira coluna (Ck) para fazer DataFrame na mesma estrutura 
df_PCm2 = df_PCm1.copy() 


scaler = StandardScaler() # Instanciando o objeto StandardScaler
normalized_data = scaler.fit_transform(combined_df) # Normalizando os dados
pca = PCA(n_components=2) # Instanciando o objeto PCA com 2 componentes principais 
pca_final = pca.fit_transform(normalized_data) # Aplicando o PCA nos dados normalizados
# Invertendo a transformação do PCA
pca_inv = pca.inverse_transform(pca_final) #  Desnormalizando os dados 

# Desnormalizando os dados
denormalized_data = scaler.inverse_transform(pca_inv)  # Desnormalizando os dados 

# Getting the first and second principal components
pc1 = denormalized_data[:,0]
pc2 = denormalized_data[:,1]

# Saving the PCs in the dataframes
df_PCm1['Valor'] = pc1
df_PCm2['Valor'] = pc2

df_PC1 = df_PCm1['Valor'].unstack()
df_PC2 = df_PCm2['Valor'].unstack()

df_PC1.index.name = None
df_PC1.columns.name = None
df_PC2.index.name = None
df_PC2.columns.name = None

# Define o caminho do diretório
PCA_DIR = pathlib.Path(path, 'PCA')

# Verifica se o diretório existe, caso não exista cria o diretório
if not PCA_DIR.is_dir():
    PCA_DIR.mkdir(parents=True, exist_ok=True)

# Salvando os dataframes em arquivos CSV
df_PC1.to_csv(PCA_DIR / 'Pc1_m3p.csv') # m de multivariated and 3p de 3 phases 
df_PC2.to_csv(PCA_DIR / 'Pc2_m3p.csv') # m de multivariated and 3p de 3 phases 