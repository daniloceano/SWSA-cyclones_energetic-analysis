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

percentile = '0.99'
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



fases_possiveis = ['incipient', 'intensification', 'mature', 'decay','intensification 2', 'mature 2', 'decay 2', 'residual']
grupos = {}

for df in cyclist1:
    
    fases_presentes = list(df.index)

    chave_grupo = ''.join(['1' if fase in fases_presentes else '0' for fase in fases_possiveis])
    
    if chave_grupo not in grupos:
        grupos[chave_grupo] = [df]
    else:
        grupos[chave_grupo].append(df)





    

# Getting all cyclone parameters (columns names)
parameters = cyclist1[0].keys()

grupos_var = {}

# Creating a dictionary where the keys are the cyclone grupos and these groups are keys that are all parameters (to save all Az, Ae... at the same time)

variaveis = {chave_grupo: {param: [] for param in parameters} for chave_grupo in grupos.keys()}

variaveis2 = variaveis.copy()

for gp_key in grupos.keys():
    #print(gp_key)
    for i in range(len(grupos[gp_key])):
        #print(i)
        for param in parameters:
            #print(param)
            value = grupos[gp_key][i][param]
            #print(value)
            variaveis[gp_key][param].append(value)
            #print(variaveis)
            print('-----------------')




for chave_gp in variaveis.keys():
    #print(chave_gp)
    df_PC1 = grupos[chave_gp][0].copy()
    df_PC1[:] = np.nan
    df_PC2 = df_PC1.copy()
    for tr in variaveis[chave_gp].keys():
        #print(tr)
        combined_df = pd.concat(variaveis[chave_gp][tr], axis=1)
        #print(len(combined_df))
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
        df_PC1.to_csv(os.path.join(PCA_DIR, f'DF_PC1-{chave_gp}.csv'))
        df_PC2.to_csv(os.path.join(PCA_DIR, f'DF_PC2-{chave_gp}.csv'))

    #add readme file
    readme = open(os.path.join(PCA_DIR, 'readme.txt'), 'w')
    readme.write('PCA analysis of cyclone phases')
    readme.write('\n')
    readme.write('\n')
    readme.write('the species of cyclones are: \n')
    readme.write(f'{variaveis.keys()} \n associated to the following phases: \n')
    readme.write(f'{fases_possiveis}')
    readme.write('\n')  
    readme.write('if the phase is present, the value is 1, otherwise 0')
    readme.write('\n')
    readme.close()
