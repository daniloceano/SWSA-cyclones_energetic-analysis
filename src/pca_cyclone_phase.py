import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


path = '../periods-energetics/intense/'

files_output = os.listdir(path)


files_use = [file for file in files_output if "_" in file]


cyclist= []

# aqui le cada arquivo e salva o dataframe numa lista de dataframes
for caso in files_use:
  
  dfcyc = pd.read_csv(path+caso,header=0,index_col=0)
  cyclist.append(dfcyc)

# criei pra auxiliar
id_novo = ['incipient','intensification','mature','decay','intensification 2', 'mature 2', 'decay 2']

# reindexando, pra todo mundo ter todas fases
for i, df in enumerate(cyclist):
    cyclist[i] = df.reindex(id_novo)


# pegando os termos (as colunas)
termos = cyclist[0].keys()

# criando um dicionario onde as chaves sao os termos (pra poder salvar todos Az, Ae... ao mesmo tempo)
variaveis = {termo: [] for termo in termos}

# iterando na lista de dataframe, pegando termo a termo de cada caso e salvando no dicionario
for i in range(len(cyclist)):
    for termo in termos:

        # valor de cada termo em cada df 
        value = cyclist[i][termo]
        
        variaveis[termo].append(value)



# criando dataframes auxiliares 
df_novo1 = cyclist[0].copy()
df_novo1[:] = np.nan
df_novo2 = df_novo1.copy()

for tr in variaveis.keys():

  # colando todos dataframes mantendo as 7 fases (no caso vira um df com 7 linhas e 240 colunas [24termos*10casos])
  combined_df = pd.concat(variaveis[tr], axis=1)

  # p normalizar
  scaler = StandardScaler()
  normalized_data = scaler.fit_transform(combined_df)

  # nunca tinha visto assim, mas chatgpt falou que é o imputer do sklearn entao bora
  imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)

  # aq o dado normalizado é atualizado pelo imputer (NaN = 0)
  dado_final = imputer.fit_transform(normalized_data)

  print(dado_final.shape) # 7 fases, 10 casos -> Para cada termo
  
  # PCA (a cada iteracao o 'dado_final' é um termo)
  pca = PCA(n_components=2)
  pca_final = pca.fit_transform(dado_final)

  # separando as pcs 
  pc1 = pca_final[:,0]
  pc2 = pca_final[:,1]


  # atualizando os termos dos dataframes auxiliares
  df_novo1[tr] = pc1
  df_novo2[tr] = pc2




df_novo1.to_csv(path+'Pca1.csv')
df_novo2.to_csv(path+'Pca2.csv')