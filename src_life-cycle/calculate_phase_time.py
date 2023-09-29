

import pandas as pd 
import glob 
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm 

# mode = 'BY_RG-all'
mode = 'all'

data_path = f'../periods-energetics/{mode}/'
figure_path = f'../figures/periods_statistics/{mode}/phase_time/'

if not os.path.exists(figure_path):
    os.makedirs(figure_path)


# List of RGs
RGs = ['1', '2', '3', 'all'] if mode == 'BY_RG-all' else ['all']

for RG in RGs:
    
    if mode == 'BY_RG-all':
        arquivos_csv = glob.glob(f'{data_path}/RG{RG}_*.csv')
        figure_path_RG = figure_path + f'RG{RG}/'
        if not os.path.exists(figure_path_RG):
            os.makedirs(figure_path_RG)
    else:
        arquivos_csv = glob.glob(f'{data_path}/*.csv')
        figure_path_RG = figure_path

    duration = {}

    # Processa cada arquivo
    for arquivo in arquivos_csv:
        df = pd.read_csv(arquivo, parse_dates=['start', 'end'], index_col=0)
        
        for fase in df.index:
            if fase not in duration:
                duration[fase] = []  # Inicializa a key com uma lista vazia
            
            inicio = df.loc[fase, 'start']
            fim = df.loc[fase, 'end']
            duração = (fim - inicio).total_seconds() / 3600  # Duração em horas
            duration[fase].append(duração)

    for fase, duracoes in duration.items():

        fig, ax = plt.subplots(figsize=(10,6))

        #plt.figure(figsize=(10,6))
        
        # Histograma
        ax.hist(duracoes, bins=20, density=False, alpha=0.6, color='b', label='Dados')
        
        # Calcular média e desvio padrão
        mu = np.mean(duracoes)
        std = np.std(duracoes)
        
        # Adicionar média e desvio padrão ao gráfico
        ax.axvline(mu, color='r', linestyle='dashed', linewidth=1, label=f'Média: {mu:.2f} horas')

        # adicionar informação de desvio padrao na legenda 
        textstr = '\n'.join((
            r'$media=%.2f$' % (mu, ),
            r'$std=%.2f$' % (std, )))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        ax.text(0.80, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
        

        ax.set_title(f'Distribuição da duração da fase: {fase}')
        ax.set_xlabel('Duração (horas)')
        ax.set_ylabel('Número de casos')

        fig.tight_layout()


        #plt.show() 
        plt.savefig(figure_path_RG + fase + '.png')


