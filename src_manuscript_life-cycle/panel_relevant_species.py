import pandas as pd
from glob import glob

periods_dir = '../periods-energetics/BY_RG-all/RG1_19800056.csv'

count_species = pd.read_csv('total_count_of_systems_filtered.csv', index_col=0)

df_periods = pd.read_csv(periods_dir, index_col=0)

periods = df_periods.index.tolist()

