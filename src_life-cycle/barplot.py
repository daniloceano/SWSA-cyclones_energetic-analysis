
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import mannwhitneyu

def process_df(df, percentage_threshold=1, filter_df=False, exclude_residual=False, contain_mature_phase=False):

    processed_df = df.copy()

    # Filter rows based on percentage and number of phases
    if filter_df:
        processed_df = df[df['Percentage'] > percentage_threshold].copy()  # Make a copy here
        processed_df['Num Phases'] = processed_df['Type of System'].apply(lambda x: len(x.split(', ')))

    if exclude_residual:
        # Exclude systems with the 'residual' stage
        processed_df['Type of System'] = processed_df['Type of System'].str.replace(', residual', '')  
        processed_df = processed_df.groupby('Type of System', as_index=False).agg({
                    'Total Count': 'sum',
                    'Percentage': 'sum'
                    }).sort_values(by='Total Count', ascending=False)
    
    if contain_mature_phase:
        processed_df = processed_df[processed_df['Type of System'].str.contains('mature')]
    
    return processed_df

def get_data(csv_directory):
    dfs = [] 

    for RG in RGs:
        suffix = f'_{RG}' if RG else '_SAt'
        seasons = ['DJF', 'MAM', 'JJA', 'SON', 'total']

        for season in seasons:
            df_season = pd.read_csv(f'{csv_directory}/{season}_count_of_systems{suffix}.csv')
            df_season = df_season.sort_values(by='Total Count', ascending=False)

            # df for phases with more than 1% and with mature
            filtered_df_season = process_df(df_season, filter_df=True, exclude_residual=True, contain_mature_phase=True)

            # export csv
            csv_name = f'{csv_directory_processed}/{season}_count_of_systems_processed{suffix}.csv'
            filtered_df_season.to_csv(csv_name)
            print(f'{csv_name} saved.')

            filtered_df_season['RG'] = RG
            filtered_df_season['Season'] = season
            dfs.append(filtered_df_season)

    df = pd.concat(dfs)
    return df

def plot_comparison(df, fname):
    palette = {'DJF': 'salmon',  # pastel red
           'JJA': 'lightblue'  # pastel blue
          }
    
    label_mapping = {
        'incipient': 'Ic',
        'incipient 2': 'Ic2',
        'intensification': 'It',
        'intensification 2': 'It2',
        'mature': 'M',
        'mature 2': 'M2',
        'decay': 'D',
        'decay 2': 'D2',
    }

    ordered_RGs = ["Total", "ARG", "LA-PLATA", "SE-BR", "SE-SAO", "AT-PEN", "WEDDELL", "SA-NAM"]

    # Filter out only DJF and JJA seasons
    df_filtered = df[df['Season'].isin(['DJF', 'JJA'])]
    df_filtered['RG'] = df_filtered['RG'].replace(False, 'Total')

    # Replace labels with the desired scheme and join them back
    df_filtered['Type of System'] = df_filtered['Type of System'].apply(lambda x: ', '.join([label_mapping.get(word, word) for word in x.split(', ')]))

    # Create a mapping from "Type of System" to a unique number
    unique_systems = df_filtered['Type of System'].unique()
    system_map = {system: i+1 for i, system in enumerate(unique_systems)}
    
    # Add a new column with this mapping
    df_filtered['System Number'] = df_filtered['Type of System'].map(system_map)
    
    # Create a figure with multiple subplots (one for each region)
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 8))

    # Create an empty list to store the legend handles and labels
    handles, labels = [], []
    
    # Iterate over each region and plot
    for i, RG in enumerate(ordered_RGs):
        ax_row = i // 3
        ax_col = i % 3
        ax = axes[ax_row][ax_col]
        df_rg = df_filtered[df_filtered['RG'] == RG]
        
        # Use seaborn to plot the barplot using 'System Number' for the x-axis
        sns.barplot(data=df_rg, x='System Number', y='Percentage', hue='Season', ax=ax, palette=palette)
        
        ax.set_title(f"{RG}")
        ax.set_ylabel("Percentage")
        ax.set_xlabel("Type of System")
        
        # Get the legend data from the first plot
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            # Place the legend on the top right corner of the first subplot
            ax.legend(handles, labels, loc='upper right', title="Season")
        
        # Remove the individual legend for other subplots
        else:
            ax.legend().set_visible(False)
    
    # Clear last subplot and hide its axes
    axes[-1, -1].clear()
    axes[-1, -1].axis('off')
    
    # Add a legend for the "Type of System" mapping in the last subplot
    system_labels = [f"{num}. {system}" for system, num in system_map.items()]
    axes[-1, -1].legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                        markersize=10) for _ in system_labels],
                        labels=system_labels, title="System Mapping", loc='center')
    
    plt.tight_layout()
    plt.savefig(fname)

def plot_total_season(df, fname):
    label_mapping = {
        'incipient': 'Ic',
        'incipient 2': 'Ic2',
        'intensification': 'It',
        'intensification 2': 'It2',
        'mature': 'M',
        'mature 2': 'M2',
        'decay': 'D',
        'decay 2': 'D2',
    }

    # Filter the data for the "Total" season
    df_total = df[df['Season'] == 'total']
    df_total['RG'] = df_total['RG'].replace(False, 'Total')

    # Replace labels with the desired scheme and join them back
    df_total['Type of System'] = df_total['Type of System'].apply(lambda x: ', '.join([label_mapping.get(word, word) for word in x.split(', ')]))

    # Set custom order for 'RG' column with "Total" at the start
    order = ['Total'] + [rg for rg in df_total['RG'].unique() if rg != 'Total']
    df_total['RG'] = pd.Categorical(df_total['RG'], categories=order, ordered=True)
    
    # Filter out 'Type of System' categories that do not appear for all 'RG' values
    valid_systems = df_total.groupby('Type of System')['RG'].nunique() == len(df_total['RG'].cat.categories)
    valid_systems = valid_systems[valid_systems].index
    df_total = df_total[df_total['Type of System'].isin(valid_systems)]

    # Compute aggregated percentage for each 'Type of System' and create a custom order
    system_order = df_total.groupby('Type of System')['Percentage'].mean().sort_values(ascending=False).index

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_total, x='Type of System', y='Percentage', hue='RG', palette='pastel', edgecolor='grey', order=system_order)
    plt.title("Type of System Analysis for 'Total' Season")
    plt.ylabel("Percentage")
    plt.xlabel("Type of System")
    plt.xticks(rotation=45)  # Rotate x labels for better visibility if needed
    plt.legend(title="Region", loc="upper right")
    plt.tight_layout()
    plt.savefig(fname)

analysis_type = '70W-no-continental'
RGs = [False, "ARG", "LA-PLATA", "SE-BR", "SE-SAO", "AT-PEN", "WEDDELL", "SA-NAM"]

# Set output directories
fig_output_directory = f'../figures/periods_statistics/{analysis_type}/barplots/'
csv_directory = f'../periods_species_statistics/{analysis_type}/count_systems_raw/'
csv_directory_processed = f'../periods_species_statistics/{analysis_type}/count_systems_processed/'
os.makedirs(fig_output_directory, exist_ok=True)
os.makedirs(csv_directory_processed, exist_ok=True)

df = get_data(csv_directory)

fname = f'{fig_output_directory}/all_regions_djf_jja.png'
plot_comparison(df, fname)

fname_total = f'{fig_output_directory}/total_season_by_region.png'
plot_total_season(df, fname_total)