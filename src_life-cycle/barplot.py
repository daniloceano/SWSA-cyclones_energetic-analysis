# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    barplot.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 16:44:54 by Danilo            #+#    #+#              #
#    Updated: 2023/10/26 10:42:12 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

"""
Reads the csv containing the number of systems and the percentage of each type of system
produced by export_species.py and create barplots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import mannwhitneyu

pd.options.mode.chained_assignment = None  # default='warn'

def process_df(df, percentage_threshold=1, filter_df=False, exclude_residual=False):

    processed_df = df.copy()

    # Filter rows based on percentage and if contains mature stage
    if filter_df:
        processed_df = df[df['Percentage'] > percentage_threshold]
        processed_df.loc[:, 'Num Phases'] = processed_df.loc[:, 'Type of System'].copy().apply(lambda x: len(x.split(', ')))
        processed_df = processed_df[processed_df['Type of System'].str.contains('mature')]

    if exclude_residual:
        # Exclude systems with the 'residual' stage
        processed_df.loc[:, 'Type of System'] = processed_df.loc[:, 'Type of System'].copy().str.replace(', residual', '')
        processed_df = processed_df.groupby('Type of System', as_index=False).agg({
                    'Total Count': 'sum',
                    'Percentage': 'sum'
                    }).sort_values(by='Total Count', ascending=False)
    
    return processed_df

def plot_combined_barplot(df_counts, title, fname):
    plt.close('all')

    sns.set_theme(style="ticks", palette="pastel")

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

    # Calculate total counts for each season
    total_counts = df_counts.groupby('Season')['Total Count'].sum()

    # Compute percentage for each Type of System based on the total count of its respective season
    df_counts['Percentage_season'] = df_counts.apply(lambda row: (row['Total Count'] / total_counts[row['Season']]) * 100, axis=1)

    # Replace labels with the desired scheme and join them back
    df_counts['Type of System'] = df_counts['Type of System'].apply(lambda x: ', '.join([label_mapping.get(word, word) for word in x.split(', ')]))

    # Separate DJF and JJA counts for each Type of System for statistical testing
    df_djf = df_counts[df_counts['Season'] == 'DJF']
    df_jja = df_counts[df_counts['Season'] == 'JJA']

    significant_diff = {}

    # Perform Mann-Whitney U test for each Type of System
    for system_type in df_counts['Type of System'].unique():
        djf_values = df_djf[df_djf['Type of System'] == system_type]['Total Count']
        jja_values = df_jja[df_jja['Type of System'] == system_type]['Total Count']

        # Ensure values are of type float or int and not NaN
        djf_values = int(djf_values.dropna().astype(float).values) if len(djf_values) > 0 else 0
        jja_values = int(jja_values.dropna().astype(float).values) if len(jja_values) > 0 else 0
        
        _, p_val = mannwhitneyu(djf_values, jja_values, alternative='two-sided')
        if p_val < 0.05:
            significant_diff[system_type] = "*"

    # Draw a nested barplot by species and season using Percentage on y-axis
    g = sns.catplot(
        data=df_counts, kind="bar",
        x="Type of System", y="Percentage_season", hue="Season",
        palette="pastel", height=6
        )
    
    # Annotate bars with * for significant differences
    for p in g.ax.patches:
        height = p.get_height()
        g.ax.text(p.get_x() + p.get_width()/2., height + 1, 
                significant_diff.get(p.get_x(), ""), 
                ha="center")
    
    g.legend.set_title("")
    plt.title(title, fontweight='bold')

    if df_counts['Type of System'].nunique() > 1:
        plt.xticks(rotation=45, ha='right')

    plt.savefig(fname, bbox_inches='tight')
    print(f'{fname} saved.')

    if df_counts['Type of System'].nunique() > 1:
        plt.xticks(rotation=45, ha='right')

    plt.savefig(fname, bbox_inches='tight')
    print(f'{fname} saved.')

def plot_barplot(df, title, fname):
    plt.close('all')

    # Mapping labels to the desired scheme
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

    total_count = df['Total Count'].sum()
    total_percentage = df['Percentage'].sum()

    # Convert 'Type of System' column to string to handle NaN values
    df['Type of System'] = df['Type of System'].astype(str)

    # Replace labels with the desired scheme and join them back
    df['Type of System'] = df['Type of System'].apply(lambda x: ', '.join([label_mapping.get(word, word) for word in x.split(', ')]))

    # Create a bar plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(y='Total Count', x='Type of System', data=df, orient='v', ci=None, palette='pastel', edgecolor='grey')
    plt.title(f'{title} ({total_count} - {total_percentage:.1f}%)', fontweight='bold')
    
    # Add text annotations for total count and percentage on the right side of each bar
    for index, value in enumerate(df['Total Count']):
        percentage = df.loc[df.index[index], 'Percentage']
        plt.text(index, value + 30, f"{value} ({percentage:.2f}%)", ha='center', color='black', fontweight='bold')

    # Set y-axis limit to start at 0
    plt.xlim(-0.5, len(df) - 0.5)

    # Hide axis titles
    plt.xlabel(None)
    plt.ylabel(None)

    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

    plt.tight_layout()

    # Save the plot as an image file
    plt.savefig(fname)
    print(f'{fname} saved.')

# analysis_type = 'BY_RG-all'
# analysis_type = 'all'
# analysis_type = '70W'
# analysis_type = '48h'
# analysis_type = '70W-48h'
analysis_type = '70W-no-continental'


# Set output directories
fig_output_directory = f'../figures/periods_statistics/{analysis_type}/barplots/'
csv_directory = f'../periods_species_statistics/{analysis_type}/count_systems_raw/'
csv_directory_processed = f'../periods_species_statistics/{analysis_type}/count_systems_processed/'
os.makedirs(fig_output_directory, exist_ok=True)
os.makedirs(csv_directory_processed, exist_ok=True)

# List of season names
seasons = ['JJA', 'DJF', 'total']

# List of RGs
if analysis_type == 'BY_RG-all': 
    RGs = ['RG1', 'RG2', 'RG3', 'all_RG']
elif analysis_type == '70W-no-continental':
    RGs = ["SE-BR", "LA-PLATA","ARG", "SE-SAO", "SA-NAM",
                "AT-PEN", "WEDDELL", False]
else:
    RGs = ['']

for RG in RGs:
    print(f'---------------------------\n RG: {RG}'
          ) if analysis_type in ['BY_RG-all', '70W-no-continental'] else print(
              f'---------------------------')

    # Suffix for creating files
    if analysis_type in ['BY_RG-all', '70W-no-continental']:
        suffix = f'_{RG}' if RG else '_SAt'
    else:
        suffix = '' 

    if analysis_type == 'BY_RG-all':
        seasonal_df = pd.DataFrame(columns=['Type of System', 'Season', 'RG', 'Total Count', 'Percentage'])
    else:
        seasonal_df = pd.DataFrame(columns=['Type of System', 'Season', 'Total Count', 'Percentage'])
    
    for season in seasons:
        print(f'\n Season: {season}')

        df = pd.read_csv(f'{csv_directory}/{season}_count_of_systems{suffix}.csv')
        df = df.sort_values(by='Total Count', ascending=False)

        # df excluding residual phases
        df_excluded_residual = process_df(df, filter_df=False, exclude_residual=True)

        # df for phases with more than 1% and with mature
        filtered_df = process_df(df, filter_df=True, exclude_residual=False)

        # df for phases with more than 1% and exluding residual phases
        filtered_df_exclude_residual = process_df(df, filter_df=True, exclude_residual=True)

        # export csv
        csv_name = f'{csv_directory_processed}/{season}_count_of_systems_processed{suffix}.csv'
        filtered_df_exclude_residual.to_csv(csv_name)
        print(f'{csv_name} saved.')

        print(filtered_df_exclude_residual)

        if season == 'total':
            fname = f'{fig_output_directory}/{season}_processed{suffix}.png'
            title = f'{RG} - {season}' if analysis_type == 'BY_RG-all' else f'{season}'
            plot_barplot(filtered_df_exclude_residual, title, fname)
        
        else:
            tmp = filtered_df_exclude_residual.copy()
            tmp['Season'] = season
            if analysis_type == 'BY_RG-all':
                tmp['RG'] = RG
            seasonal_df = pd.concat([seasonal_df, tmp], ignore_index=True)

    # Plot combined barplot for counts and percentages
    combined_fname = f'{fig_output_directory}/combined{suffix}.png'
    title = f'{RG}' if analysis_type == 'BY_RG-all' else 'all'
    plot_combined_barplot(seasonal_df, RG, combined_fname)