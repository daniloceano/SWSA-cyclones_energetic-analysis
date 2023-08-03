# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    boxplot.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 16:44:54 by Danilo            #+#    #+#              #
#    Updated: 2023/08/03 20:20:53 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_filtered_df(df, percentage_threshold=1, num_phases_threshold=2):
    # Filter rows based on percentage and number of phases
    filtered_df = df[df['Percentage'] > percentage_threshold]
    filtered_df['Num Phases'] = filtered_df['Type of System'].apply(lambda x: len(x.split(', ')))
    filtered_df = filtered_df[filtered_df['Num Phases'] > num_phases_threshold]
    return filtered_df

def plot_barplot(df, season, output_directory, filter=False):
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

    # Convert 'Type of System' column to string to handle NaN values
    df['Type of System'] = df['Type of System'].astype(str)

    # Replace labels with the desired scheme and join them back
    df['Type of System'] = df['Type of System'].apply(lambda x: ', '.join([label_mapping.get(word, word) for word in x.split(', ')]))

    # Create a bar plot using Seaborn
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Total Count', y='Type of System', data=df, orient='h', ci=None, palette='pastel', edgecolor='grey')
    plt.title(season)
    
    # Add text annotations for total count and percentage on the right side of each bar
    for index, value in enumerate(df['Total Count']):
        percentage = df.loc[df.index[index], 'Percentage']
        plt.text(value + 20, index, f"{value} ({percentage:.2f}%)", va='center', color='black', fontweight='bold')

    # Set y-axis limit to start at 0
    plt.ylim(-0.5, len(df) - 0.5)

    # Hide axis titles
    plt.xlabel(None)
    plt.ylabel(None)

    plt.tight_layout()

    # Save the plot as an image file
    if filter:
        output_file = os.path.join(output_directory, f'{season}_barplot_filtered.png')
    else:
        output_file = os.path.join(output_directory, f'{season}_barplot.png')
    plt.savefig(output_file)
    print(f'{output_file} saved.')
    
def plot_barplots(df, filtered_df, season, output_directory):
    # Plot bar plot for all systems
    plot_barplot(df, season, output_directory)

    # Reset the index of filtered DataFrame before plotting
    filtered_df = filtered_df.reset_index(drop=True)

    # Plot bar plot for filtered systems
    plot_barplot(filtered_df, season, output_directory, filter=True)

if __name__ == "__main__":
    output_directory = '../figures/periods_statistics/'
    os.makedirs(output_directory, exist_ok=True)

    # List of season names
    seasons = ['JJA', 'MAM', 'SON', 'DJF', 'total']

    for season in seasons:
        # Read data from CSV file
        df = pd.read_csv(f'{season}_count_of_systems.csv')
        df = df.sort_values(by='Total Count', ascending=False)

        # Assuming 'Percentage' and 'Type of System' columns are present in the DataFrame
        filtered_df = get_filtered_df(df)

        plot_barplots(df, filtered_df, season, output_directory)
