# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    boxplot.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 16:44:54 by Danilo            #+#    #+#              #
#    Updated: 2023/08/03 19:21:42 by Danilo           ###   ########.fr        #
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

def plot_boxplot(df, season, output_directory):
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

    # Create a boxplot using Seaborn
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Total Count', y='Type of System', data=df, orient='h')
    plt.xlabel('Total Count')
    plt.ylabel('Type of System')
    plt.title(f'Boxplot of Total Count by Type of System for {season} season')
    
    # Save the plot as an image file
    output_file = os.path.join(output_directory, f'{season}_boxplot.png')
    plt.savefig(output_file)
    
def plot_boxplots(df, filtered_df, season, output_directory):
    # Plot boxplot for all systems
    print(f"Boxplot for All Systems in {season} season:")
    plot_boxplot(df, season, output_directory)

    # Plot boxplot for filtered systems
    print(f"Boxplot for Filtered Systems in {season} season:")
    plot_boxplot(filtered_df, season, output_directory)

if __name__ == "__main__":
    output_directory = '../figures/periods_statistics/'
    os.makedirs(output_directory, exist_ok=True)

    # List of season names
    seasons = ['JJA', 'MAM', 'SON', 'DJF', 'total']

    for season in seasons:
        # Read data from CSV file
        df = pd.read_csv(f'{season}_count_of_systems.csv')

        # Assuming 'Percentage' and 'Type of System' columns are present in the DataFrame
        filtered_df = get_filtered_df(df)

        plot_boxplots(df, filtered_df, season, output_directory)

