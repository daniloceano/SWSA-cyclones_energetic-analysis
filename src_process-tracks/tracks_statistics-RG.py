# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    tracks_statistics-RG.py                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/06/19 10:06:59 by Danilo            #+#    #+#              #
#    Updated: 2023/06/19 11:49:09 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import matplotlib.pyplot as plt
from scipy.stats import mode
import seaborn as sns
import pandas as pd
import glob
import os

def extract_systems(df, RG, quantiles):
    max_vor42 = df.groupby('track_id')['vor42'].max()
    extracted_systems = []
    for q in quantiles:
        systems_above_quantile = df[df['track_id'].isin(max_vor42[max_vor42 >= max_vor42.quantile(q)].index)].reset_index(drop=True)
        systems_above_quantile = systems_above_quantile.drop('index', axis=1)
        extracted_systems.append(systems_above_quantile)
        export_path = os.path.join("../stats_tracks/BY_RG", f'tracks-RG{RG}_q{q}.csv')
        systems_above_quantile.to_csv(export_path, index=False)
        print(f"Extracted systems for quantile {q} exported to: {export_path}")
    return extracted_systems


def plot_stats(df, quantiles, RG):
    # Calculate mean, median, mode, and quantiles
    max_vor42 = df.groupby('track_id')['vor42'].max()
    mean_value = max_vor42.mean()
    median_value = max_vor42.median()
    mode_value = mode(max_vor42)[0][0]
    quantiles = max_vor42.quantile(quantiles)  # Include quantile 0.999

    # Create a kernel density estimate plot (PDF curve)
    plt.close('all')
    plt.figure()
    sns.kdeplot(max_vor42, shade=True)
    plt.xlabel('Maximum vor42')
    plt.ylabel('PDF')

    # Add vertical lines for mean, median, mode, and quantiles
    plt.axvline(mean_value, color='r', linestyle='--', label=f'Mean: {mean_value:.2f}')
    plt.axvline(median_value, color='b', linestyle='-.', label=f'Median: {median_value:.2f}')
    plt.axvline(mode_value, color='m', linestyle=':', label=f'Mode: {mode_value:.2f}')
    for q in quantiles:
        label = f'Quantile {quantiles[quantiles == q].index[0]}: {q:.2f}'
        plt.axvline(q, color='g', linestyle=':', label=label)

    # Add the number of occurrences for each quantile in the plot title
    title = f"Number of cyclones in RG{RG}: {len(df['track_id'].unique())}\n"
    title += f"Quantile 0.9: {int(len(max_vor42[max_vor42 >= quantiles[0.9]]))} occurrences\n"
    title += f"Quantile 0.95: {int(len(max_vor42[max_vor42 >= quantiles[0.95]]))} occurrences\n"
    title += f"Quantile 0.99: {int(len(max_vor42[max_vor42 >= quantiles[0.99]]))} occurrences\n"
    title += f"Quantile 0.999: {int(len(max_vor42[max_vor42 >= quantiles[0.999]]))} occurrences"  # Add quantile 0.999
    plt.title(title)

    # Move the legend to the right of the plot, outside the plotting area
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    # Create the directory if it doesn't exist
    directory = '../stats_tracks/BY_RG'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save the figure
    file_path = os.path.join(directory, f'PDF-RG{RG}.png')
    plt.savefig(file_path, dpi=300)
    print("Saved to {}".format(file_path))



for RG in range(1,4):
    
    directory = f'../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG{RG}_csv/'

    files = glob.glob(f"{directory}/*")

    df = pd.DataFrame([])

    for file in files:
        try:
            tmp = pd.read_csv(file, header=None)
            if not tmp.empty:  # Skip empty files
                df = pd.concat([df, tmp])
        except pd.errors.EmptyDataError:
            continue  # Skip the empty file and continue with the next file

    df.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']
    df = df.reset_index()

    for col in ['lon vor', 'lon mslp', 'lon 10spd']:
        df[col] = df[col].apply(lambda x: x - 360 if x > 180 else x)

    # Group the DataFrame by 'track_id' and find the maximum 'vor42'
    max_vor42 = df.groupby('track_id')['vor42'].max()

    quantiles = [0.9, 0.95, 0.99, 0.999]
    extracted_systems = extract_systems(df, RG, quantiles)
    
    plot_stats(df, quantiles, RG)