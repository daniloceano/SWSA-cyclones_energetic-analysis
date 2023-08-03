# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_species.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 14:07:51 by Danilo            #+#    #+#              #
#    Updated: 2023/08/03 14:44:18 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import glob

csv_directory = '../periods-energetics/BY_RG-all_raw/'
csv_files = glob.glob(f'{csv_directory}/*')

phase_counts = {}
seasons_list = ['Summer', 'Fall', 'Winter', 'Spring']
seasonal_phase_counts = {season: {} for season in seasons_list}

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    if 'residual' in df.columns:
        df = df.drop(['residual'], axis=1)
    phases = list(df.columns)
    phase_arrangement = ', '.join(phases)
    phase_counts[phase_arrangement] = phase_counts.get(phase_arrangement, 0) + 1

    # Assuming index contains dates in datetime format
    df.index = pd.to_datetime(df.index)
    
    # Categorize seasons based on months
    df['Season'] = pd.cut(df.index.month, [0, 2, 5, 8, 11], labels=seasons_list)
    
    # Count the occurrences of each type of system in each season
    for season in seasons_list:
        seasonal_df = df[df['Season'] == season]
        count = len(seasonal_df)
        seasonal_phase_counts[season].setdefault(phase_arrangement, 0)
        seasonal_phase_counts[season][phase_arrangement] += count

# Export total count to CSV
total_df = pd.DataFrame(list(phase_counts.items()), columns=['Type of System', 'Total Count'])
total_df.to_csv('total_count_of_systems.csv', index=False)

# Export seasonal counts to separate CSV files
for season, type_counts in seasonal_phase_counts.items():
    season_df = pd.DataFrame(list(type_counts.items()), columns=['Type of System', 'Count'])
    season_df.to_csv(f'{season.lower()}_count_of_systems.csv', index=False)