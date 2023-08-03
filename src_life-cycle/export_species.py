# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_species.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo <danilo.oceano@gmail.com>           +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 14:07:51 by Danilo            #+#    #+#              #
#    Updated: 2023/08/03 16:34:39 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import glob

csv_directory = '../periods-energetics/BY_RG-all_raw/'
csv_files = glob.glob(f'{csv_directory}/*')

phase_counts = {}
seasonal_phase_counts = {season: {} for season in ['DJF', 'MAM', 'JJA', 'SON']}

month_season_map = {
        12: 'DJF', 1: 'DJF', 2: 'DJF',
        3: 'MAM', 4: 'MAM', 5: 'MAM',
        6: 'JJA', 7: 'JJA', 8: 'JJA',
        9: 'SON', 10: 'SON', 11: 'SON'
    }

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    if 'residual' in df.columns:
        df = df.drop(['residual'], axis=1)
    phases = list(df.columns)
    phase_arrangement = ', '.join(phases)
    phase_counts[phase_arrangement] = phase_counts.get(phase_arrangement, 0) + 1

    # Get the month of the system_start
    if len(df.columns) > 0:
        system_start = pd.to_datetime(df.iloc[0][0])
        system_month = system_start.month
    else:
        continue

    # Find the corresponding season in the month_season_map
    corresponding_season = month_season_map[system_month]
    
    # Count the seasonal occurrences of the current type beginning on the first day of the event
    seasonal_phase_counts[corresponding_season].setdefault(phase_arrangement, 0)
    seasonal_phase_counts[corresponding_season][phase_arrangement] += 1

# Export total count to CSV
total_df = pd.DataFrame(list(phase_counts.items()), columns=['Type of System', 'Total Count'])
total_df.to_csv('total_count_of_systems.csv', index=False)

# Export seasonal counts to separate CSV files
for season in seasonal_phase_counts.keys():
    season_df = pd.DataFrame(list(seasonal_phase_counts[season].items()), columns=['Type of System', 'Count'])
    season_df.to_csv(f'{season}_count_of_systems.csv', index=False)
