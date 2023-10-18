# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    export_species.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: Danilo  <danilo.oceano@gmail.com>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/08/03 14:07:51 by Danilo            #+#    #+#              #
#    Updated: 2023/10/18 09:44:58 by Danilo           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import pandas as pd
import glob
import os 

""" 
This script reads the tracks processed by export_periods.py and counts the number of systems
that fits in one of each category of the life cycle.
"""

# analysis_type = 'BY_RG-all'
# analysis_type = 'all'
# analysis_type = '70W'
# analysis_type = '48h'
# analysis_type = '70W-48h'
analysis_type = '70W-no-continental'


csv_directory = f'../periods-energetics/{analysis_type}/'
csv_files = glob.glob(f'{csv_directory}/*')

seasonal_phase_counts = {season: {} for season in ['DJF', 'MAM', 'JJA', 'SON']}

month_season_map = {
    12: 'DJF', 1: 'DJF', 2: 'DJF',
    3: 'MAM', 4: 'MAM', 5: 'MAM',
    6: 'JJA', 7: 'JJA', 8: 'JJA',
    9: 'SON', 10: 'SON', 11: 'SON'
}

total_systems = len(csv_files)
total_systems_season = {'DJF': 0, 'MAM': 0, 'JJA': 0, 'SON': 0}

# List of RGs
if analysis_type == 'BY_RG-all': 
    RGs = ['RG1', 'RG2', 'RG3', 'all_RG']
elif analysis_type == '70W-no-continental':
    RGs = ["SE-BR", "LA-PLATA","ARG", "SE-SAO", "SA-NAM",
                "AT-PEN", "WEDDELL", False]
else:
    RGs = ['']

for RG in RGs:

    if analysis_type == '70W-no-continental':
        RG_str = f'_{RG}'if RG else '_SAt'
        csv_directory = f'../periods-energetics/{analysis_type}{RG_str}/'
        csv_files = glob.glob(f'{csv_directory}/*')
        total_systems = len(csv_files)

    phase_counts = {}

    ## Open data
    for csv_file in csv_files:
        if analysis_type == 'BY_RG-all':
            if RG in csv_file:
                df = pd.read_csv(csv_file, index_col=[0])
            else:
                continue 
        else:
            df = pd.read_csv(csv_file, index_col=[0])

        phases = list(df.index)
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

        total_systems_season[corresponding_season] += 1

        # Count the seasonal occurrences of the current type beginning on the first day of the event
        seasonal_phase_counts[corresponding_season].setdefault(phase_arrangement, 0)
        seasonal_phase_counts[corresponding_season][phase_arrangement] += 1

    outdir = f'../periods_species_statistics/{analysis_type}/count_systems_raw/'
    os.makedirs(outdir, exist_ok=True)

    if analysis_type == 'BY_RG-all':
        suffix = f'_{RG}' if RG != '' else '_all_RG'
    elif analysis_type == '70W-no-continental':
        suffix = f'_{RG}' if RG else '_SAt'
    else:
        suffix = ''

    # Export total count and relative percentages to CSV
    total_df = pd.DataFrame(list(phase_counts.items()), columns=['Type of System', 'Total Count'])
    total_df['Percentage'] = total_df['Total Count'] / total_systems * 100
    csv_name = os.path.join(outdir, f'total_count_of_systems{suffix}.csv')
    total_df.to_csv(csv_name, index=False)
    print(f'{csv_name} saved.')

    # Export seasonal counts and relative percentages to separate CSV files
    for season in seasonal_phase_counts.keys():
        season_df = pd.DataFrame(list(seasonal_phase_counts[season].items()), columns=['Type of System', 'Total Count'])
        season_df['Percentage'] = season_df['Total Count'] / total_systems_season[season] * 100
        csv_name = os.path.join(outdir, f'{season}_count_of_systems{suffix}.csv')
        season_df.to_csv(csv_name, index=False)
        print(f'{csv_name} saved.')
