import os
from glob import glob

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cyclophaser import determine_periods
from cyclophaser.determine_periods import periods_to_dict, process_vorticity

def plot_all_periods(phases_dict, df, ax, vorticity, i):
    colors_phases = {'incipient': '#65a1e6',
                      'intensification': '#f7b538',
                        'mature': '#d62828',
                          'decay': '#9aa981',
                          'residual': 'gray'}

    ax.plot(vorticity.time, vorticity.zeta, linewidth=2.5, color='gray', label=r'ζ')

    ax2 = ax.twinx()
    ax2.axis('off')
    ax2.plot(vorticity.time, vorticity.filtered_vorticity, linewidth=2, c='#d68c45', label=r'$ζ_{f}$')
    ax2.plot(vorticity.time, vorticity.vorticity_smoothed, linewidth=2, c='#1d3557', label=r'$ζ_{fs}$')
    ax2.plot(vorticity.time, vorticity.vorticity_smoothed2, linewidth=2, c='#e63946', label=r'$ζ_{fs^{2}}$')

    if i == 0:
        legend_labels = []
        # Add legend labels for Vorticity and ζ
        for label in [r'$ζ_{f}$', r'$ζ_{fs}$', r'$ζ_{fs^{2}}$']:
            legend_labels.append(label)

        ax2.legend(legend_labels, loc='lower right', bbox_to_anchor=(-0.5, 0.42))

        legend_labels = []  # To store unique legend labels
        legend_labels.append(r'ζ')

    # Shade the areas between the beginning and end of each period
    for phase, (start, end) in phases_dict.items():
        # Extract the base phase name (without suffix)
        base_phase = phase.split()[0]

        # Access the color based on the base phase name
        color = colors_phases[base_phase]

        # Fill between the start and end indices with the corresponding color
        ax.fill_between(vorticity.time, vorticity.zeta.values, where=(vorticity.time >= start) & (vorticity.time <= end),
                        alpha=0.4, color=color, label=base_phase)

        if i == 0:
            # Add the base phase name to the legend labels set
            legend_labels.append(base_phase)

    if i == 0:
        # Remove duplicate labels from the legend
        handles, labels = ax.get_legend_handles_labels()

        # Get handles and labels from ax2
        handles2, labels2 = ax2.get_legend_handles_labels()

        # Combine handles and labels from both ax and ax2
        handles += handles2
        labels += labels2

        unique_labels = []
        for label in labels:
            if label not in unique_labels and label in legend_labels:
                unique_labels.append(label)

        ax.legend(handles, unique_labels, loc='upper right', bbox_to_anchor=(-5, 1))

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    date_format = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

options = {
        "vorticity_column": 'vor42',
        "plot": False,
        "plot_steps": False,
        "export_dict": False,
        "process_vorticity_args": {
            "use_filter": False,
            "use_smoothing_twice": "auto"}
    }

results_dir = '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/'
RG = 'RG3'

count_species = pd.read_csv('total_count_of_systems_filtered.csv', index_col=0)

fig = plt.figure(figsize=(10, 10))

i = 0
for species_to_be_ploted in count_species['Type of System']:

    species_to_be_ploted = species_to_be_ploted.split(', ')
    print(f'Matching species: {species_to_be_ploted}')

    flag = False

    for track_file in glob(f'{results_dir}/*'):
        # If already plotted this phase, skip it
        if flag:
            break
        
        if "40W" in track_file:
            continue

        try:
            tracks = pd.read_csv(track_file)
        except pd.errors.EmptyDataError:
            continue
        
        tracks.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']

        id_cyclones = tracks['track_id'].unique()

        for id_cyclone in id_cyclones:

            # Create temporary files for cyclophaser function
            track = tracks[tracks['track_id']==id_cyclone][['date','vor42']]
            track = track.rename(columns={"date":"time"})
            track['vor42'] = - track['vor42'] * 1e-5
            tmp_file = (f"tmp_{RG}-{id_cyclone}.csv")
            track.to_csv(tmp_file, index=False, sep=';')

            df_periods = determine_periods(tmp_file, **options)
            periods = list(df_periods['periods'].unique())

            if species_to_be_ploted == periods:
                print(f"Matching species: {species_to_be_ploted} for {id_cyclone}")

                df = determine_periods(tmp_file, **options)
                periods_dict = periods_to_dict(df)

                print(i)
                ax = fig.add_subplot(3, 3, i+1)

                zeta_df = pd.DataFrame(track['vor42'].rename('zeta'))
                zeta_df.index = pd.to_datetime(track['time'])

                vorticity = process_vorticity(zeta_df.copy(), **options['process_vorticity_args'])

                plot_all_periods(periods_dict, df, ax, vorticity, i)

                flag = True
                i += 1

            os.remove(tmp_file)
        
fname = f'main_species_life-cycle.png'
plt.savefig(fname)
print(f"{fname} created.")