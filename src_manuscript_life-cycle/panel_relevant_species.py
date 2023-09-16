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

    ax.plot(vorticity.time, vorticity.zeta, linewidth=6, color='gray', alpha=0.8, label=r'ζ')
    ax.plot(vorticity.time, vorticity.vorticity_smoothed, linewidth=4,
             c='#1d3557', alpha=0.8, label=r'$ζ_{fs}$')
    ax.plot(vorticity.time, vorticity.vorticity_smoothed2, linewidth=4,
             c='#e63946', alpha=0.6, label=r'$ζ_{fs^{2}}$')

    if len(vorticity.time) < 50:
        dt = pd.Timedelta(1, unit='h')
    else:
       dt = pd.Timedelta(0, unit='h')

    # Shade the areas between the beginning and end of each period
    for phase, (start, end) in phases_dict.items():
        # Extract the base phase name (without suffix)
        base_phase = phase.split()[0]

        # Access the color based on the base phase name
        color = colors_phases[base_phase]

        # Fill between the start and end indices with the corresponding color
        ax.fill_between(vorticity.time, vorticity.zeta.values,
                         where=(vorticity.time >= start) & (vorticity.time <= end + dt),
                        alpha=0.5, color=color, label=base_phase)

    if i == 0:
        ax.legend(loc='upper right', bbox_to_anchor=(2.1, -2.2), fontsize=14)

    ax.text(0.84, 0.84, labels[i], fontsize=16, fontweight='bold', ha='left', va='bottom', transform=ax.transAxes)

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    date_format = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(date_format)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

options = {
        "vorticity_column": 'vor42',
        "plot": False,
        "plot_steps": False,
        "export_dict": False,
        "process_vorticity_args": {
            "use_filter": False,
            "use_smoothing_twice": "auto"}
    }

results_dirs = {'RG1': '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/',
               'RG2': '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG2_csv/',
               'RG3': '../raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG3_csv/'
               }

systems_for_representing = ['RG1_19800410', # Ic, It, M, D
                             'RG1_19820363', # It, M, D
                             'RG1_19820102', # Ic, It, M, D, It2, M2, D2
                             'RG1_19830146', # It, M, D, It2, M2, D2
                             'RG1_19800232', # Ic, D, It, M, D
                             'RG1_19820786', # It
                             'RG1_19830405', # D, It, M, D2
                            ]   

labels = ["A)", "B)", "C)", "D)", "E)", "F)", "G)"]

count_species = pd.read_csv('total_count_of_systems_filtered.csv', index_col=0)

fig = plt.figure(figsize=(10, 10))

i = 0
for system in systems_for_representing:

    print(f"Processing {system}")

    cyclone_id = system.split('_')[1]
    RG = system.split('_')[0]

    year = str(cyclone_id[0:4])

    results_dir = results_dirs[RG]

    results = glob(f'{results_dir}/*')

    track_file_years = [result for result in results if year in result]

    flag = False
    for track_file in track_file_years:

        try:
            tracks = pd.read_csv(track_file)
        except:
            continue

        cyclone_id = int(cyclone_id)
        
        tracks.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']

        track = tracks[tracks['track_id']==cyclone_id][['date','vor42']]

        if len(track) > 0:
            print (f"Found track for {cyclone_id} in file {track_file}")

            # Create temporary files for cyclophaser function
            track = track.rename(columns={"date":"time"})
            track['vor42'] = - track['vor42'] * 1e-5
            tmp_file = (f"tmp_{RG}-{cyclone_id}.csv")
            track.to_csv(tmp_file, index=False, sep=';')


            df_periods = determine_periods(tmp_file, **options)
            periods = list(df_periods['periods'].unique())

            df = determine_periods(tmp_file, **options)
            periods_dict = periods_to_dict(df)

            ax = fig.add_subplot(3, 3, i+1)

            zeta_df = pd.DataFrame(track['vor42'].rename('zeta'))
            zeta_df.index = pd.to_datetime(track['time'])

            vorticity = process_vorticity(zeta_df.copy(), **options['process_vorticity_args'])

            plot_all_periods(periods_dict, df, ax, vorticity, i)

            flag = True
            i += 1

            os.remove(tmp_file)
        
plt.subplots_adjust(hspace=0.6)
fname = f'../figures/manuscript_life-cycle/main_species_life-cycle.png'
plt.savefig(fname, dpi=500)
print(f"{fname} created.")