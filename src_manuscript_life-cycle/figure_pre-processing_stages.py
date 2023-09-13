import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os

from cyclophaser import determine_periods
from cyclophaser.determine_periods import process_vorticity
from cyclophaser import find_stages as find

def adjust_labels(ax):
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
    date_format = mdates.DateFormatter("%b-%d")
    ax.xaxis.set_major_formatter(date_format)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=12)
    plt.setp(ax.get_yticklabels(), fontsize=12)

def plot_label(ax, label):
    ax.text(0.05, 0.9, label, fontsize=16, fontweight='bold', ha='center',
             va='center', transform=ax.transAxes)

def plot_vorticity(df, ax=None, vorticity=None, fig_dir=None):

    ax.plot(vorticity.time, vorticity.zeta, linewidth=3, color='gray', label=r'ζ')
    ax.plot(vorticity.time, vorticity.filtered_vorticity, linewidth=3, c='#f25c54', label=r'$ζ_{f}$')
    ax.plot(vorticity.time, vorticity.vorticity_smoothed, linewidth=3, c='#5fa8d3', label=r'$ζ_{fs}$')
    ax.plot(vorticity.time, vorticity.vorticity_smoothed2, linewidth=3, c='k', label=r'$ζ_{fs^{2}}$')

    ax.legend(loc= 'center left', bbox_to_anchor=(-0.4, 0.5), ncol=1, fontsize=18)

    plot_label(ax, "A)")
    adjust_labels(ax)
 

def plot_phase(df, phase, label, ax=None):
    # Create a copy of the DataFrame
    df_copy = df.copy()

    zeta = df_copy['z_unfil']
    vorticity_smoothed = df_copy['z']

    colors_phases = {'incipient': '#65a1e6', 'intensification': '#f7b538',
                     'mature': '#d62828', 'decay': '#9aa981', 'residual': 'gray'}

    # Find the start and end indices of the period
    phase_starts = df_copy[(df_copy['periods'] == phase) &
                            (df_copy['periods'].shift(1) != phase)].index
    phase_ends = df_copy[(df_copy['periods'] == phase) &
                          (df_copy['periods'].shift(-1) != phase)].index

    ax.axhline(0, c='k', linewidth=1, linestyle='dashed')

    # Iterate over the periods and fill the area
    for start, end in zip(phase_starts, phase_ends):
        ax.fill_between(df_copy.index, zeta, where=(df_copy.index >= start) &
                        (df_copy.index <= end), alpha=0.7, color=colors_phases[phase])

    ax.plot(df_copy.index, zeta, c='gray', lw=3)
    ax.plot(df_copy.index, vorticity_smoothed, lw=3, c='k')

    plot_label(ax, label)

    adjust_labels(ax)

def plot_specific_peaks_valleys(df, ax, *kwargs):
    # Define the series and colors for plotting
    series_colors = {'z': 'k', 'dz': '#d62828', 'dz2': '#f7b538'}
    marker_sizes = {'z': 190, 'dz': 120, 'dz2': 50}

    zeta = df['z']

    for key in kwargs:
        key_name = key.split('_')[0]

        peaks_valleys_series = df[f"{key_name}_peaks_valleys"]

        color = series_colors[key_name]
        marker_size = marker_sizes[key_name]
        zorder = 99 if key_name == 'z' else 100 if key_name == 'dz' else 101

        mask_notna = peaks_valleys_series.notna()
        mask_peaks = peaks_valleys_series == 'peak'

        # Plot peaks
        ax.scatter(df.index[mask_notna & mask_peaks],
                   zeta[mask_notna & mask_peaks],
                   marker='o', color=color, s=marker_size, zorder=zorder)

        # Plot valleys
        ax.scatter(df.index[mask_notna & ~mask_peaks],
                   zeta[mask_notna & ~mask_peaks],
                   marker='o', edgecolors=color, facecolors='none',
                   s=marker_size, linewidth=2, zorder=zorder)

# Sample data from cycllone RG3-0.99-20080518
track_file = '/home/daniloceano/Documents/Programs_and_scripts/SWSA-cyclones_energetic-analysis/LEC_results-0.99/RG3-0.99-20080518_ERA5_track-15x15/RG3-0.99-20080518_ERA5_track-15x15_track'
track = pd.read_csv(track_file, parse_dates=[0], delimiter=';', index_col=[0])
zeta_df = pd.DataFrame(track["min_zeta_850"].rename('zeta'))

# Directory for saving figures
fig_dir = "../figures/manuscript_life-cycle/"
os.makedirs(fig_dir, exist_ok=True)

# Determine periods
options = {
        "vorticity_column": 'min_zeta_850',
        "plot": False,
        "plot_steps": False,
        "export_dict": False,
        "process_vorticity_args": {
            "use_filter": "auto",
            "use_smoothing_twice": "auto"}
    }

df = determine_periods(track_file, **options)
vorticity = process_vorticity(zeta_df, **options['process_vorticity_args'])

fig = plt.figure(figsize=(6.5*3, 5*2))
ax = fig.add_subplot(231)

plot_vorticity(df, ax=ax, vorticity=vorticity, fig_dir=None)

ax2 = fig.add_subplot(232)
df_int = find.find_intensification_period(df.copy())
plot_phase(df_int, "intensification", "B)", ax2)
plot_specific_peaks_valleys(df, ax2, "z_peaks", "z_valleys")

fname = os.path.join(fig_dir, "methodology.png")
plt.savefig(fname, dpi=500)
print(f"{fname} created.")