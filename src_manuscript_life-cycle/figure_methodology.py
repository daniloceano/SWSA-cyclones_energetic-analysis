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
    ax.text(0.95, 0.05, label, fontsize=16, fontweight='bold', ha='center',
             va='center', transform=ax.transAxes)

def plot_vorticity(df, ax=None, vorticity=None, fig_dir=None):

    ax.plot(vorticity.time, vorticity.zeta, linewidth=3, color='gray', label=r'ζ')

    ax2 = ax.twinx()
    ax2.axis('off')

    ax2.plot(vorticity.time, vorticity.filtered_vorticity, linewidth=3, c='#f25c54', label=r'$ζ_{f}$')
    ax2.plot(vorticity.time, vorticity.vorticity_smoothed, linewidth=3, c='#5fa8d3', label=r'$ζ_{fs}$')
    ax2.plot(vorticity.time, vorticity.vorticity_smoothed2, linewidth=3, c='k', label=r'$ζ_{fs^{2}}$')

    # Combine the legends from both axes into a single legend
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper left',
               fontsize=18, bbox_to_anchor=(-0.45, 0.8))

    plot_label(ax, "(A)")
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

    ax2 = ax.twinx()
    ax2.axis('off')

    ax2.plot(df_copy.index, vorticity_smoothed, lw=3, c='k')

    plot_label(ax, label)

    adjust_labels(ax)

def plot_specific_peaks_valleys(df, ax, *kwargs):
    # Define the series and colors for plotting
    series_colors = {'z': 'k', 'dz': '#d62828', 'dz2': '#f7b538'}
    marker_sizes = {'z': 190, 'dz': 120, 'dz2': 50}

    zeta = df['z']

    ax2 = ax.twinx()
    ax2.axis('off')

    for key in kwargs:
        key_name = key.split('_')[0]

        peaks_valleys_series = df[f"{key_name}_peaks_valleys"]

        color = series_colors[key_name]
        marker_size = marker_sizes[key_name]
        zorder = 99 if key_name == 'z' else 100 if key_name == 'dz' else 101

        mask_notna = peaks_valleys_series.notna()
        mask_peaks = peaks_valleys_series == 'peak'

        # Plot peaks
        ax2.scatter(df.index[mask_notna & mask_peaks],
                   zeta[mask_notna & mask_peaks],
                   marker='o', color=color, s=marker_size, zorder=zorder)

        # Plot valleys
        ax2.scatter(df.index[mask_notna & ~mask_peaks],
                   zeta[mask_notna & ~mask_peaks],
                   marker='o', edgecolors=color, facecolors='none',
                   s=marker_size, linewidth=2, zorder=zorder)

# Sample data
track_file = './LEC_20080518_track/20080518_track_trackfile'
track = pd.read_csv(track_file, parse_dates=[0], delimiter=';', index_col=[0])
zeta_df = pd.DataFrame(track["min_max_zeta_850"].rename('zeta'))
zeta_list = zeta_df['zeta'].tolist()
times = zeta_df.index.tolist()

# Directory for saving figures
fig_dir = "../figures/manuscript_life-cycle/"
os.makedirs(fig_dir, exist_ok=True)

# Determine periods
options = {
        "plot": False,
        "plot_steps": False,
        "export_dict": False,
        "process_vorticity_args": {
            "use_filter": "auto",
            "use_smoothing_twice": "auto"}
    }

df = determine_periods(zeta_list, x=zeta_df.index.tolist(), **options)

vorticity = process_vorticity(zeta_df.copy(), **options['process_vorticity_args'])

fig = plt.figure(figsize=(6.5*3, 5*2))
ax = fig.add_subplot(231)

plot_vorticity(df, ax=ax, vorticity=vorticity, fig_dir=None)

ax2 = fig.add_subplot(232)
df_int = find.find_intensification_period(df.copy())
plot_phase(df_int, "intensification", "(B)", ax2)
plot_specific_peaks_valleys(df_int, ax2, "z_peaks", "z_valleys")

ax3 = fig.add_subplot(233)
df_decay = find.find_decay_period(df.copy())
plot_phase(df_decay, "decay", "(C)", ax3)
plot_specific_peaks_valleys(df_decay, ax3, "z_peaks", "z_valleys")

ax4 = fig.add_subplot(234)
df_mature = find.find_mature_stage(df.copy())
plot_phase(df_mature, "mature", "(D)", ax4)
plot_specific_peaks_valleys(df_mature, ax4, "z_peaks", "z_valleys")

ax5 = fig.add_subplot(235)
df_residual = find.find_residual_period(df.copy())
plot_phase(df_residual, "residual", "(E)", ax5)
#plot_specific_peaks_valleys(df_residual, ax5, "z_peaks", "z_valleys")

ax6 = fig.add_subplot(236)
df_incipient = find.find_mature_stage(df.copy())
plot_phase(df_incipient, "incipient", "(F)", ax6)
plot_specific_peaks_valleys(df_incipient, ax6, "z_peaks", "z_valleys")

fname = os.path.join(fig_dir, "methodology.png")
plt.savefig(fname, dpi=500)
print(f"{fname} created.")