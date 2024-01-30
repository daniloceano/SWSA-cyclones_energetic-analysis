# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    check_residual.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: daniloceano <danilo.oceano@gmail.com>      +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/01/30 14:30:32 by daniloceano       #+#    #+#              #
#    Updated: 2024/01/30 15:55:14 by daniloceano      ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


"""
This script reads the tracks processed by export_periods.py and checks which 
systems have residual stage. 
"""

from glob import glob
import pandas as pd
import os

from cyclophaser import determine_periods

TRACKS_DIRECTORY = "../processed_tracks_with_periods/"
FIGURES_DIRECTORY = "../figures/periods/residual_only/"

def export_period(track):

    system_id = track['track_id'].iloc[0]

    periods_outfile = os.path.join(FIGURES_DIRECTORY, f"{system_id}_periods")

    # if not periods_outfile_exists:
    options = {
        "plot": periods_outfile,
        "plot_steps": False,
        "export_dict": False,
        "process_vorticity_args": {
            "use_filter": False,
            "use_smoothing_twice": "auto"}
    }

    x = pd.to_datetime(track['date']).tolist()
    y = (-track['vor42'] * 10e5).tolist()

    determine_periods(y, x, **options)

os.makedirs(FIGURES_DIRECTORY, exist_ok=True)

track_files = glob(os.path.join(TRACKS_DIRECTORY, "*.csv"))
tracks = pd.concat((pd.read_csv(file) for file in track_files))

residual_tracks = tracks[tracks['phase'].str.lower().str.contains('residual') & ~tracks['phase'].isna()]

system_with_residual_ids = residual_tracks['track_id'].unique()

for id in system_with_residual_ids:
    track = tracks[tracks['track_id'] == id]
    export_period(track)
