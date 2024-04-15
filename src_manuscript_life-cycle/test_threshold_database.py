import os
from glob import glob 
import pandas as pd

# Configuration variables
TRACKS_DIRECTORY = "../processed_tracks_with_periods/"

system_ids = [20101172, 20190644, 20001176, 19840092, 19970580, 20170528]

for system_id in system_ids:
    tracks = glob(TRACKS_DIRECTORY + "*.csv")

    for track_file in tracks:
        tracks_df = pd.read_csv(track_file, index_col=0)
        desired_cyclone = tracks_df[tracks_df['track_id'] == system_id]

        if desired_cyclone.empty:
            continue  # Skip if no track files for this system_id

        desired_cyclone = tracks_df[tracks_df['track_id'] == int(system_id)]

        desired_cyclone.loc[:, 'date'] = pd.to_datetime(desired_cyclone['date'])
        desired_cyclone.loc[:, 'vor42'] = desired_cyclone['vor42'] * -1e-5

        track = desired_cyclone[['date', 'vor42']].rename(columns={'vor42': 'zeta'})
        track.set_index('date', inplace=True)
        track.index.name = 'time'
        track.to_csv(f'track_test_thresholds_{system_id}.csv')
