from determine_periods import determine_periods
import pandas as pd
 
id = 19800730
month = 10
track_file_raw = f'../../SWSA-cyclones_energetic-analysis/raw_data/TRACK_BY_RG-20230606T185429Z-001/24h_1000km_add_RG1_csv/ff_trs_neg_ExSAt_1980{month}_joint_mslp_10spd_GenRG1.csv'

track = pd.read_csv(track_file_raw, parse_dates=[2])
track.columns = ['track_id', 'dt', 'date', 'lon vor', 'lat vor', 'vor42', 'lon mslp', 'lat mslp', 'mslp', 'lon 10spd', 'lat 10spd', '10spd']
track_id = track[track['track_id'] == id]

track_format = track_id[['date', 'lat vor', 'lon vor', 'lon mslp', 'lat mslp', 'vor42', 'mslp', '10spd']]
track_format['lon vor'] = (track_format['lon vor'] + 180) % 360 - 180
track_format.columns = ['time','Lat','Lon','length', 'width', 'min_zeta_850', 'min_hgt_850', 'max_wind_850']
track_format[['length', 'width']] = 15
track_format['min_zeta_850'] = track_format['min_zeta_850'] * -1e-5

track_file = './track-test-periods'
track_format.to_csv(track_file, sep=';', index=False)

output_directory = './'
# determine_periods(track_file, output_directory)
