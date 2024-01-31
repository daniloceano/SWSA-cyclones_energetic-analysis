import cdsapi
import math
import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import numpy as np
from glob import glob



TRACKS_DIRECTORY = "../processed_tracks_with_periods/"
STUDY_CASE = 19920876


def get_cdsapi_data(track, infile) -> xr.Dataset:

    # Extract bounding box (lat/lon limits) from track
    min_lat, max_lat = track['lat vor'].min(), track['lat vor'].max()
    min_lon, max_lon = track['lon vor'].min(), track['lon vor'].max()

    # Apply a 15-degree buffer and round to nearest integer
    buffered_max_lat = math.ceil(max_lat + 15)
    buffered_min_lon = math.floor(min_lon - 15)
    buffered_min_lat = math.floor(min_lat - 15)
    buffered_max_lon = math.ceil(max_lon + 15)

    # Define the area for the request
    area = f"{buffered_max_lat}/{buffered_min_lon}/{buffered_min_lat}/{buffered_max_lon}" # North, West, South, East. Nort/West/Sout/East

    pressure_levels = ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70',
                       '100', '125', '150', '175', '200', '225', '250', '300', '350',
                       '400', '450', '500', '550', '600', '650', '700', '750', '775',
                       '800', '825', '850', '875', '900', '925', '950', '975', '1000']
    
    variables = ["u_component_of_wind", "v_component_of_wind", "temperature",
                 "vertical_velocity", "geopotential"]
    

    # Convert unique dates to string format for the request
    dates = pd.to_datetime(track['date'].tolist())
    start_date = dates[0].strftime("%Y%m%d")
    end_date = dates[-1].strftime("%Y%m%d")
    time_range = f"{start_date}/{end_date}"
    time_step = '3'

    # Log track file bounds and requested data bounds
    print(f"Track File Limits: max_lon (east):  min_lon (west): {min_lon}, max_lon (west): {max_lon}, min_lat (south): {min_lat}, max_lat (north): {max_lat}")
    print(f"Buffered Data Bounds: min_lon (west): {buffered_min_lon}, max_lon (east): {buffered_max_lon}, min_lat (south): {buffered_min_lat}, max_lat (north): {buffered_max_lat}")
    print(f"Requesting data for time range: {time_range}, and time step: {time_step}...")

    # Load ERA5 data
    print("Retrieving data from CDS API...")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "format": "netcdf",
            "pressure_level": pressure_levels,
            "date": time_range,
            "area": area,
            'time': f'00/to/23/by/{time_step}',
            "variable": variables,
        }, infile # save file as passed in arguments
    )

    if not os.path.exists(infile):
        raise FileNotFoundError("CDS API file not created.")
    
    ds = xr.open_dataset(infile)

    return ds

track_file = glob(f"{TRACKS_DIRECTORY}/*{str(STUDY_CASE)[:4]}*.csv")
tracks = pd.concat([pd.read_csv(f) for f in track_file])
track = tracks[tracks['track_id'] == STUDY_CASE]

infile = f"{STUDY_CASE}.nc"

if os.path.exists(infile):
    ds = xr.open_dataset(infile)
else:
    ds = get_cdsapi_data(track, infile)

print(ds)