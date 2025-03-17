# =============================================================================
# Authors: Oriol L & Arnau T
# Company: Vortex F.d.C.
# Year: 2024
# =============================================================================

"""
Overview:
---------
This script demonstrates the process of reading and processing various types of meteorological data files. The goal is to compare measurements from different sources and formats by resampling, interpolating, and merging the data for further analysis.

The script uses functions to load and manipulate data from four distinct file formats:

1. Measurements (NetCDF) - Contains multiple heights and variables.
2. Vortex NetCDF - NetCDF file format with multiple heights and variables.
3. Vortex Text Series - Text file containing time series data of meteorological measurements.
4. Measurements Text Series - Text file containing time series data of observations.

Data Storage:
-------------
The acquired data is stored and processed in two data structures for comparison and analysis:
- **Xarray Dataset**: For handling multi-dimensional arrays of the meteorological data, useful for complex operations and transformations.
- **Pandas DataFrame**: For flexible and powerful data manipulation and analysis, allowing easy integration and comparison of different datasets.

Objective:
----------
- **Read and Interpolate Data**: Load data from NetCDF and text files, and interpolate Vortex data to match the measurement levels.
- **Resample Data**: Convert the time series data to an hourly frequency to ensure uniformity in the analysis.
- **Data Comparison**: Merge the datasets to facilitate a detailed comparison of measurements from different sources.
- **Statistical Overview**: Utilize the 'describe' method from Pandas for a quick statistical summary of the datasets, providing insights into the distribution and characteristics of the data.
- **Concurrent Period Analysis**: Clean the data by removing non-concurrent periods (no data) to focus on the overlapping timeframes for accurate comparison.

By following these steps, the script aims to provide a comprehensive approach to handling and analyzing meteorological data from various sources, ensuring a clear understanding of the data's behavior and relationships.
"""

# =============================================================================
# 1. Import Libraries
# =============================================================================

from example_3_merge_functions import *
import pandas as pd

# =============================================================================
# 2. Define Paths and Site
# =============================================================================

SITE = 'froya'
pwd = os.getcwd()
base_path = str(os.path.join(pwd, '../data'))

print()
measurements_netcdf = os.path.join(base_path, f'{SITE}/measurements/obs.nc')
vortex_netcdf = os.path.join(base_path, f'{SITE}/vortex/SERIE/vortex.serie.era5.utc0.nc')

vortex_txt = os.path.join(base_path, f'{SITE}/vortex/SERIE/vortex.serie.era5.utc0.100m.txt')
measurements_txt = os.path.join(base_path, f'{SITE}/measurements/obs.txt')

# Print filenames
print('Measurements NetCDF: ', measurements_netcdf)
print('Vortex NetCDF: ', vortex_netcdf)

print()
print('#'*26, 'Vortex F.d.C. 2024', '#'*26)
print()

# =============================================================================
# 3. Read Vortex Series in NetCDF and Text
# =============================================================================

# Read NetCDF
ds_obs_nc = xr.open_dataset(measurements_netcdf)
ds_vortex_nc = xr.open_dataset(vortex_netcdf)
#ds_vortex_nc = ds_vortex_nc.rename_vars({'D': 'Dir'})

# Read Text Series
ds_vortex_txt = read_vortex_serie(vortex_txt)
df_obs_txt = read_vortex_obs_to_dataframe(measurements_txt)[['M', 'Dir']]
ds_obs_txt = convert_to_xarray(df_obs_txt)[['M', 'Dir']]

# =============================================================================
# 4. Interpolate Vortex Series to the same Measurements level. Select M and Dir.
# =============================================================================

print()
max_height = ds_obs_nc.squeeze().coords['lev'].max().values
print("Max height in measurements: ", max_height)
ds_obs_nc = ds_obs_nc.sel(lev=max_height).squeeze().reset_coords(drop=True)[['M', 'Dir']]

ds_vortex_nc = ds_vortex_nc.interp(lev=max_height).squeeze().reset_coords(drop=True)[['M', 'Dir']]
ds_vortex_txt = ds_vortex_txt[['M', 'Dir']].squeeze().reset_coords(drop=True)

# =============================================================================
# 5. Measurements Time Resampling to Hourly
# =============================================================================

# No need to perform any resampling to Vortex data, as SERIES products is already hourly

# convert ds_obs_nc to hourly
ds_obs_nc = ds_obs_nc.resample(time='1H').mean()
# convert ds_obs_txt to hourly
ds_obs_txt = ds_obs_txt.resample(time='1H').mean()

# =============================================================================
# 6. Convert all to DataFrame, Rename and Merge
# =============================================================================

df_obs_nc = ds_obs_nc.to_dataframe()
df_vortex_nc = ds_vortex_nc.to_dataframe()
df_obs_txt = ds_obs_txt.to_dataframe()
df_vortex_txt = ds_vortex_txt.to_dataframe()

# rename columns so they do now have the same name when merging
df_obs_nc.columns = ['M_obs_nc', 'Dir_obs_nc']
df_vortex_nc.columns = ['M_vortex_nc', 'Dir_vortex_nc']
df_obs_txt.columns = ['M_obs_txt', 'Dir_obs_txt']
df_vortex_txt.columns = ['M_vortex_txt', 'Dir_vortex_txt']

# merge using index (time) all dataframes
df_nc = df_obs_nc.merge(df_vortex_nc, left_index=True, right_index=True)
df_txt = df_obs_txt.merge(df_vortex_txt, left_index=True, right_index=True)
df = df_nc.merge(df_txt, left_index=True, right_index=True)
print()

# force to show all describe columns
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())
    print()
    print(df.describe())
    print()

print("After Cleaning Nodatas: Concurrent period")
print()
# If you want to have only concurrent period, remove nodatas
df = df.dropna(how='any', axis=0)
# force to show all describe columns
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df.head())
    print()
    print(df.describe())
    print()