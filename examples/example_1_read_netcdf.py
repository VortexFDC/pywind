# =============================================================================
# Authors: Oriol L & Arnau T
# Company: Vortex F.d.C.
# Year: 2024
# =============================================================================

"""
Overview:
---------
This script demonstrates the process of reading various types of meteorological data files.
The script uses functions to load and manipulate data from four distinct file formats:

1. Measurements (NetCDF) - Contains multiple heights and variables.
2. Vortex NetCDF - NetCDF file format with multiple heights and variables.

Data Storage:
------------
The acquired data is stored in two data structures for comparison and analysis:
- Xarray Dataset
- Pandas DataFrame

Objective:
----------
- To understand the variance in data storage when using Xarray and Pandas.
- Utilize the 'describe' method from Pandas for a quick statistical overview of the dataset.
"""


# =============================================================================
# 1. Import Libraries
# =============================================================================

import xarray as xr
import os

# =============================================================================
# 2. Define Paths and Site
# =============================================================================

SITE = 'froya'
pwd = os.getcwd()
# you may have to adjust this path relative to script depending on your Python configuration
base_path = os.path.join(pwd, '../data')

print()
measurements_netcdf = os.path.join(base_path, f'{SITE}/measurements/obs.nc')
vortex_netcdf = os.path.join(base_path, f'{SITE}/vortex/SERIE/vortex.serie.era5.utc0.nc')

# Print filenames
print('Measurements NetCDF: ', measurements_netcdf)
print('Vortex NetCDF: ', vortex_netcdf)

print()
print('#'*26, 'Vortex F.d.C. 2024', '#'*26)
print()

# =============================================================================
# 3. Read NetCDF functions
# =============================================================================

# Read measurements NetCDF
ds_measurements = xr.open_dataset(measurements_netcdf)
#print(ds_measurements)

# Read Vortex NetCDF
ds_vortex = xr.open_dataset(vortex_netcdf)
#print(ds_vortex)

# =============================================================================
# 4. Convert to Pandas DataFrame and inspect
# =============================================================================

df_measurements = ds_measurements.to_dataframe()
df_vortex = ds_vortex.to_dataframe()

#print(df_measurements.head())
#print(df_vortex.head())

# =============================================================================
# 5. Interpolate to the same height
# =============================================================================

max_height = ds_measurements.squeeze().coords['lev'].max().values
print("Max height in measurements: ", max_height)
print()

ds_vortex = ds_vortex.interp(lev=max_height)
#print(ds_vortex)

# =============================================================================
# 6. Now we can compare statistics for M and Dir
# =====

# Drop coordinates lat and lon and dimensions
ds_vortex = ds_vortex.squeeze().reset_coords(drop=True)
ds_measurements = ds_measurements.squeeze().reset_coords(drop=True)

print('Vortex:')
print(ds_vortex[['M', 'Dir']].to_dataframe().describe().apply(lambda x: x.apply('{:,.6f}'.format)))
print()

print('Measurements:')
print(ds_measurements[['M', 'Dir']].to_dataframe().describe().apply(lambda x: x.apply('{:,.6f}'.format)))
print()