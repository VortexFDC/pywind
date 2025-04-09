# =============================================================================
# Authors: Oriol L 
# Company: Vortex F.d.C.
# Year: 2025
# =============================================================================

"""
Overview:
---------
This script demonstrates the process of plotting basic information once a dataset from both measurements and synthetic data has been merged.

"""

# =============================================================================
# 1. Import Libraries
# =============================================================================

from example_3_merge_functions import *
import pandas as pd

# =============================================================================
# 2. Define Paths and Site
#  Repeat the process in chapter 3 to read netcdf and merge datasets
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
print('#'*26, 'Vortex F.d.C. 2025', '#'*26)
print()


# Read NetCDF
ds_obs_nc = xr.open_dataset(measurements_netcdf)
ds_vortex_nc = xr.open_dataset(vortex_netcdf)
#ds_vortex_nc = ds_vortex_nc.rename_vars({'D': 'Dir'})

# =============================================================================
# 3. Interpolate Vortex Series to the same Measurements level. Select M and Dir.
# =============================================================================

print()
max_height = ds_obs_nc.squeeze().coords['lev'].max().values
print("Max height in measurements: ", max_height)
ds_obs_nc = ds_obs_nc.sel(lev=max_height).squeeze().reset_coords(drop=True)[['M', 'Dir']]

ds_vortex_nc = ds_vortex_nc.interp(lev=max_height).squeeze().reset_coords(drop=True)[['M', 'Dir']]


# convert ds_obs_nc to hourly
ds_obs_nc = ds_obs_nc.resample(time='1h').mean()

# =============================================================================
# 6. Convert all to DataFrame, Rename and Merge
# =============================================================================

df_obs_nc = ds_obs_nc.to_dataframe()
df_vortex_nc = ds_vortex_nc.to_dataframe()


# rename columns so they do now have the same name when merging
df_obs_nc.columns = ['M_obs_nc', 'Dir_obs_nc']
print("df_obs_nc columns: ", df_obs_nc.columns)
df_vortex_nc = df_vortex_nc[['M','Dir']]
print("df_vortex_nc columns: ", df_vortex_nc.columns)
df_vortex_nc.columns = ['M_vortex_nc', 'Dir_vortex_nc']

# merge using index (time) all dataframes
df = df_obs_nc.merge(df_vortex_nc, left_index=True, right_index=True)
print()

# If you want to have only concurrent period, remove nodatas
df = df.dropna(how='any', axis=0)
# force to show all describe columns

print(df.head())
