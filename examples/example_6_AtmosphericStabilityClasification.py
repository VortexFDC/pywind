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
from example_2_read_txt_functions import *
from example_6_AtmosphericStabilityClassification_functions import build_ds_stability, plot_stability_frequency
import os



# =============================================================================
# 2. Define Paths and Site
#  Repeat the process in chapter 3 to read netcdf and merge datasets
# =============================================================================

SITE = 'froya'
pwd = os.getcwd()
base_path = str(os.path.join(pwd, '../data'))

vortex_netcdf = os.path.join(base_path, f'{SITE}/vortex/SERIE/vortex.serie.era5.utc0.nc')

print()
print('#'*26, 'Vortex F.d.C. 2025', '#'*26)
print()

# Read Text Series

ds_vortex = xr.open_dataset(vortex_netcdf)

ds_vortex = ds_vortex[['M','Dir','T']]

print("Levels in Vortex dataset:")
print(ds_vortex['lev'].values)

print("Select two levels for which to compute the stability classification:")
ds_vortex = ds_vortex.isel(lev=[2, 5])
print("Selected levels:")
print(ds_vortex['lev'].values)

print(ds_vortex)

# Calcular z1 i z2
z1, z2 = ds_vortex['lev'].values
print(f'Levels for stability analysis: z1={z1}, z2={z2}')
ds_obs_stability = build_ds_stability(ds_vortex,z1, z2)

site = "Froya"
lat = str(round(ds_vortex['lat'].values[0], 2))
lon = str(round(ds_vortex['lon'].values[0], 2))
output_dir = "output"
# Plot stability frequency
title = f'Obs: {site}, lat={lat}, lon={lon}, z1={z1}m and z2={z2}m'
print(f'Plotting stability frequency for {site} with levels z1={z1} and z2={z2}')
plot_stability_frequency(ds_obs_stability, site, output_dir, plot_types='all', title=title)








