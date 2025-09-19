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
import os
import xarray as xr
from example_2_read_txt_functions import *
from example_6_AtmosphericStabilityClassification_functions import build_ds_stability, plot_stability_frequency
from example_7_Shear_functions import *

# =============================================================================
# 2. Define Paths and Site
# =============================================================================

SITE = 'froya'
pwd = os.getcwd()
base_path = os.path.join(pwd, '../data')
vortex_netcdf = os.path.join(base_path, f'{SITE}/vortex/SERIE/vortex.serie.era5.utc0.nc')

print("\n" + "#"*26 + " Vortex F.d.C. 2025 " + "#"*26 + "\n")

# =============================================================================
# 3. Read Vortex Dataset and Basic Info
# =============================================================================

ds_vortex = xr.open_dataset(vortex_netcdf)
ds_vortex = ds_vortex[['M', 'Dir', 'T']]  # Keep only relevant variables

print("Levels in Vortex dataset:")
print(ds_vortex['lev'].values)

# Print mean wind speed at each level
ds_vortex_mean = ds_vortex.mean(dim='time')
print("Mean wind speeds at each level:")
for level, speed in zip(ds_vortex['lev'].values, ds_vortex_mean['M'].values.flatten()):
    print(f"Level {level:.0f} m: {speed:.2f} m/s")

# =============================================================================
# 4. Add Time Features and Plot Profiles by Hour and Month
# =============================================================================

ds_vortex['hour'] = ds_vortex['time'].dt.hour
ds_vortex['month'] = ds_vortex['time'].dt.month

# Group by hour and month, then plot
ds_vortex_meanhour = ds_vortex.groupby('hour').mean(dim='time')
ds_vortex_meanmonth = ds_vortex.groupby('month').mean(dim='time')
plot_shear_profile_by(ds_vortex_meanhour, ds_vortex['lev'].values, "hour", SITE)
plot_shear_profile_by(ds_vortex_meanmonth, ds_vortex['lev'].values, "month", SITE)

# =============================================================================
# 5. Temperature Binning and Profile Plot
# =============================================================================

# Compute temperature bins at lev=2 (one value per time)
T_bins = (ds_vortex.isel(lev=2)['T'] / 5).astype(int).squeeze()

# Add T_bins as a variable with dimension 'time'
ds_vortex = ds_vortex.assign(T_bins=("time", T_bins.data))

# Group by T_bins and plot
ds_vortex_meanT = ds_vortex.groupby('T_bins').mean(dim='time')
plot_shear_profile_by(ds_vortex_meanT, ds_vortex['lev'].values, "T_bins", f"(T/5) {SITE}")

# =============================================================================
# 6. Atmospheric Stability Classification and Profile Plot
# =============================================================================

print("Select two levels for which to compute the stability classification:")
ds_vortex_stab = ds_vortex.isel(lev=[2, 5])
print("Selected levels:", ds_vortex_stab['lev'].values)

# Calculate stability between selected levels
z1, z2 = ds_vortex_stab['lev'].values
print(f'Levels for stability analysis: z1={z1}, z2={z2}')
ds_obs_stability = build_ds_stability(ds_vortex_stab, z1, z2)

# Add stability as a variable and plot grouped profiles
ds_vortex['stability'] = ds_obs_stability['stability']
ds_vortex_mean_stab = ds_vortex.groupby('stability').mean(dim='time')
plot_shear_profile_by(ds_vortex_mean_stab, ds_vortex['lev'].values, "stability", SITE)


### 
# now make an xy plot Temp vs stablity
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(ds_vortex['T'].isel(lev=2), ds_vortex['stability'], alpha=0.5)
plt.xlabel('Temperature (°C)')
plt.ylabel('Stability Class')
plt.title(f'Temperature vs Stability Class - {SITE.capitalize()}')
plt.grid()
plt.yticks(ticks=range(7), labels=[
    "0 - Very Unstable",
    "1 - Unstable",
    "2 - Near-neutral Unstable",        
    "3 - Near-neutral Stable",
    "4 - Stable",
    "5 - Very Stable",
    "6 - Extremely Stable",
])
plt.show()


# =============================================================================
# 7. Compute Shear Exponent Alpha for Each Timestamp
# =============================================================================

# Select two levels for shear calculation (example: lev=2 and lev=5)
for firstlev in [2]:
    secondlev = firstlev + 7
    z1, z2 = ds_vortex['lev'].values[firstlev], ds_vortex['lev'].values[secondlev]
    M1 = ds_vortex['M'].isel(lev=firstlev)  # Wind speed at z1
    M2 = ds_vortex['M'].isel(lev=secondlev)  # Wind speed at z2

    # Compute shear exponent alpha for each timestamp
    alpha = np.log(M2 / M1) / np.log(z2 / z1)

    # Add alpha as a variable to the dataset
    ds_vortex['alpha'] = alpha

    # Print summary statistics
    print(f"Shear exponent alpha first lev {firstlev}: {np.nanmean(alpha):.3f}")
    print(f"Shear exponent alpha (std) first lev {firstlev}: {np.nanstd(alpha):.3f}")
    #plot XY of alpha vs stability
    plt.figure(figsize=(8, 6))
    plt.scatter(ds_vortex['alpha'], ds_vortex['stability'], alpha=0.5)
    plt.xlabel('Shear Exponent Alpha')
    plt.ylabel('Stability Class')
    plt.title(f'Shear Exponent Alpha vs Stability Class [{firstlev}, {secondlev}] - {SITE.capitalize()}')
    plt.grid()
    plt.yticks(ticks=range(7), labels=[
        "0 - Very Unstable",
        "1 - Unstable",
        "2 - Near-neutral Unstable",        
        "3 - Near-neutral Stable",
        "4 - Stable",
        "5 - Very Stable",
        "6 - Extremely Stable",
    ])
    plt.show()

### 8. Fit Shear Exponent Alpha Using All Levels for Each Timestamp
# This provides a more robust estimate of alpha by using all available levels

import numpy as np
from scipy.stats import linregress

# Use .sizes instead of .dims for time dimension
levs = ds_vortex['lev'].values
levs2 = levs**2
alphas = []
r_values = []

for i in range(ds_vortex.sizes['time']):
    M_tmp = ds_vortex['M'].isel(time=i).values.flatten()
    # Fit M_tmp = a + b * lev²
    #print(M_tmp)
    #print(levs2)
    slope, intercept, r_value, p_value, std_err = linregress(levs2, M_tmp)
    #print(f"Fitted slope: {slope}, intercept: {intercept}")
    alphas.append(slope)
    r_values.append(r_value)

ds_vortex['alpha_all_levels'] = (('time'), np.array(alphas))
ds_vortex['r_value'] = (('time'), np.array(r_values))

print(f"Shear exponent alpha using all levels (mean): {np.nanmean(ds_vortex['alpha_all_levels'].values):.3f}")
print(f"Shear exponent alpha using all levels (std): {np.nanstd(ds_vortex['alpha_all_levels'].values):.3f}")

# Plot XY of alpha_all_levels vs stability
plt.figure(figsize=(8, 6))
plt.scatter(ds_vortex['alpha_all_levels'], ds_vortex['stability'], alpha=0.5)
plt.xlabel('Shear Exponent Alpha (All Levels)')
plt.ylabel('Stability Class')
plt.title(f'Shear Exponent Alpha (All Levels) vs Stability Class - {SITE.capitalize()}')
plt.grid()
plt.yticks(ticks=range(7), labels=[
    "0 - Very Unstable",
    "1 - Unstable",
    "2 - Near-neutral Unstable",        
    "3 - Near-neutral Stable",
    "4 - Stable",
    "5 - Very Stable",
    "6 - Extremely Stable",
])
plt.show()