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

# =============================================================================
# 7. Create XY Plot with Linear Regression
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Create figure and axis
plt.figure(figsize=(8, 8))

# Scatter plot of M observations vs M vortex
plt.scatter(df['M_obs_nc'], df['M_vortex_nc'], alpha=0.5, color='blue')

# Calculate linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(df['M_obs_nc'], df['M_vortex_nc'])
r_squared = r_value**2

# Create regression line
x = np.linspace(df['M_obs_nc'].min(), df['M_obs_nc'].max(), 100)
y = slope * x + intercept
plt.plot(x, y, 'r-', label=f'y = {slope:.3f}x + {intercept:.3f}')

# Add identity line (perfect agreement)
plt.plot([0, max(df['M_obs_nc'].max(), df['M_vortex_nc'].max())], 
         [0, max(df['M_obs_nc'].max(), df['M_vortex_nc'].max())], 
         'k--', alpha=0.3, label='1:1')

# Add annotations with regression statistics
plt.annotate(f'$R^2$ = {r_squared:.3f}', 
             xy=(0.05, 0.95), xycoords='axes fraction', 
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# Add labels and title
plt.xlabel('Measurement Wind Speed (m/s)', fontsize=12)
plt.ylabel('Vortex Wind Speed (m/s)', fontsize=12)
plt.title(f'Comparison of Measured vs Vortex Wind Speed at {SITE.capitalize()}', fontsize=14)

# Add grid and legend
plt.grid(True, alpha=0.3)
plt.legend()

# Equal aspect ratio
plt.axis('equal')
plt.tight_layout()

# Save the figure
output_dir = os.path.join(pwd, '../output')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, f'{SITE}_wind_speed_comparison.png'), dpi=300)

# Show the plot
plt.show()

# Print regression statistics
print(f"\nRegression Statistics:")
print(f"Slope: {slope:.4f}")
print(f"Intercept: {intercept:.4f}")
print(f"R-squared: {r_squared:.4f}")
print(f"p-value: {p_value:.4e}")
print(f"Standard Error: {std_err:.4f}")

# =============================================================================
# 8. Create Histogram of Wind Speed
# =============================================================================

plt.figure(figsize=(10, 6))

# Define number of bins and range
bins = np.linspace(0, max(df['M_obs_nc'].max(), df['M_vortex_nc'].max()) + 1, 25)

# Plot histograms with transparency
plt.hist(df['M_obs_nc'], bins=bins, alpha=0.6, label='Measurements', color='blue', edgecolor='black')
plt.hist(df['M_vortex_nc'], bins=bins, alpha=0.6, label='Vortex', color='red', edgecolor='black')

# Add labels and title
plt.xlabel('Wind Speed (m/s)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Wind Speed Distribution Comparison at {SITE.capitalize()}', fontsize=14)

# Add grid and legend
plt.grid(True, alpha=0.3)
plt.legend()

# Calculate and display some statistics
obs_mean = df['M_obs_nc'].mean()
vortex_mean = df['M_vortex_nc'].mean()

plt.axvline(obs_mean, color='blue', linestyle='dashed', linewidth=1.5)
plt.axvline(vortex_mean, color='red', linestyle='dashed', linewidth=1.5)

# Add annotations for mean values
plt.annotate(f'Obs Mean: {obs_mean:.2f} m/s', 
             xy=(obs_mean, plt.ylim()[1] * 0.9),
             xytext=(obs_mean + 0.5, plt.ylim()[1] * 0.9),
             arrowprops=dict(arrowstyle='->', color='blue'),
             color='blue')

plt.annotate(f'Vortex Mean: {vortex_mean:.2f} m/s', 
             xy=(vortex_mean, plt.ylim()[1] * 0.8),
             xytext=(vortex_mean + 0.5, plt.ylim()[1] * 0.8),
             arrowprops=dict(arrowstyle='->', color='red'),
             color='red')

plt.tight_layout()

# Save the figure
plt.savefig(os.path.join(output_dir, f'{SITE}_wind_speed_histogram.png'), dpi=300)

# Show the plot
plt.show()


