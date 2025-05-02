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
from example_4_basic_plots_functions import *
import os

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

df = df_vortex_nc.merge(df_obs_nc, left_index=True, right_index=True)
# subtitute NA with 9999
df = df.dropna(how='any', axis=0)

# add vortex series with only concurrent period to obs.
df_with_na = pd.merge(df_vortex_nc, df_obs_nc, left_index=True, right_index=True, how='outer')
df_ST = df[['M_vortex_nc','Dir_vortex_nc']]
df_ST.columns = ['M_vortex_nc_ST', 'Dir_vortex_nc_ST']
df_with_na = pd.merge(df_with_na, df_ST, left_index=True, right_index=True, how='outer')

# check....
## checked, they are the same df_with_na = df_with_na.dropna(how='any', axis=0)

# =============================================================================
# 8. Use the functions for plotting
# =============================================================================
output_dir = "output"



# Use the functions to create the plots
xy_stats = plot_xy_comparison(
    df=df, 
    x_col='M_obs_nc', 
    y_col='M_vortex_nc',
    x_label='Measurement Wind Speed (m/s)',
    y_label='Vortex Wind Speed (m/s)',
    site=SITE,
    output_dir=output_dir,
    outlyer_threshold=4
)
output_dir = "output"
# Use the functions to create the plots
xy_stats = plot_xy_comparison(
    df=df, 
    x_col='M_obs_nc', 
    y_col='M_vortex_nc',
    x_label='Measurement Wind Speed (m/s)',
    y_label='Vortex Wind Speed (m/s)',
    site=SITE,
    output_dir=output_dir,
    outlyer_threshold=6
)

# Print regression statistics
print(f"\nRegression Statistics:")
print(f"Slope: {xy_stats['slope']:.4f}")
print(f"Intercept: {xy_stats['intercept']:.4f}")
print(f"R-squared: {xy_stats['r_squared']:.4f}")
print(f"p-value: {xy_stats['p_value']:.4e}")
print(f"Standard Error: {xy_stats['std_err']:.4f}")

# Create histogram
hist_stats = plot_histogram_comparison(
    df=df,
    cols=['M_obs_nc', 'M_vortex_nc'],
    labels=['Measurements', 'Vortex'],
    colors=['blue', 'red'],
    site=SITE,
    output_dir=output_dir,
    bins=25,
    alpha=0.6
)

# Create histogram
hist_stats = plot_histogram_comparison(
    df=df,
    cols=['Dir_obs_nc', 'Dir_vortex_nc'],
    labels=['Measurements', 'Vortex'],
    colors=['blue', 'red'],
    site=SITE+" Dir",
    output_dir=output_dir,
    bins=12,
    alpha=0.6
)



# =============================================================================
# 9. Plot Annual and Daily Cycles
# =============================================================================

# Plot annual cycle for wind speed
annual_stats_M = plot_annual_means(
    df=df,
    cols=['M_obs_nc', 'M_vortex_nc'],
    labels=['Measurements', 'Vortex'],
    colors=['blue', 'red'],
    site=SITE,
    output_dir=output_dir
)

# Plot daily cycle for wind speed
daily_stats_M = plot_daily_cycle(
    df=df,
    cols=['M_obs_nc', 'M_vortex_nc'],
    labels=['Measurements', 'Vortex'],
    colors=['blue', 'red'],
    site=SITE,
    output_dir=output_dir
)

# =============================================================================
# 10. Plot Yearly Means
# =============================================================================

# Plot yearly means for wind speed
yearly_stats_M = plot_yearly_means(
    df=df,
    cols=['M_obs_nc', 'M_vortex_nc'],
    labels=['Measurements', 'Vortex'],
    colors=['blue', 'red'],
    site=SITE,
    output_dir=output_dir
)



# now I want to compare long term histogram using full ds_vortex_nc compared to the df period
hist_stats = plot_yearly_means(
    df = df_with_na,
    cols = ['M_vortex_nc','M_obs_nc','M_vortex_nc_ST'],
    labels=['Vortex LT','OBS','Vortex ST'],
    colors=['green','blue','red'],
    site =SITE,
    output_dir=output_dir
)

# describe to check number of NaNs in years 2010 to 2014
# check the number of NaNs in the years 2010 to 2014
print(df_with_na.loc['2010-01-01':'2014-12-31'].describe())
print(df_with_na.loc['2010-01-01':'2014-12-31'].isna().sum())












