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

vortex_txt = os.path.join(base_path, f'{SITE}/vortex/SERIE/vortex.serie.era5.utc0.100m.txt')
measurements_txt = os.path.join(base_path, f'{SITE}/measurements/obs.txt')

print()
print('#'*26, 'Vortex F.d.C. 2025', '#'*26)
print()


# Read Text Series
ds_vortex_txt = read_vortex_serie(vortex_txt)
df_obs_txt = read_vortex_obs_to_dataframe(measurements_txt)[['M', 'Dir']]
ds_obs_txt = convert_to_xarray(df_obs_txt)[['M', 'Dir']]
#ds_vortex_nc = ds_vortex_nc.rename_vars({'D': 'Dir'})
ds_vortex_txt = ds_vortex_txt[['M', 'Dir']].squeeze().reset_coords(drop=True)
# convert ds_obs_txt to hourly
ds_obs_txt = ds_obs_txt.resample(time='1H').mean()
# =============================================================================
# 6. Convert all to DataFrame, Rename and Merge
# =============================================================================
df_obs_txt = ds_obs_txt.to_dataframe()
df_vortex_txt = ds_vortex_txt.to_dataframe()

df_obs_txt.columns = ['M_obs_txt', 'Dir_obs_txt']
df_vortex_txt.columns = ['M_vortex_txt', 'Dir_vortex_txt']

df_concurrent = df_obs_txt.merge(df_vortex_txt, left_index=True, right_index=True).dropna()
df_all = df_obs_txt.merge(df_vortex_txt, left_index=True, right_index=True, how='outer')
print(df_concurrent.describe())

print(df_all.describe())


# =============================================================================
# 8. Use the functions for plotting
# =============================================================================
output_dir = "output"
# Use the functions to create the plots
xy_stats = plot_xy_comparison(
    df=df_concurrent, 
    x_col='M_vortex_txt', 
    y_col='M_obs_txt',
    x_label='Measurement Wind Speed (m/s)',
    y_label='Vortex Wind Speed (m/s)',
    site=SITE,
    output_dir=output_dir,
    outlyer_threshold=4
)

# Print regression statistics
print(f"\nRegression Statistics:")
print(f"Slope: {xy_stats['slope']:.4f}")
print(f"Intercept: {xy_stats['intercept']:.4f}")
print(f"R-squared: {xy_stats['r_squared']:.4f}")
print(f"p-value: {xy_stats['p_value']:.4e}")
print(f"Standard Error: {xy_stats['std_err']:.4f}")

## create a new column with the MCP
## Ymcp ) = {xy_stats['intercept'] + {xy_stats['slope']*df_all['M_vortex_txt']}
df_all['Ymcp'] = xy_stats['intercept'] + xy_stats['slope']*df_all['M_vortex_txt']

# concurrent stats

print(df_all.dropna().describe())

# Define 8 directional sectors (0-45, 45-90, etc.)
sector_bounds = list(range(0, 361, 45))
sector_labels = [f"{sector_bounds[i]}-{sector_bounds[i+1]}" for i in range(len(sector_bounds)-1)]

# Add a column for the wind direction sector
df_concurrent['dir_sector'] = pd.cut(df_concurrent['Dir_vortex_txt'], 
                                    bins=sector_bounds,
                                    labels=sector_labels,
                                    include_lowest=True,
                                    right=False)
df_all['dir_sector'] = pd.cut(df_all['Dir_vortex_txt'], 
                             bins=sector_bounds,
                             labels=sector_labels,
                             include_lowest=True,
                             right=False)

# Initialize results dictionary to store regression parameters
sector_regressions = {}

# Perform regression for each sector
for sector in sector_labels:
    sector_data = df_concurrent[df_concurrent['dir_sector'] == sector]
    
    # Skip sectors with too few data points
    if len(sector_data) < 5:
        print(f"Warning: Sector {sector} has insufficient data points. Using global regression.")
        sector_regressions[sector] = {'slope': xy_stats['slope'], 'intercept': xy_stats['intercept']}
        continue
    
    # Perform linear regression for this sector
    x = sector_data['M_vortex_txt']
    y = sector_data['M_obs_txt']
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    sector_regressions[sector] = {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_value**2,
        'p_value': p_value,
        'std_err': std_err
    }
    
    print(f"\nSector {sector} Regression Statistics:")
    print(f"Slope: {slope:.4f}, Intercept: {intercept:.4f}, R-squared: {r_value**2:.4f}")


print(df_all.head())
print(df_concurrent.describe())

# Apply sector-specific regression to create the new MCP column
df_all['Ymcp_sectorial'] = None

for sector in sector_labels:
    mask = df_all['dir_sector'] == sector
    if mask.any():  # Only proceed if there's data in this sector
        slope = sector_regressions[sector]['slope']
        intercept = sector_regressions[sector]['intercept']
        df_all.loc[mask, 'Ymcp_sectorial'] = intercept + slope * df_all.loc[mask, 'M_vortex_txt']
    else:
        print(f"Warning: Sector {sector} has no data points. Skipping.")

# Print comparison of overall and sectorial MCP methods
print("\nComparison of MCP methods (for concurrent data):")
print(df_all.describe())
print(df_concurrent.describe())
exit(0)
stats_comparison = df_concurrent[['M_obs_txt', 'Ymcp', 'Ymcp_sectorial']].describe()
print(stats_comparison)

# Plot comparison of the two MCP methods
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_concurrent['M_obs_txt'], df_concurrent['Ymcp'], alpha=0.4, label='Global MCP')
ax.scatter(df_concurrent['M_obs_txt'], df_concurrent['Ymcp_sectorial'], alpha=0.4, label='Sectorial MCP')
ax.plot([0, df_concurrent['M_obs_txt'].max()], [0, df_concurrent['M_obs_txt'].max()], 'k--', label='1:1 Line')
ax.set_xlabel('Observed Wind Speed (m/s)')
ax.set_ylabel('MCP Wind Speed (m/s)')
ax.set_title(f'{SITE} - Comparison of MCP Methods')
ax.legend()
plt.savefig(os.path.join(output_dir, f'{SITE}_mcp_comparison.png'), bbox_inches='tight')

exit(0)


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












