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
from example_3_merge_functions import *
from example_4_basic_plots_functions import *
from scipy.stats import wasserstein_distance
import os
from example_5_MeasureCorrelatePredict_functions import plot_histogram_comparison_lines


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
# 3. Convert all to DataFrame, Rename and Merge
# =============================================================================
df_obs_txt = ds_obs_txt.to_dataframe()
df_vortex_txt = ds_vortex_txt.to_dataframe()

df_obs_txt.columns = ['M_obs_txt', 'Dir_obs_txt']
df_vortex_txt.columns = ['M_vortex_txt', 'Dir_vortex_txt']

df_concurrent = df_obs_txt.merge(df_vortex_txt, left_index=True, right_index=True).dropna()
df_all = df_obs_txt.merge(df_vortex_txt, left_index=True, right_index=True, how='outer')

# SAVED DATASETS, CONCURRENT AND ALL PERIODS
#print(df_concurrent.describe())
#print(df_all.describe())


# =============================================================================
# 4. Use the functions for regression
# =============================================================================
output_dir = "output"
# Use the functions to compute metrics
xy_stats = plot_xy_comparison(
    df=df_concurrent, 
    x_col='M_vortex_txt', 
    y_col='M_obs_txt',
    x_label='Measurement Wind Speed (m/s)',
    y_label='Vortex Wind Speed (m/s)',
    site=SITE,
    output_dir=output_dir,
    outlyer_threshold=4,
    show=False
)

# Print regression statistics
print(f"\nRegression Statistics:")
print(f"Slope: {xy_stats['slope']:.4f}")
print(f"Intercept: {xy_stats['intercept']:.4f}")
print(f"R-squared: {xy_stats['r_squared']:.4f}")
print(f"p-value: {xy_stats['p_value']:.4e}")
print(f"Standard Error: {xy_stats['std_err']:.4f}")

# =============================================================================
# 5. Compute the MCP
# =============================================================================

## create a new column with the MCP
## Ymcp ) = {xy_stats['intercept'] + {xy_stats['slope']*df_all['M_vortex_txt']}
df_all['Ymcp'] = xy_stats['intercept'] + xy_stats['slope']*df_all['M_vortex_txt']

# concurrent stats

#print(df_all.dropna().describe())

hist_stats = plot_histogram_comparison(
    df=df_all.dropna(),
    cols=['M_obs_txt', 'Ymcp'],
    labels=['Measurements', 'MCP'],
    colors=['blue',  'orange'],
    site=SITE,
    output_dir=output_dir,
    bins=25,
    alpha=0.3
)

# =============================================================================
# 6. Compute the Sectorial MCP
# =============================================================================

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

# Convert Ymcp_sectorial column to float64
df_all['Ymcp_sectorial'] = df_all['Ymcp_sectorial'].astype('float64')
df = df_all.copy().dropna()[['M_obs_txt', 'M_vortex_txt','Ymcp', 'Ymcp_sectorial']]

# Create histogram
hist_stats = plot_histogram_comparison(
    df=df,
    cols=['M_obs_txt',  'Ymcp_sectorial'],
    labels=['Measurements', 'Sectorial MCP'],
    colors=['blue', 'red'],
    site=SITE,
    output_dir=output_dir,
    bins=25,
    alpha=0.3
)
# =============================================================================
# 7. Read remodeling
# =============================================================================

# we now introduce a different method, Vortex Remodeling

file_remodeling_txt =  os.path.join(base_path, f'{SITE}/vortex/SERIE/vortex.remodeling.utc0.100m.txt')
ds_remodeling_txt = read_remodeling_serie(file_remodeling_txt)
df_remodeling_txt = ds_remodeling_txt.to_dataframe().rename(columns={'M': 'M_remodeling_txt'})
df =df.merge(df_remodeling_txt[['M_remodeling_txt']], left_index=True, right_index=True, how='outer').dropna()

hist_stats = plot_histogram_comparison(
    df=df,
    cols=['M_obs_txt', 'M_remodeling_txt'],
    labels=['Measurements', 'Remodeling'],
    colors=['blue',  'orange'],
    site=SITE,
    output_dir=output_dir,
    bins=25,
    alpha=0.3
)

# =============================================================================
# 8. Compare statsistics
# =============================================================================
# Import required library for Earth Mover's Distance calculation

# Calculate Earth Mover's Distance (Wasserstein distance) for each method

emd_results = {}
for col in ['Ymcp', 'Ymcp_sectorial', 'M_remodeling_txt']:
    # Calculate EMD between the method and observations
    emd = wasserstein_distance(df['M_obs_txt'], df[col])
    emd_results[col] = emd




# Calculate mean absolute error and root mean squared error for each prediction method compared to observations
print("\nError Metrics (compared to M_obs_txt):")
print("=" * 80)
for col in ['Ymcp', 'Ymcp_sectorial', 'M_remodeling_txt']:
    mae = (df[col] - df['M_obs_txt']).abs().mean()
    rmse = ((df[col] - df['M_obs_txt']) ** 2).mean() ** 0.5
    bias = (df[col] - df['M_obs_txt']).mean()
    print(f"{col}:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} m/s")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f} m/s")
    print(f"  Bias: {bias:.4f} m/s")
    print(f"  Histogram error(EMD): {emd_results[col]:.4f}")

# =============================================================================
# 9. Liner histogram for histogram comparison
# =============================================================================


# Example usage for the new function
hist_line_stats = plot_histogram_comparison_lines(
    df=df,
    cols=['M_obs_txt', 'Ymcp', 'Ymcp_sectorial', 'M_remodeling_txt'],
    labels=['Measurements', 'MCP', 'Sectorial MCP', 'Remodeling'],
    colors=['blue', 'orange', 'red', 'green'],
    site=SITE,
    output_dir=output_dir,
    bins=50
)
exit()














