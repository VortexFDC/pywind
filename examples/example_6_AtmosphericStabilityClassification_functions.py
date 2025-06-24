'''
Atmospheric Stability Analysis Pipeline
--------------------------------------

This module implements a comprehensive workflow for analyzing atmospheric stability across multiple sites:

1. Data Acquisition and Processing:
    - Reads meteorological data from NetCDF files for multiple sites
    - Handles both observational measurements and model outputs (series/times formats)
    - Supports missing data interpolation across vertical levels

2. Stability Calculation Methods:
    - Implements Richardson number (Ri) based stability classification
    - Categorizes stability into 7 classes: Very Unstable → Very Stable
    - Automatically identifies optimal vertical levels (z1, z2) for gradient calculations

3. Analysis Pipeline:
    - Extracts and processes wind components (U, V) and temperature (T)
    - Calculates derived variables: wind speed (M), direction (Dir), Richardson number (RI)
    - Performs statistical frequency analysis across temporal and meteorological dimensions

4. Visualization:
    - Generates standardized stability frequency plots showing distributions by:
      * Diurnal cycle (hourly)
      * Annual cycle (monthly)
      * Wind speed dependencies
      * Wind direction dependencies
    - Supports comparative analysis between observational and model data

The implementation uses standard meteorological gradient methods with configurable thresholds
for stability classification based on established boundary layer meteorology principles.

5. Inputs and Outputs:
    - Inputs: NetCDF files containing meteorological variables (U, V, T, M, Dir) from validate_obs_stability.py
    - Outputs: Plots saved as PNG files in specified output directories
    - Returns processed datasets with stability classifications and attributes
    - Provides functions to build datasets for stability analysis and plot results
    - Includes error handling for missing variables and invalid levels


Dependencies: xarray, numpy, pandas, matplotlib, vortexpy, pyvtx
'''


import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib_vortexpy_vars import (
    find_zonal_wind, 
    find_meridional_wind, 
    find_wind_speed, 
    find_direction,
    find_richardson,
    find_stability_class,
    find_stability,
    find_temperature_celsius,
    atmospheric_stability_classes,
    atmospheric_stability_edges_rmol,
)
from lib_vortexpy_vdatas import (
    add_attrs_coords, 
    add_attrs_vars,
    apply_general,
    vtx_attributes_vars as attributes_vars
)


def find_stability_ri(vs):
    """
    Calculate the atmospheric stability index (from 0 to 6). Given an RI
    time series, the atmospheric stability index is returned as a vArray.
    Parameters
    ----------
    vs: vSet
        Time series. 'RMOL' must be contained in the vSet.

    Returns
    -------
    Stability: vArray
        Atmospheric stability time series.

        Atmospheric stability classes based on Richardson number (RI) thresholds.
        The classes are defined as follows:
        - Very Unstable: RI < -0.05
        - Unstable: -0.05 <= RI < 0
        - Near-neutral Unstable: 0 <= RI < 0.01
        - Neutral: 0.01 <= RI < 0.05
        - Near-neutral Stable: 0.05 <= RI < 0.25
        - Stable: 0.25 <= RI < 1.0
        - Very Stable: RI >= 1.0
    """
    atmospheric_stability_edges_ri = [-0.05, 0, 0.01, 0.05, 0.25, 1.0]

    if 'stability' in vs:
        stability = vs['stability']
    else:
        try:
            ri = vs['RI']
        except KeyError:
            raise ValueError('Cannot obtain stability (no RI)')

        # The 5 RMOL values that define the 7 categories of
        # atmospheric stability
        stability = apply_general(np.digitize, ri,
                                  bins=atmospheric_stability_edges_ri)

        stability = stability.rename('stability')
        stability.attrs = attributes_vars['stability']

    return stability


def plot_stability_frequency(ds, site, output_dir, plot_types='all', **kwargs):
    """
    Plot atmospheric stability frequency analysis for a given site and vertical levels.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the meteorological variables.
    site : str
        Name of the site being analyzed, used in plot titles and filenames.
    output_dir : str
        Directory where to save the output plot.
    ilev : int, default=5
        Index of the upper vertical level to use for stability calculations.
    dlev : int, default=2
        Difference in index between upper and lower levels for calculating gradients.
    """
    
    # Define stability classes and colors
    atmospheric_stability_classes = ['Very Unstable', 'Unstable', 'Near-neutral Unstable', 'Neutral', 'Near-neutral Stable', 'Stable', 'Very Stable']

    stability_classes = pd.Categorical(atmospheric_stability_classes, 
                                       categories=['Very Unstable', 'Unstable', 'Near-neutral Unstable', 
                                                  'Neutral', 'Near-neutral Stable', 'Stable', 'Very Stable'], 
                                       ordered=True)
    
    colors = ['#d73027', '#fc8d59', '#fee090', '#91cf60', '#e0f3f8', '#4575b4', '#313695']
    colors = pd.Categorical(colors, 
                            categories=colors, 
                            ordered=True)

    # Check if M and Dir variables are present
    if 'M' not in ds or 'Dir' not in ds:
        raise ValueError("The dataset does not contain the 'M' or 'Dir' variables. Please check the dataset.")
    # Check if the dataset has RI variable
    if 'RI' in ds:
        var = 'RI'
        # Check if stability and stabilityClass variables are present
        if 'stability' not in ds or 'stabilityClass' not in ds:
            ds['stability'] = find_stability_ri(ds)
            ds['stabilityClass'] = find_stability_class(ds)

        # Convert dataset variables to a DataFrame
        df = pd.DataFrame({
            'time': ds.time.values,
            'M': ds['M'].values.squeeze(),
            'Dir': ds['Dir'].values.squeeze(),
            'RI': ds['RI'].values.squeeze(),
            'stability': ds['stability'].values.squeeze(),
            'stabilityClass': ds['stabilityClass'].values.squeeze()
        })
    elif 'RMOL' in ds:
        var = 'RMOL'
        # Check if stability and stabilityClass variables are present
        if 'stability' not in ds or 'stabilityClass' not in ds:
            ds['stability'] = find_stability(ds)
            ds['stabilityClass'] = find_stability_class(ds)
        # Convert dataset variables to a DataFrame
        df = pd.DataFrame({
            'time': ds.time.values,
            'M': ds['M'].values.squeeze(),
            'Dir': ds['Dir'].values.squeeze(),
            'RMOL': ds['RMOL'].values.squeeze(),
            'stability': ds['stability'].values.squeeze(),
            'stabilityClass': ds['stabilityClass'].values.squeeze()
        })
    else:
        raise ValueError("The dataset does not contain the 'RI' or 'RMOL' variables. Please check the dataset.")
    # Add month and hour columns for grouping
    df['month'] = pd.Categorical(
        df['time'].dt.strftime('%B'),
        categories=['January', 'February', 'March', 'April', 'May', 'June', 
                   'July', 'August', 'September', 'October', 'November', 'December'],
        ordered=True
    )
    df['hour'] = df['time'].dt.hour

    # Prepare grouped dataframes
    df['stabilityClass'] = df['stabilityClass'].astype(str)
    df['stabilityClass'] = pd.Categorical(df['stabilityClass'],
                                          categories=atmospheric_stability_classes,
                                          ordered=True)
    
    # Group by hour
    df_hour = df.groupby(['hour', 'stabilityClass'], observed=True).size().unstack().fillna(0)
    df_hour = df_hour.div(df_hour.sum(axis=1), axis=0) * 100
    
    # Group by month
    df_month = df.groupby(['month', 'stabilityClass'], observed=True).size().unstack().fillna(0)
    df_month = df_month.reindex(['January', 'February', 'March', 'April', 'May', 'June', 
                                'July', 'August', 'September', 'October', 'November', 'December'])
    df_month = df_month.div(df_month.sum(axis=1), axis=0) * 100
    
    # Binar la velocitat del vent (M) en intervals de 2 m/s de 0 a 24
    bins_M = range(0, 26, 2)
    df['M_bin'] = pd.cut(df['M'], bins=bins_M, right=False, labels=bins_M[:-1])

    df_M = df.groupby(['M_bin', 'stabilityClass'], observed=True).size().unstack().fillna(0)
    df_M = df_M.reindex(bins_M[:-1])

    # Binar la direcció del vent (Dir) en intervals de 30 graus de 0 a 360
    bins_Dir = range(0, 361, 30)
    df['Dir_bin'] = pd.cut(df['Dir'], bins=bins_Dir, right=False, labels=bins_Dir[:-1])

    df_Dir = df.groupby(['Dir_bin', 'stabilityClass'], observed=True).size().unstack().fillna(0)
    df_Dir = df_Dir.reindex(bins_Dir[:-1])
    
    # Get title from kwargs or default to empty string
    title = kwargs.get('title', '')
    if title:
        title = f" - {title}"
    else:
        title = ''
    
    # Plotting
    # Get which plots to generate
    if plot_types == 'all':
        plot_types = ['hour', 'month', 'M', 'Dir']
    elif isinstance(plot_types, str):
        plot_types = [plot_types]
    
    # Determine number of subplots needed
    n_plots = len(plot_types)
    if n_plots == 1:
        fig, ax = plt.subplots(figsize=(10, 6))
        axes = np.array([[ax]])
    elif n_plots == 2:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes = axes.reshape(1, 2)
    elif n_plots <= 4:
        rows = (n_plots + 1) // 2
        cols = min(n_plots, 2)
        fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 6*rows))
        if n_plots == 3:  # If 3 plots, we'll have a 2x2 grid with one empty
            axes = axes.reshape(2, 2)
    
    # Add a general title to the entire figure
    plt.suptitle(f'Atmospheric Stability Analysis for {var}{title}', 
                 fontsize=18, y=0.98)
    
    # Plot counter
    plot_idx = 0
    
    # Plot based on selected types
    if 'hour' in plot_types:
        row, col = plot_idx // 2, plot_idx % 2
        df_hour.plot(kind='bar', stacked=True, color=colors, ax=axes[row, col], width=0.8, legend=False)
        axes[row, col].set_title('Stability Frequency by Hour')
        axes[row, col].set_xlabel('Hour')
        axes[row, col].set_ylabel('Frequency (%)')
        plot_idx += 1
    
    if 'month' in plot_types:
        row, col = plot_idx // 2, plot_idx % 2
        df_month.plot(kind='bar', stacked=True, color=colors, ax=axes[row, col], width=0.8, legend=False)
        axes[row, col].set_title('Stability Frequency by Month')
        axes[row, col].set_xlabel('Month')
        axes[row, col].set_ylabel('Frequency (%)')
        plot_idx += 1
    
    if 'M' in plot_types:
        row, col = plot_idx // 2, plot_idx % 2
        df_M.plot(kind='bar', stacked=True, color=colors, ax=axes[row, col], width=0.8, legend=False)
        axes[row, col].set_title('Stability Frequency by Wind Speed')
        axes[row, col].set_xlabel('Mean Wind Speed (m/s)')
        axes[row, col].set_ylabel('Frequency (%)')
        plot_idx += 1
    
    if 'Dir' in plot_types:
        row, col = plot_idx // 2, plot_idx % 2
        df_Dir.plot(kind='bar', stacked=True, color=colors, ax=axes[row, col], width=0.8, legend=True)
        axes[row, col].set_title('Stability Frequency by Wind Direction')
        axes[row, col].set_xlabel('Wind Direction (°)')
        axes[row, col].set_ylabel('Frequency (%)')
        
        # Add legend to the last plot
        axes[row, col].legend(stability_classes, title='Stability Classification', 
                             bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Remove any unused subplots
    if n_plots == 3:
        fig.delaxes(axes[1, 1])
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the figure title
    
    # Create filename based on plot types
    if plot_types == ['all'] or len(plot_types) > 1:
        filename = f'{site}_{var}_stability_frequency.png'
    else:
        filename = f'{site}_{var}_stability_{plot_types[0]}.png'
    
    plt.savefig(f'{output_dir}/{filename}', dpi=300)
    plt.show()
    
    # Check if the saved file exists
    if os.path.exists(f'{output_dir}/{site}_{var}_stability_frequency.png'):
        print(f"Plot saved successfully to {output_dir}/{site}_{var}_stability_frequency.png")
    else:
        print("Error saving the plot.")


def build_ds_stability(ds, z1, z2):
    """
    Build a dataset with stability variables at specified levels z1 and z2.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing meteorological variables.
    z1 : float
        Lower level for stability analysis.
    z2 : float
        Upper level for stability analysis.
    
    Returns
    -------
    xarray.Dataset
        Dataset with stability variables at levels z1 and z2.
    """
    # Check U i V variables
    if 'U' not in ds or 'V' not in ds:
        print("U and V variables are not present in the dataset. Calculating U and V.")
        ds['U'] = find_zonal_wind(ds)
        ds['V'] = find_meridional_wind(ds)
    # Check if z1 and z2 are provided
    if z1 is None or z2 is None:
        print("z1 and z2 must be provided to build the stability dataset.")
        return None

    # Construïr dataset per a U, V i T als lev z1 i z2
    ds_stability = {}

    # Check if z1 and z2 are levels in the dataset
    if 'lev' not in ds['U'].dims or 'lev' not in ds['V'].dims or 'lev' not in ds['T'].dims:
        print("The dataset does not contain the 'lev' dimension in U, V or T variables. Cannot build stability dataset.")
        return None
    # Average z1 z2
    z12 = (z1 + z2) / 2
    print(f"Building stability dataset for levels z1: {z1}, z2: {z2}")
    # Check if z1 and z2 are valid levels
    if z1 not in ds['U'].lev.values or z2 not in ds['U'].lev.values or z12 not in ds['U'].lev.values:
        print(f"z1 ({z1}) or z2 ({z2}) are not valid levels in the dataset.")
        # Create a list of all needed levels
        needed_levels = sorted(list(set(list(ds['U'].lev.values) + [z1, z2])))
        
        # Create new dataset with expanded levels
        ds_expanded = ds.copy()
        
        # Expand the lev coordinate for all variables
        for var in ['U', 'V', 'T']:
            # Create a new DataArray with the expanded levels
            expanded_var = ds[var].reindex(lev=needed_levels)
            # Interpolate to fill the newly added levels
            expanded_var = expanded_var.interpolate_na(dim='lev')
            # Assign to the new dataset
            ds_expanded[var] = expanded_var
            
        # Use the expanded dataset for further processing
        ds = ds_expanded
        
        print(f"Added levels z1 ({z1}) and z2 ({z2}) to the dataset and interpolated values.")
    else:
        print(f"z1 ({z1}) and z2 ({z2}) are valid levels in the dataset. Proceeding to build stability dataset.")
    
    # Select the closest available levels to z1 and z2
    ds_stability['U'] = ds['U'].sel(lev=[z1, z2], method='nearest')
    ds_stability['V'] = ds['V'].sel(lev=[z1, z2], method='nearest')
    ds_stability['T'] = ds['T'].sel(lev=[z1, z2], method='nearest')
    print("Dataset de estabilitat construït amb les variables U, V i T.")
        
    # Analitzar nivells per U, V i T
    for var in ['U', 'V', 'T']:
        if var in ds_stability:
            # Elimina els nivells on totes les dades són NaN
            ds_var_clean = ds_stability[var].dropna(dim='lev', how='all')
            levs = ds_var_clean.lev.values
            print(f"Levs amb dades per {var}: {levs}")
        else:
            print(f"La variable {var} no és al dataset.")
    # Define dimensions for the stability dataset
    ds_stability = xr.Dataset(ds_stability)
    ds_stability = ds_stability.assign_coords(lev=[z1, z2])
    ds_stability = add_attrs_coords(ds_stability)
    ds_stability = add_attrs_vars(ds_stability)
    print("Dataset de estabilitat completat amb les coordenades i atributs.")

    
    if ds_stability is  None:
        print("Dataset de estabilitat és buit. Comprova les variables U, V i T.")
        return None

    ds_stability['M'] = find_wind_speed(ds_stability)
    ds_stability['Dir'] = find_direction(ds_stability)
    print("Variables M i Dir afegides al dataset de estabilitat.")
    ds_stability = add_attrs_vars(ds_stability)
    print("Atributs de les variables afegits al dataset de estabilitat.")
    ds_stability

    # Calculate the Richardson number
    try:
        richardson_number = find_richardson(ds_stability)

    except Exception as e:
        print(f'Error calculating Richardson number: {e}')
    
    # Get M and Dir at levels close to richardson_number['lev'].values from ds
    lev = richardson_number['lev'].values
    print(f"Richardson Number lev: {lev}")

    # Find closest available levels in the dataset
    available_levs_M = ds['M'].lev.values
    available_levs_Dir = ds['Dir'].lev.values

    # Find the closest available levels for each variable
    closest_lev_M = available_levs_M[np.abs(available_levs_M - lev[0]).argmin()]
    closest_lev_Dir = available_levs_Dir[np.abs(available_levs_Dir - lev[0]).argmin()]

    print(f"Closest available level for M: {closest_lev_M}")
    print(f"Closest available level for Dir: {closest_lev_Dir}")

    if closest_lev_M == closest_lev_Dir:
        print("Both variables are at the same level.")
    else:
        print("Variables are at different levels.")

    # Extract data at the closest levels
    print(f"Extracting data for M and Dir at level {closest_lev_M} and {closest_lev_Dir} respectively.")

    if lev - closest_lev_M > 20 or lev - closest_lev_Dir > 20:
        print("Warning: The selected levels are significantly different from the Richardson number level.")
    # Extract data at closest levels
    df_M = ds['M'].sel(lev=closest_lev_M)
    df_Dir = ds['Dir'].sel(lev=closest_lev_Dir)

    # Build dataframe for plotting with df_M, df_Dir, and richardson_number

    # Extract data correctly by reducing dimensions
    # Create xarray dataset directly from the data variables
    ds_plot = xr.Dataset(
        data_vars={
            'M': (['time'], df_M.squeeze(dim=['lat', 'lon']).values),
            'Dir': (['time'], df_Dir.squeeze(dim=['lat', 'lon']).values),
            'RI': (['time'], richardson_number.squeeze(dim=['lat', 'lon', 'lev']).values)
        },
        coords={
            'time': df_M.time.values,
            'lev': lev
        }
    )
    # Calculate the atmospheric stability index
    stability = find_stability_ri(ds_plot)
    # Add stability to the dataset
    ds_plot['stability'] = stability

    # Stability classes
    ds_plot['stabilityClass'] = find_stability_class(ds_plot)
    # Display the stability dataset
    print(ds_plot)
    print("Dataset de estabilitat completat amb les coordenades i atributs.")

    return ds_plot


def levels_with_T(ds):
    """
    Find the levels with temperature (T) in the dataset.
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing meteorological variables.
    
    Returns
    -------
    tuple
        Levels z1 and z2 for stability analysis, or (None, None) if not found.
    """

    # Analitzar nivells per M, Dir i T
    df = {}
    for var in ['M', 'Dir', 'T']:
        if var in ds:
            # Elimina els nivells on totes les dades són NaN
            ds_var_clean = ds[var].dropna(dim='lev', how='all')
            levs = ds_var_clean.lev.values
            print(f"Levs amb dades per {var}: {levs}")
            # Guarda els nivells originals en un diccionari nou
            df[var + '_levs'] = levs
        else:
            print(f"La variable {var} no és al dataset.")

    # Troba les variables de vent zonal i meridional
    ds['U'] = find_zonal_wind(ds)
    ds['V'] = find_meridional_wind(ds)
    print("Variables U i V afegides al dataset.")
    # Interpolate_na for M, Dir and T
    for var in ['U', 'V', 'T']:
        if var in ds:
            # Interpolació de NaN per a la variable
            ds[var] = ds[var].interpolate_na(dim='lev')
            print(f"Interpolació de NaN per {var} completada.")
        else:
            print(f"La variable {var} no és al dataset.")

    # Analitzar nivells per M, Dir i T
    for var in ['M', 'U', 'V', 'T']:
        if var in ds:
            # Elimina els nivells on totes les dades són NaN
            ds_var_clean = ds[var].dropna(dim='lev', how='all')
            levs = ds_var_clean.lev.values
            print(f"Levs amb dades per {var}: {levs}")
        else:
            print(f"La variable {var} no és al dataset.")

    # Buscar z1 i z2
    min_lev_T = ds['T'].dropna(dim='lev', how='all').lev.min().item()
    print(f"Min lev T: {min_lev_T}")
    min_lev_M = ds['M'].dropna(dim='lev', how='all').lev.min().item()
    print(f"Min lev M: {min_lev_M}")
    if min_lev_T == min_lev_M:
        z1 = min_lev_T
    else:
        z1 = max(min_lev_T, min_lev_M)
    print(f"z1: {z1}")
    max_lev_T = ds['T'].dropna(dim='lev', how='all').lev.max().item()
    print(f"Max lev T: {max_lev_T}")
    max_lev_M = ds['M'].dropna(dim='lev', how='all').lev.max().item()
    print(f"Max lev M: {max_lev_M}")
    if max_lev_T == max_lev_M:
        z2 = max_lev_T
    else:
        z2 = min(max_lev_T, max_lev_M)
    print(f"z2: {z2}")
    # Comprovar si z1 i z2 són vàlids
    if z1 is not None and z2 is not None and z1 < z2 and z2 - z1 >= 10:
        print(f"z1 i z2 són vàlids: {z1}, {z2}")
    else:
        print("z1 i z2 no són vàlids. Comprova els nivells de les variables M i T.")
        z1 = None
        z2 = None
  
    return z1, z2



