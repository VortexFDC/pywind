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

1. Vortex Text Series - Text file with multiple columns and a header.
2. Vortex remodeling - txt: A LT extrapolation combining measurements and vortex time series.

Data Storage:
------------
The acquired data is stored in two data structures for comparison and analysis:
- Pandas DataFrame

Objective:
----------
- To understand the variance in data storage when using Pandas.
- Utilize the 'describe' , head and other methods from Pandas for a quick overview of the dataset.
"""

# =============================================================================
# 1. Import Libraries
# =============================================================================

from typing import Dict
from example_2_read_txt_functions import *
from example_2_read_txt_functions import _get_coordinates_vortex_header

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
print('Measurements txt: ', measurements_txt)
print('Vortex txt: ', vortex_txt)

print()
print('#'*26, 'Vortex f.d.c. 2024', '#'*26)
print()

# =============================================================================
# 3. Read Vortex Text Series Functions
# =============================================================================

# Read Text Series

# Call read_txt_to_pandas with particular options for file vortex_txt
# `vortex_txt` format is like this:

# Lat=52.16632  Lon=14.12259  Hub-Height=100  Timezone=00.0   ASL-Height(avg. 3km-grid)=68  (file requested on 2023-09-28 10:30:31)
# VORTEX (www.vortexfdc.com) - Computed at 3km resolution based on ERA5 data (designed for correlation purposes)
#
# YYYYMMDD HHMM  M(m/s) D(deg)  T(C)  De(k/m3) PRE(hPa)      RiNumber  RH(%)   RMOL(1/m)
# 20030101 0000    7.5    133   -9.2    1.32    1000.2           0.26   80.3      0.0081
# 20030101 0100    7.4    136  -10.0    1.32     999.8           0.25   82.1      0.0059

def read_vortex_serie(filename: str = "vortex.txt",
                      vars_new_names: Dict = None) -> xr.Dataset:
    """
    Read typical vortex time series from SERIES product and return
    an xarray.Dataset

    Parameters
    ----------
    vars_new_names: Dict
        the dictionary with the old names to new names

    filename: str
        just the filename is enough

    Returns
    -------
    ds: xarray.Dataset
        Dataset

    Examples
    --------
    Lat=52.90466  Lon=14.76794  Hub-Height=130  Timezone=00.0   ASL-Height(avg. 3km-grid)=73  (file requested on 2022-10-17 11:34:05)
    VORTEX (www.vortex.es) - Computed at 3km resolution based on ERA5 data (designed for correlation purposes)
    YYYYMMDD HHMM  M(m/s) D(deg)  T(C)  De(k/m3) PRE(hPa)      RiNumber  RH(%)   RMOL(1/m)
    19910101 0000    8.5    175    2.1    1.25     988.1           0.56   91.1      0.

    """
    patterns = {'Lat=': 'lat',
                'Lon=': 'lon',
                'Timezone=': 'utc',
                'Hub-Height=': 'lev'}
    metadata = _get_coordinates_vortex_header(filename, patterns, line=0)
    data = read_txt_to_pandas(filename, utc=metadata['utc'],
                              skiprows=3, header=0, names=None)
    __ds = convert_to_xarray(data, coords=metadata).squeeze()

    if vars_new_names is None:
        vars_new_names = {'M(m/s)': 'M',
                          'D(deg)': 'Dir',
                          'T(C)': 'T',
                          'De(k/m3)': 'D',
                          'PRE(hPa)': 'P',
                          'RiNumber': 'RI',
                          'RH(%)': 'RH',
                          'RMOL(1/m)': 'RMOL'}
    __ds = rename_vars(__ds, vars_new_names)

    __ds = add_attrs_vars(__ds)
    return __ds

ds_vortex = read_vortex_serie(vortex_txt)
print(ds_vortex)
print()

df_vortex = ds_vortex.to_dataframe() # convert to dataframe

# Quickly inspect with head() and describe() methods

print('Vortex SERIES:\n' ,df_vortex[['M', 'Dir']].head())
print()

# =============================================================================
# 4. Read Measurements Txt
# =============================================================================

def read_vortex_obs_to_dataframe(infile: str,
                                 with_sd: bool = False,
                                 out_dir_name: str = 'Dir',
                                 **kwargs) -> pd.DataFrame:
    """
    Read a txt file with flexible options as a pandas DataFrame.

    Parameters
    ----------
    infile: str
        txt file. by default, no header, columns YYYYMMDD HHMM M D

    with_sd: bool
        If True, an 'SD' column is appended
    out_dir_name: str
        Wind direction labeled which will appear in the return dataframe

    Returns
    -------
    df: pd.DataFrame
        Dataframe

    Examples
    --------
    >>> print("The default files read by this function are YYYYMMDD HHMM M D:")
    20050619 0000 6.2 331 1.1
    20050619 0010 6.8 347 0.9
    20050619 0020 7.3 343 1.2

    """

    columns = ['YYYYMMDD', 'HHMM', 'M', out_dir_name]

    if with_sd:
        columns.append('SD')

    readcsv_kwargs = {
        'skiprows': 0,
        'header': None,
        'names': columns,
    }
    readcsv_kwargs.update(kwargs)

    df: pd.DataFrame = read_txt_to_pandas(infile, **readcsv_kwargs)
    return df

df_obs = read_vortex_obs_to_dataframe(measurements_txt)
ds_obs = convert_to_xarray(df_obs)

print('Measurements:\n', df_obs.head())
print()

# =============================================================================
# 5. Now we can compare statistics
# =============================================================================

print('Vortex SERIES Statistics:\n', df_vortex[['M', 'Dir']].describe().round(2))
print()
print('Measurements Statistics:\n', df_obs.describe().round(2))
print()