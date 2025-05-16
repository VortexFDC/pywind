import pandas as pd
import xarray as xr
import numpy as np
import os
from typing import Union, Dict, List


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
    VORTEX f.d.c. (www.vortexfdc.com) - Computed at 3km resolution based on ERA5 data (designed for correlation purposes)
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

def read_remodeling_serie(filename: str = "vortex.txt",
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
    VORTEX f.d.c. (www.vortexfdc.com) - Computed at 3km resolution based on ERA5 data (designed for correlation purposes)
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
                          'PRE(hPa)': 'P'
        }
    __ds = rename_vars(__ds, vars_new_names)

    __ds = add_attrs_vars(__ds)
    return __ds




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


def read_txt_to_pandas(infile: str,
                       utc: float = 0.,
                       silent: bool = True,
                       **kwargs) -> pd.DataFrame:
    """
    Read a txt file with flexible options as a pandas DataFrame.
    Converts to UTC 0 if not in that UTC hour.

    Parameters
    ----------
    infile: str
        txt file. by default, columns separated by spaces

    utc: float, optional
        If utc (float number) is passed, the txt is assumed to be in that
        offset and converted to UTC0.

    silent: bool
        if silent, suppress all print statements.

    kwargs
        to override defaults: maybe there is a header, or we want other
        names for the columns, or the separator is not spaces, or we want
        to skip the first rows because they are not part of the dataframe.
        Also specify the date time columns (parse_dates argument)

    Returns
    -------
    pd.DataFrame

    """
    if not silent:
        # print a bit of the file to see the structure
        print(f'Reading txt file {infile}. First 5 lines:')
        print(''.join(read_head_txt(infile, lines=5)))

    readcsv_kwargs = {
        'sep': r"\s+",                      # sep = one or more spaces
        'parse_dates': {'time': [0, 1]},    # make sure col is time
        'index_col': 'time',                # do not change
        'date_format': '%Y%m%d %H%M',
    }
    readcsv_kwargs.update(kwargs)
    df: pd.DataFrame = pd.read_csv(infile, **readcsv_kwargs)

    if not silent:
        print(f'Read csv using kwargs: {readcsv_kwargs}')
        print(df.head())

    df.dropna(inplace=True)

    # Change UTC
    df.index = _convert_from_local_to_utc(df.index,
                                          utc_local=utc,
                                          silent=silent)

    if not silent:
        print('Formatted DataFrame')
        print(df.head())

    return df


# This function uses two auxiliary functions to read the head and to deal with time zones
def read_head_txt(infile: str, lines: int = 8) -> List[str]:
    """
    Get a list of the first lines of a txt file. Useful to print logs,
    or use in other functions that can deduce metadata of a file from
    the first lines of text of the file.

    Parameters
    ----------
    infile: str
        Path to the .txt file
    lines: int
        Maximum number of lines to read and return

    Returns
    -------
    head: List[str]
        Concatenated lines read from the file

    Examples
    --------
    >>> print(''.join(read_head_txt('/path/to/file.txt', lines=4)))
    Will print the first 4 lines (respecting the line skips).

    """
    if not os.path.isfile(infile):
        raise IOError('File ' + infile + ' not found.')

    head = [line for i, line in enumerate(open(infile, 'r')) if i < lines]

    return head


def _convert_from_local_to_utc(time_values: Union[pd.Series, pd.DatetimeIndex,
                               pd.Index], utc_local=0., silent=True) -> \
        Union[pd.Series, pd.DatetimeIndex, pd.Index]:
    """
    Convert time values from local time to UTC

    Parameters
    ----------
    time_values: Union[pd.Series, pd.DatetimeIndex, pd.Index]
        Datetime values to be UTC0 converted
    utc_local: float
        Timezone difference
    silent: bool
        Print some info if True

    Returns
    -------
    Union[pd.Series, pd.DatetimeIndex]

    """

    if not silent:
        print(f'Changing utc: {pd.Timedelta(utc_local, "h")}')

    return time_values - pd.Timedelta(utc_local, 'h')


# We also set up some other function to rename variables
def rename_vars(dataset: xr.Dataset,
                vars_new_names: Dict[str, str]) \
                -> xr.Dataset:
    """
    Rename the variables in the given dataset with the dictionary provided.

    Parameters
    ----------
    dataset: xr.Dataset
        the dataset we want to change the variables/columns names.

    vars_new_names: Dict[str, str]
        original and new name for the variable we want to rename in
        the dataset. The dataset is overwritten.

    Returns
    -------
    dataset: xr.Dataset
        Dataset with the new variables names overwritten.

    """
    for old_name, new_name in vars_new_names.items():
        if old_name not in dataset.variables:
            raise UserWarning("This variable is not in the dataset: " + str(
                old_name))
        if new_name not in vtx_attributes_vars.keys():
            raise UserWarning("This new variable name variable implemented "
                              "in the vortexpy")
        dataset = dataset.rename({old_name: new_name})
    return dataset


def _get_coordinates_vortex_header(filename: str,
                                   patterns: Dict[str, str] = None,
                                   line: int = 0)\
        -> Dict[str, float]:
    """
    Read a txt file header

    Parameters
    ----------
    filename: str

    patterns: Dictionary
        What to search for , just before a =

    line: int
        which line to read

    Returns
    -------
    metadata: Dict[str, float]
        Dictionary containing

    """
    if patterns is None:
        patterns = {'Lat=': 'lat', 'Lon=': 'lon',
                    'Timezone=': 'utc',
                    'Hub-Height=': 'lev'}

    headerfile = read_head_txt(filename, lines=15)
    metadata = {}
    for info in headerfile[line].split(' '):
        for pattern, keyword in patterns.items():
            if pattern in info:
                metadata[keyword] = float(info.replace(pattern, ''))
    return metadata


def convert_to_xarray(df: pd.DataFrame,
                      coords: Dict[str, Union[float, np.ndarray]] = None
                      ) -> xr.Dataset:
    """
    Convert a dataframe to a xarray object.

    Parameters
    ----------
    df: pd.DataFrame
    coords: Dict[str, Union[float, np.ndarray]]
        Info about lat, lon, lev so that the new dimensions can be added

    Returns
    -------
    xr.Dataset
        With un-squeezed dimensions and added attributes
    """
    ds: xr.Dataset = df.to_xarray()
    if coords is not None:
        coords_dict = {name: [float(val)] for name, val in coords.items()
                       if name not in ds.dims}
        ds = ds.expand_dims(coords_dict)
    ds = add_attrs_vars(ds)
    ds = add_attrs_coords(ds)
    return ds


def add_attrs_vars(ds: xr.Dataset,
                   attributes_vars: Dict[str, Dict[str, str]] = None,
                   remove_existing_attrs: bool = False) -> xr.Dataset:
    """
    Add attributes information to variables from a dataset.

    If no `attributes_vars` dictionary is passed, the default
    attributes from the vars module are used.

    In xarray, a variable can have attributes :

    .. code-block:: python

        data['U'].attrs = {'description': 'Zonal Wind Speed',
                           'long_name'  : 'U wind speed',
                           'units'      : 'm/s'}

    Parameters
    ----------
    ds : xarray.Dataset

    attributes_vars : dict, optional
        An attributes_vars is a dictionary whose keys are strings that
        represent variables (this could produce clashing of models) and
        each has some attributes like description, long_name, units.

    remove_existing_attrs : bool, False
        True will put only the attributes of `attributes_vars` and
        remove existing attributes, **including ENCODING details**.

    Returns
    -------
    xarray.Dataset
        Data with the new attributes
    """
    if attributes_vars is None:
        attributes_vars = vtx_attributes_vars

    for var in ds.data_vars:
        if remove_existing_attrs:
            attributes = {}
        else:
            attributes = ds[var].attrs

        if var in attributes_vars:
            # noinspection PyTypeChecker
            for key, info in attributes_vars[var].items():
                attributes[key] = info

        ds[var].attrs = attributes

    return ds


def add_attrs_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Add attributes information to coordinates from a dataset.

    Used for lat, lon and lev.

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.Dataset
        Data with the new attributes for the coordinates
    """
    if 'lat' in ds:
        ds['lat'].attrs = {'units': 'degrees', 'long_name': 'Latitude'}
    if 'lon' in ds:
        ds['lon'].attrs = {'units': 'degrees', 'long_name': 'Longitude'}
    if 'lev' in ds:
        ds['lev'].attrs = {'units': 'metres', 'long_name': 'Level'}

    return ds


vtx_attributes_vars = {
    'U': {'description': 'Zonal Wind Speed',
          'long_name': 'U wind speed',
          'units': 'm/s'},
    'V': {'description': 'Meridional Wind Speed Component',
          'long_name': 'V wind speed',
          'units': 'm/s'},
    'W': {'description': 'Vertical Wind Speed Component',
          'long_name': 'W wind speed',
          'units': 'm/s'},
    'M': {'description': 'Wind Speed (module velocity)',
          'long_name': 'Wind speed',
          'units': 'm/s'},
    'TI': {'long_name': 'Turbulence Intensity',
           'description': 'Turbulence Intensity',
           'units': '%'},
    'Dir': {'description': 'Wind Direction',
            'long_name': 'Wind direction',
            'units': 'degrees'},
    'SD': {'description': 'Wind Speed Standard Deviation',
           'long_name': 'Wind Speed Standard Deviation',
           'units': 'm/s'},
    'DSD': {'description': 'Wind Direction Standard Deviation',
            'long_name': 'Wind Direction Standard Deviation',
            'units': 'degrees'},
    'variance': {'description': 'Wind Speed Variance',
                 'long_name': 'Wind Speed Variance',
                 'units': 'm^2/s^2'},
    'T': {'description': 'Air Temperature',
          'long_name': 'Air Temperature',
          'units': 'Deg.Celsius'},
    'P': {'description': 'Pressure',
          'long_name': 'Pressure',
          'units': 'hPa'},
    'D': {'long_name': 'Density',
          'description': 'Air Density',
          'units': 'kg/m^(-3)'},
    'RMOL': {'description': 'Inverse Monin Obukhov Length',
             'long_name': 'Inverse Monin Obukhov Length',
             'units': 'm^-1'},
    'L': {'description': 'Monin Obukhov Length',
          'long_name': 'Monin Obukhov Length',
          'units': 'm'},
    'stability': {'description': 'Atmospheric Stability Index (RMOL)',
                  'long_name': 'Atmospheric Stability (idx)',
                  'units': ''},
    'stabilityClass': {'description': 'Atmospheric Stability Class (RMOL)',
                       'long_name': 'Atmospheric Stability (class)',
                       'units': ''},
    'HGT': {'description': 'Terrain Height (above sea level)',
            'long_name': 'Terrain Height',
            'units': 'm'},
    'inflow': {'long_name': 'Inflow angle',
               'description': 'Inflow angle',
               'units': 'degrees'},
    'RI': {'long_name': 'Richardson Number',
           'description': 'Richardson Number',
           'units': ''},
    'shear': {'long_name': 'Wind Shear Exponent',
              'description': 'Wind Shear Exponent',
              'units': ''},
    'shear_sd': {'long_name': 'Wind SD Shear',
                 'description': 'Wind SD Shear',
                 'units': ''},
    'veer': {'long_name': 'Wind Directional Bulk Veer',
             'description': 'Wind Directional Bulk Veer',
             'units': 'degrees m^-1'},
    'total_veer': {'long_name': 'Wind Directional TotalVeer',
                   'description': 'Wind Directional Total Veer',
                   'units': 'degrees m^-1'},
    'sector': {'long_name': 'Wind Direction Sector',
               'description': 'Wind Direction Sector',
               'units': ''},
    'Mbin': {'long_name': 'Wind Speed Bin',
             'description': 'Wind Speed Bin (round to nearest int)',
             'units': ''},
    'daynight': {'long_name': 'Day or Night',
                 'description': 'Day or Night',
                 'units': ''},
    'solar_elev': {'long_name': 'Solar Elevation',
                   'description': 'Solar Elevation Angle',
                   'units': 'degrees'},
    'power': {'long_name': 'Power',
              'description': 'Approximation to the power expected at '
                             'this instant (energy/time)',
              'units': 'kW'},
    'energy': {'long_name': 'Energy Production',
               'description': 'Approximation to the energy expected from '
                              'the power and time frequency of the series',
               'units': 'kWh'},
    'SST': {'long_name': 'Sea Surface Temperature',
            'description': 'Sea Surface Temperature',
            'units': 'K'},
    'HFX': {'long_name': 'Heat Flux Surface',
            'description': 'Upward heat flux at the surface',
            'units': 'W m-2'},
    'PBLH': {'long_name': 'Boundary Layer Height',
             'description': 'Boundary Layer Height',
             'units': 'm'},
    'RH': {'long_name': 'Relative Humidity',
           'description': 'Relative Humidity',
           'units': '%'},
    'TP': {'long_name': 'Potential Temperature',
           'description': 'Potential Temperature',
           'units': 'K'},
    'T2': {'long_name': 'Air Temperature at 2m',
           'description': 'Air Temperature at 2m',
           'units': 'K'},
    'TKE_PBL': {'long_name': 'Turbulent Kinetic Energy',
                'description': 'Turbulent Kinetic Energy',
                'units': 'm^2/s^2'},
    'Gust3s': {'long_name': '3-second Wind Gust',
               'description': '3-second Wind Gust',
               'units': 'm/s'},
}