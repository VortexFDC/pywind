"""
Functions to handle vData objects.

Details
-------


"""


from typing import Union, Callable, Dict

import numpy as np
import pandas as pd
import xarray as xr


# TYPES
vSet = Union[xr.Dataset, pd.DataFrame]
vArray = Union[xr.DataArray, pd.Series]
vData = Union[vSet, vArray]


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
    't_lapse_rate': {'long_name': 'Temperature Lapse Rate',
                     'description': 'Temperature Lapse Rate',
                     'units': 'degrees km^-1'},
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


# Coordinates and index
def get_times(vd: vData) -> np.ndarray:
    """
    Obtain the array of times from a vData. The resulting object is
    a numpy vector with the timestamps.

    Parameters
    ----------
    vd: vData
        Vortex data object (xarray or pandas) from which we can
        obtain the time coordinate.

    Returns
    -------
    times: np.ndarray
        Numpy vector with timestamps (pd.Datetime)
    """
    if 'time' in vd:
        times = vd['time'].values
        if not isinstance(times, pd.DatetimeIndex):
            # the 'time' column is forced to datetime type
            # in case it is made up by strings
            times = pd.to_datetime(times)

    elif hasattr(vd, 'index'):
        try:
            times = vd.get_level_values(level='time').values
        except KeyError:
            times = vd.get_level_values(level=0).values
            if not isinstance(times, pd.DatetimeIndex):
                raise ValueError('The index of the vData is not a time.')
    else:
        raise ValueError('Cannot obtain time from vData.')

    return times


# Apply numpy functions
def apply_general(func: Callable, va: vArray, *args,
                  **kwargs) -> vArray:
    """
    Apply a customized function to a vArray

    It doesn't matter if it is a xr.DataArray or a pd.Series, it will
    return the same object that was passed. It works for dask xarray
    objects too.

    It is not the recommended method! If numpy has a function to do it,
    better use it! For example, np.mean() or np.sqrt() will accept
    pandas and xarray objects and return them in the correct form too.

    Parameters
    ----------
    func: Callable
    va: vArray
        pd.Series or xr.DataArray to which we apply `func`
    args: list of objects
        Other positional arguments passed to `func`
    kwargs: dict of str: objects
        Other keyword arguments passed to `func`

    Returns
    -------
    vArray
        Result of applying the function
    """
    if isinstance(va, xr.DataArray):
        result_va = xr.apply_ufunc(func, va, *args, dask='allowed',
                                   kwargs=kwargs)
    elif isinstance(va, pd.Series):
        # we apply the function to the series' values
        # and then create a pd.Series with those values
        result_values = func(va.values, *args, **kwargs)
        result_va = pd.Series(result_values, index=va.index)
    else:
        raise ValueError('Not a vArray! Cannot apply func.')
    return result_va


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
        ds = ds.assign_coords(**coords_dict)

    ds = add_attrs_vars(ds)
    ds = add_attrs_coords(ds)
    return ds


# ATTRIBUTES DATASET

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
