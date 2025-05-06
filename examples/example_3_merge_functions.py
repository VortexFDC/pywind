import pandas as pd
import xarray as xr
import numpy as np
from typing import Union, Dict, List
import math

# VORTEX TYPES
vSet = Union[xr.Dataset, pd.DataFrame]
vArray = Union[xr.DataArray, pd.Series]
vData = Union[vSet, vArray]

def find_wind_speed(vs: vSet) -> vArray:
    """
    Calculate the wind speed.

    Given a vSet we return the vArray
    of wind speed, which may be already on the ``vSet``
    or it may need to be obtained from the wind components.
    It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    vs: vSet
        vSet with wind speed (M) or wind components (U & V)

    Returns
    -------
    vArray
        Wind speed data (named M)
    """
    if 'M' in vs:
        m = vs['M']
    else:
        try:
            u = vs['U']
            v = vs['V']
        except KeyError:
            raise ValueError('Cannot obtain M (no U or V)')

        m = np.sqrt(u ** 2 + v ** 2).rename('M')

        m.attrs = vtx_attributes_vars['M']
    return m


def find_direction(vs: vSet) -> vArray:
    """
    Calculate the wind direction.

    Given a vSet we return the vArray
    of wind direction, which may be already on the ``vSet``
    or it may need to be obtained from the wind components.
    It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    vs: vSet
        vSet with wind direction (Dir) or wind components (U & V)

    Returns
    -------
    vArray:
        Wind direction data (named Dir)

    """
    if 'Dir' in vs:
        d = vs['Dir']
    else:
        try:
            u = vs['U']
            v = vs['V']
        except KeyError:
            raise ValueError('Cannot obtain Dir (no U or V)')

        radians = np.arctan2(u, v)
        d = (radians * 180 / math.pi + 180).rename('Dir')

        d.attrs = vtx_attributes_vars['Dir']
    return d

def find_var(var: str, ds: vSet, **kwargs) -> vArray:
    """
    Return the requested variable from the vSet if possible. Given a vSet
    we return the vArray of the variable `var` using the functions
    defined in this module or simply selecting it from the vSet.
    It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    var: str
        Name of the variable. Either existing in the vSet or
        one that is standard Vortex and can be computed:

        .. code-block:: python

            find_new_vars = {
                    # Some variables need to be computed
                    'U': find_zonal_wind,
                    'V': find_meridional_wind,
                    'D': find_density,
                    'M': find_wind_speed,
                    'T': find_temperature_celsius,
                    'P': find_pressure_hpa,
                    'SD': find_standard_deviation,
                    'TI': find_turbulence_intensity,
                    'Dir': find_direction,
                    'energy': find_energy,
                    'power': find_power,
                    'RI': find_richardson,
                    'stability': find_stability,
                    'stabilityClass': find_stability_class,
                    'shear': find_shear,
                    'shear_sd': find_shear_sd,
                    'veer': find_veer,
                    'total_veer': find_total_veer,
                    'inflow': find_inflow,
                    'sector': find_sectors,
                    'Mbin': find_wind_bins,
                    'daynight': find_daynight,
                    'solar_elev': find_solar_elevation,
                    'variance': find_wind_variance,
                }

    ds: vSet

    Returns
    -------
    v: vArray
        Array called `var` and, in case it is a xarray object, with attributes.

    """

    if var in find_new_vars:
        v = find_new_vars[var](ds, **kwargs)
    elif var in ds:
        v = ds[var]
        if var in vtx_attributes_vars:
            v.attrs = vtx_attributes_vars[var]
    else:
        raise ValueError('Cannot obtain variable ' + var + ' from vSet.')

    return v


def get_dataset(vd: vData,
                vars_list: List[str] = None,
                strict: bool = True,
                no_zarr: bool = True) -> Union[xr.Dataset, None]:
    """
    Given a vData return the data in xr.Dataset format

    Sometimes it is useful to know what kind of objects are we dealing
    with instead of having the flexibility of vDatas.

    This function tries to smartly convert your vData to a xarray
    Dataset, and compute the requested variables.

    If the input is:
        - a xr.Datarray: simply convert to dataset
        - a pd.Series: convert to dataframe and then apply convert_to_xarray
        - a pd.DataFrame: add to the index lat, lon, lev, time if they were in the columns, and apply convert_to_xarray

    Try to find the variables of vars_list, and raise an error if
    none is found. If strict, fail if ANY variable is not found.

    If the vData passed was an vArray without name, and we request a
    single var, the code will assume the only var we passed is the one
    we want, and rename it to what we passed in vars_list.

    Parameters
    ----------
    vd: vData
    vars_list: list of variables
        Must be understood by find_var
    strict: bool
        If strict=True the function will fail if any variable
        is missing. If strict=False only fails if all variables fail.
    no_zarr: bool
        Compute the dask arrays if any, so that the result is not a
        dask object.

    Returns
    -------
    xr.Dataset
        The vData in xarray.Dataset format.
    """
    # Make sure we won't return a dask array (zarr)
    if no_zarr:
        if hasattr(vd, 'compute'):
            vd = vd.compute()

    # If we have a vArray, we just convert it to xr.Dataset
    if isinstance(vd, xr.DataArray):
        if vd.name == '' and len(vars_list) == 1:
            vd = vd.rename(vars_list[0])
        vd = vd.to_dataset()
    elif isinstance(vd, pd.Series):
        if vd.name == '' and len(vars_list) == 1:
            vd = vd.rename(vars_list[0])
        vd = convert_to_xarray(vd.to_dataframe())
    elif isinstance(vd, pd.DataFrame):
        newdims = [c for c in vd.columns
                   if c in ['lat', 'lon', 'lev', 'time']]
        coords = {c: np.unique([vd[c].values]) for c in vd.columns
                  if c in ['lat', 'lon', 'lev', 'time']}
        if 0 < len(newdims) < 4:
            vd = vd.set_index(newdims, append=True)
        elif len(newdims) == 4:
            vd = vd.set_index(newdims)

        vd = convert_to_xarray(vd, coords=coords)

    # If we get here, vd should be a xr.Dataset
    variables = []
    for v in vars_list:
        try:
            thisv = find_var(v, vd)
        except ValueError as e:
            if strict:
                print('One of the variables cannot be obtained: ' + v)
                raise e
        else:
            variables.append(thisv)

    if len(variables) == 0:
        return None

    full = xr.merge(variables, combine_attrs="drop")
    full = add_attrs_vars(full)
    full = add_attrs_coords(full)
    return full


find_new_vars = {
    # Some variables need to be computed
    'M': find_wind_speed,
    'Dir': find_direction

}
