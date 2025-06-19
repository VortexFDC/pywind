
from lib_vortexpy_vdatas import (vSet, vArray, apply_general)
from lib_vortexpy_vdatas import  vtx_attributes_vars as attributes_vars
import xarray as xr
import numpy as np
import math



def find_zonal_wind(vs: vSet) -> vArray:
    """
    Calculate the zonal wind component (U).

    Given a vSet we return the vArray
    of zonal wind, which may be already on the ``vSet``
    or it may need to be obtained from the wind speed and
    direction. It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    vs: vSet
        vSet with zonal wind speed (U) or wind speed and
         direction (M & Dir)

    Returns
    -------
    vArray
        Zonal wind speed data (named U)
    """
    if 'U' in vs:
        u = vs['U']
    else:
        try:
            m = vs['M']
            d = vs['Dir']
        except KeyError:
            raise ValueError('Cannot obtain U (no Dir or M)')

        u = -m * np.sin(d * math.pi / 180.)
        u = u.rename('U')

        u.attrs = attributes_vars['U']
    return u


def find_meridional_wind(vs: vSet) -> vArray:
    """
    Calculate the meridional wind component (V).

    Given a vSet we return the vArray
    of meridional wind, which may be already on the ``vSet``
    or it may need to be obtained from the wind speed and
    direction. It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    vs: vSet
        vSet with meridional wind speed (V) or wind speed and
         direction (M & Dir)

    Returns
    -------
    vArray
        Meridional wind speed data (named V)
    """
    if 'V' in vs:
        v = vs['V']
    else:
        try:
            m = vs['M']
            d = vs['Dir']
        except KeyError:
            raise ValueError('Cannot obtain V (no Dir or M)')

        v = -m * np.cos(d * math.pi / 180.)
        v = v.rename('V')

        v.attrs = attributes_vars['V']
    return v


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

        m.attrs = attributes_vars['M']
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

        d.attrs = attributes_vars['Dir']
    return d


def find_richardson(dataset: xr.Dataset, limit: float = 200.) -> xr.DataArray:
    """
    Calculate the richardson number from a dataset for heights
    at the middle height between the original heights.

    Given a xr.Dataset we return the xr.DataArray
    of the richardson number.
    It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    dataset: xr.Dataset
        Dataset with wind components (U & V) and temperature (T)
    limit: float
        Maximum absolute value that RI can take (to avoid infinite's).

    Returns
    -------
    xr.DataArray
        Richardson Number (named RI)
    """

    if 'RI' in dataset:
        ri = dataset['RI']
    elif not hasattr(dataset, 'dims') or \
            'lev' not in dataset.dims or len(dataset['lev']) <= 1:
        raise ValueError('Invalid dataset for RI calculation. The '
                         'dataset must have at least two vertical levels.')
    else:
        dataset = dataset.sortby('lev')

        u = find_zonal_wind(dataset)
        v = find_meridional_wind(dataset)
        t = find_temperature_celsius(dataset)

        levs = dataset.coords['lev'].values
        t0 = 273.15  # T in Kelvin for T = 0 ºC

        ris = []
        for i in range(len(levs) - 1):
            delta_height = levs[i + 1] - levs[i]

            # potential temperature (below level)
            tp0 = t.isel(lev=i) + t0 + 0.0098 * levs[i]
            # potential temperature (above level)
            tp2 = t.isel(lev=i + 1) + t0 + 0.0098 * levs[i + 1]
            temp_fact = 9.81 * delta_height * (tp2.data - tp0.data)

            mean_height = float(np.mean([levs[i + 1], levs[i]]))

            wind_diff = (u.isel(lev=i + 1) - u.isel(lev=i)) ** 2 + \
                        (v.isel(lev=i + 1) - v.isel(lev=i)) ** 2
            wind_diff = wind_diff.where(wind_diff > 0, 0.0001)

            t1 = t.interp(lev=mean_height) + t0
            ri_da = (temp_fact / (wind_diff * t1)).rename('RI')
            ri_da = ri_da.expand_dims({'lev': [mean_height]}, axis=1)
            ris.append(ri_da)
        ri = xr.concat(ris, dim='lev')
        ri = ri.where(ri > -limit, -limit).where(ri < limit, limit)
        ri.attrs = attributes_vars['RI']
    return ri

def find_stability(vs: vSet) -> vArray:
    """
    Calculate the atmospheric stability index (from 0 to 6). Given an RMOL
    time series, the atmospheric stability index is returned as a vArray.

    The classification has been performed according to [1], being the edges:
    >>> _atmospheric_stability_edges = [-1 / 100, -1 / 200, -1 / 500, 1 / 500, 1 / 200, 1 / 50]

    The index corresponds to these classes:
    >>> _atmospheric_stability_classes = ['Very Unstable', 'Unstable', 'Near-neutral Unstable', 'Neutral', 'Near-neutral Stable', 'Stable', 'Very Stable']

    Parameters
    ----------
    vs: vSet
        Time series. 'RMOL' must be contained in the vSet.

    Returns
    -------
    Stability: vArray
        Atmospheric stability time series.

    References
    ----------
    .. [1] "Classification of atmospheric stability", https://www.researchgate.net/figure/Classification-of-atmospheric-stability-according-to-Monin-Obukhov-length-in-tervals_tbl1_266043126

    """
    if 'stability' in vs:
        stability = vs['stability']
    else:
        try:
            rmol = vs['RMOL']
        except KeyError:
            raise ValueError('Cannot obtain stability (no RMOL)')

        # The 5 RMOL values that define the 7 categories of
        # atmospheric stability
        stability = apply_general(np.digitize, rmol,
                                  bins=atmospheric_stability_edges_rmol)

        stability = stability.rename('stability')
        stability.attrs = attributes_vars['stability']

    return stability


def find_stability_class(vs: vSet) -> vArray:
    """
    Calculate the atmospheric stability class (Stable, Very Stable...)

    Given an RMOL time series, the atmospheric stability class is
    returned as a vArray.

    The classification has been performed according to [1], being the edges:
    >>> _atmospheric_stability_edges = [-1 / 100, -1 / 200, -1 / 500, 1 / 500, 1 / 200, 1 / 50]

    The index corresponds to these classes:
    >>> _atmospheric_stability_classes = ['Very Unstable', 'Unstable', 'Near-neutral Unstable', 'Neutral', 'Near-neutral Stable', 'Stable', 'Very Stable']

    Parameters
    ----------
    vs: vSet
        Time series. 'RMOL' or 'stability' must be contained in the vSet.

    Returns
    -------
    stabilityClass: vArray
        Atmospheric stability Class time series.

    References
    ----------
    .. [1] "Classification of atmospheric stability", https://www.researchgate.net/figure/Classification-of-atmospheric-stability-according-to-Monin-Obukhov-length-in-tervals_tbl1_266043126

    """
    if 'stabilityClass' in vs:
        stability_class = vs['stabilityClass']
    else:
        stability = find_stability(vs)

        # The 5 RMOL values that define the 7 categories of
        # atmospheric stability
        classes = np.array(atmospheric_stability_classes)
        stability_class = classes[stability.values]

        if isinstance(stability, xr.DataArray):
            stability_class = stability.copy(data=stability_class)
        else:
            stability_class = pd.Series(stability_class,
                                        index=stability.index)

        stability_class = stability_class.rename('stabilityClass')
        stability_class.attrs = attributes_vars['stabilityClass']

    return stability_class


def find_temperature_celsius(vs: vSet) -> vArray:
    """
    Return the temperature in Celsius degrees.

    Given a vSet we return the vArray
    of the temperature in Celsius.
    It is computed lazily if the inputs are Dask arrays.

    Parameters
    ----------
    vs: vSet
        vSet with temperature (T)

    Returns
    -------
    vArray
        Temperature (named T)
    """
    t0 = 273.15  # T in Kelvin for T = 0 ºC

    try:
        t = vs['T']
    except KeyError:
        raise ValueError('Cannot obtain temperature T')

    # Get mean temperature
    valt = t.mean()
    if valt > 150.:
        t = t - t0

    t.attrs = attributes_vars['T']

    return t



atmospheric_stability_edges_rmol = [-1 / 100, -1 / 200, -1 / 500,
                                    1 / 500, 1 / 200, 1 / 50]
atmospheric_stability_classes = ['Very Unstable', 'Unstable',
                                 'Near-neutral Unstable', 'Neutral',
                                 'Near-neutral Stable', 'Stable',
                                 'Very Stable']