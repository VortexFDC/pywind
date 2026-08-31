# =============================================================================
# Authors: Oriol L
# Company: Vortex F.d.C.
# Year: 2026
# =============================================================================

"""
Helper functions for chapter 8: spatial and temporal wind variability.

The seasia data pack (see README_about_datasets.md) holds one year of
half-hourly hub-height (111 m) wind speed at four points of a ~40 x 45 km
mesoscale domain in Southeast Asia, plus the annual-mean wind speed raster
of the whole domain. Coordinates are km offsets from the domain SW corner
(the pack is anonymised: no geographic coordinates).

This chapter is the reproducible companion of the Vortex blog post
"A better view of variables variability (without eating your tongue)".
"""

from typing import Callable, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# One colour per point, used consistently in every figure of the chapter.
POINT_COLORS = {'P1': '#e63946', 'P2': '#457b9d',
                'P3': '#1d3557', 'P4': '#52b788'}


# =============================================================================
# Readers
# =============================================================================

def read_four_points(infile: str) -> pd.DataFrame:
    """
    Read the four-points wind speed csv (gzipped; pandas handles it).

    Parameters
    ----------
    infile: str
        Path to four_points_ws.csv.gz. First column is `time_utc`,
        then one wind speed column per point (P1..P4), in m/s.

    Returns
    -------
    pd.DataFrame
        Datetime-indexed (UTC) DataFrame, one column per point.
    """
    df = pd.read_csv(infile, index_col=0, parse_dates=True)
    df.index.name = 'time'
    return df.dropna(how='all')


def read_points_meta(infile: str) -> pd.DataFrame:
    """Read the per-point metadata (km offsets, mean WS), indexed by code."""
    return pd.read_csv(infile).set_index('code')


def read_mean_ws_map(infile: str) -> xr.Dataset:
    """Read the annual-mean wind speed raster (netCDF, km coordinates)."""
    return xr.open_dataset(infile)


def convert_index_to_local(obj, utc_offset: float = 0.0):
    """
    Shift a datetime-indexed Series/DataFrame from UTC to local time.

    Daily cycles are tied to the sun: compute them in local time, or the
    peaks will show up at misleading hours.
    """
    out = obj.copy()
    out.index = out.index + pd.Timedelta(hours=utc_offset)
    return out


# =============================================================================
# Temporal summaries
# =============================================================================

def daily_cycle(series: pd.Series,
                quantiles: Tuple[float, float] = (0.25, 0.75)
                ) -> pd.DataFrame:
    """Mean and quantile envelope per hour of day (index 0..23)."""
    grouped = series.groupby(series.index.hour)
    return pd.DataFrame({'mean': grouped.mean(),
                         'q_lo': grouped.quantile(quantiles[0]),
                         'q_hi': grouped.quantile(quantiles[1])})


def monthly_cycle(series: pd.Series,
                  quantiles: Tuple[float, float] = (0.25, 0.75)
                  ) -> pd.DataFrame:
    """Mean and quantile envelope per month of year (index 1..12)."""
    grouped = series.groupby(series.index.month)
    return pd.DataFrame({'mean': grouped.mean(),
                         'q_lo': grouped.quantile(quantiles[0]),
                         'q_hi': grouped.quantile(quantiles[1])})


def hour_month_table(series: pd.Series) -> pd.DataFrame:
    """Mean wind speed per (month, hour): the classic 12 x 24 table."""
    df = pd.DataFrame({'ws': series.values,
                       'month': series.index.month,
                       'hour': series.index.hour})
    return df.pivot_table(values='ws', index='month', columns='hour',
                          aggfunc='mean')


def variability_summary(df: pd.DataFrame,
                        calm_threshold: float = 3.0) -> pd.DataFrame:
    """
    One row per point: mean, std, P50, P90, calm fraction, and the hour /
    month of the diurnal and seasonal maxima.
    """
    rows = {}
    for col in df.columns:
        s = df[col].dropna()
        rows[col] = {
            'mean (m/s)': s.mean(),
            'std (m/s)': s.std(),
            'P50 (m/s)': s.quantile(0.50),
            'P90 (m/s)': s.quantile(0.90),
            f'calm <{calm_threshold:g} m/s (%)':
                100.0 * (s < calm_threshold).mean(),
            'peak hour (local)': int(daily_cycle(s)['mean'].idxmax()),
            'peak month': int(monthly_cycle(s)['mean'].idxmax()),
        }
    return pd.DataFrame(rows).T.round(2)


# =============================================================================
# Map plotting
# =============================================================================

def plot_mean_ws_map(ds_map: xr.Dataset, meta: pd.DataFrame = None,
                     title: str = 'Annual-mean wind speed') -> None:
    """
    Plot the mean wind speed raster (km axes, viridis) with white contours,
    optionally marking the points in `meta` (needs x_km, y_km columns).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    mesh = ax.pcolormesh(ds_map['x_km'], ds_map['y_km'], ds_map['mean_ws'],
                         cmap='viridis', shading='auto')
    contours = ax.contour(ds_map['x_km'], ds_map['y_km'], ds_map['mean_ws'],
                          levels=6, colors='white', linewidths=0.5,
                          alpha=0.6)
    ax.clabel(contours, inline=True, fontsize=7, fmt='%.1f')
    fig.colorbar(mesh, ax=ax, label='Mean wind speed (m/s)', shrink=0.85)
    if meta is not None:
        for code, row in meta.iterrows():
            _mark_point(ax, row['x_km'], row['y_km'],
                        POINT_COLORS.get(code, 'black'), code)
    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def _mark_point(ax, x: float, y: float, color: str, label: str = None):
    """Double circle marker + optional label box for a point on the map."""
    ax.scatter([x], [y], s=260, marker='o', facecolors='white',
               edgecolors=color, linewidths=3, zorder=10)
    ax.scatter([x], [y], s=70, color=color, zorder=11,
               edgecolors='black', linewidths=0.6)
    if label:
        ax.annotate(label, (x, y), textcoords='offset points',
                    xytext=(10, 10), fontsize=11, fontweight='bold',
                    color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=color, lw=1))


def _assign_corners(meta: pd.DataFrame, extent) -> Dict[str, tuple]:
    """Give each point the inset corner (fractions) closest to it."""
    x0, x1, y0, y1 = extent
    xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
    corners = {'TL': (0.02, 0.74), 'TR': (0.66, 0.74),
               'BL': (0.02, 0.04), 'BR': (0.66, 0.04)}
    remaining = dict(corners)
    assigned = {}
    for code, row in meta.iterrows():
        pref = ('T' if row['y_km'] >= yc else 'B') + \
               ('L' if row['x_km'] <= xc else 'R')
        if pref not in remaining:
            pref = next(iter(remaining))
        assigned[code] = remaining.pop(pref)
    return assigned


def plot_map_with_insets(ds_map: xr.Dataset, meta: pd.DataFrame,
                         df_local: pd.DataFrame,
                         draw_inset: Callable, title: str) -> None:
    """
    The chapter's key figure: the mean wind speed map with one small inset
    panel per point, drawn by `draw_inset(inset_ax, series, color)`.

    Use it with draw_histogram / draw_daily_cycle / draw_monthly_cycle to
    show how the distribution shape, the daily cycle and the yearly cycle
    vary across the map.
    """
    x0, x1 = float(ds_map['x_km'].min()), float(ds_map['x_km'].max())
    y0, y1 = float(ds_map['y_km'].min()), float(ds_map['y_km'].max())
    extent = (x0, x1, y0, y1)
    corners = _assign_corners(meta, extent)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    mesh = ax.pcolormesh(ds_map['x_km'], ds_map['y_km'], ds_map['mean_ws'],
                         cmap='viridis', shading='auto', alpha=0.55)
    fig.colorbar(mesh, ax=ax, label='Mean wind speed (m/s)', shrink=0.85)

    fw, fh = 0.30, 0.22   # inset size as a fraction of the map extent
    for code, row in meta.iterrows():
        color = POINT_COLORS.get(code, 'black')
        _mark_point(ax, row['x_km'], row['y_km'], color, code)
        fx, fy = corners[code]
        inset = ax.inset_axes([x0 + fx * (x1 - x0), y0 + fy * (y1 - y0),
                               fw * (x1 - x0), fh * (y1 - y0)],
                              transform=ax.transData)
        draw_inset(inset, df_local[code].dropna(), color)
        inset.set_title(f"{code} · {row['mean_ws']:.1f} m/s",
                        fontsize=8, color=color, fontweight='bold')
        inset.set_xticks([])
        inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_color(color)
            spine.set_linewidth(1.2)
        inset.patch.set_alpha(0.90)
        # Dashed connector from the point to the nearest inset edge.
        bx0, bx1 = x0 + fx * (x1 - x0), x0 + (fx + fw) * (x1 - x0)
        by0, by1 = y0 + fy * (y1 - y0), y0 + (fy + fh) * (y1 - y0)
        cx = min(max(row['x_km'], bx0), bx1)
        cy = min(max(row['y_km'], by0), by1)
        ax.plot([row['x_km'], cx], [row['y_km'], cy], ls='--', lw=0.9,
                color=color, alpha=0.75, zorder=9)

    ax.set_xlabel('Easting (km)')
    ax.set_ylabel('Northing (km)')
    ax.set_aspect('equal')
    ax.set_title(title, fontweight='bold')
    plt.tight_layout()
    plt.show()


# =============================================================================
# Inset drawers (pass one of these to plot_map_with_insets)
# =============================================================================

def draw_histogram(inset, series: pd.Series, color: str) -> None:
    """Wind speed histogram, 1 m/s bins."""
    inset.hist(series.values, bins=np.arange(0, 20.01, 1.0), density=True,
               color=color, alpha=0.75, edgecolor='black', linewidth=0.3)


def draw_daily_cycle(inset, series: pd.Series, color: str) -> None:
    """Mean wind speed per hour of day (series must be in local time)."""
    cycle = daily_cycle(series)['mean']
    inset.plot(cycle.index, cycle.values, color=color, lw=1.8)
    inset.fill_between(cycle.index, cycle.values, color=color, alpha=0.2)
    inset.set_ylim(bottom=0)


def draw_monthly_cycle(inset, series: pd.Series, color: str) -> None:
    """Mean wind speed per month of year."""
    cycle = monthly_cycle(series)['mean']
    inset.bar(cycle.index, cycle.values, color=color, alpha=0.8,
              edgecolor='black', linewidth=0.3)
    inset.set_ylim(bottom=0)
