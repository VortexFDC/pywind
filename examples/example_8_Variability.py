# =============================================================================
# Authors: Oriol L
# Company: Vortex F.d.C.
# Year: 2026
# =============================================================================

"""
Overview:
---------
This script studies WIND VARIABILITY in its two dimensions — spatial and
temporal — from the seasia data pack: one year of half-hourly wind speed at
111 m at four points of a ~40 x 45 km mesoscale domain in Southeast Asia,
plus the annual-mean wind speed raster of the whole domain.

It is the reproducible companion of the Vortex blog post
"A better view of variables variability (without eating your tongue)":

1. The MAP of annual-mean wind speed: spatial variability of the mean.
2. The SERIES at one point (year + ten-day zoom): raw temporal variability.
3. Three temporal summaries at that point: histogram, daily cycle, yearly
   cycle — each answers a different question.
4. The map again, with histogram / daily-cycle / yearly-cycle insets at the
   four points: the temporal summaries themselves vary in space, well
   beyond the mean.

Objective:
----------
- Show that variability is temporal AND spatial.
- Show that spatial variability is much more than a map of means.
- Show that temporal variability can be analysed in several distinct ways.
"""

# =============================================================================
# 1. Import Libraries
# =============================================================================
import os
import matplotlib.pyplot as plt
import numpy as np
from example_8_Variability_functions import (
    read_four_points,
    read_points_meta,
    read_mean_ws_map,
    convert_index_to_local,
    daily_cycle,
    monthly_cycle,
    hour_month_table,
    variability_summary,
    plot_mean_ws_map,
    plot_map_with_insets,
    draw_histogram,
    draw_daily_cycle,
    draw_monthly_cycle,
)

# =============================================================================
# 2. Define Paths and Point
# =============================================================================

SITE = 'seasia'
pwd = os.getcwd()
base_path = os.path.join(pwd, '../data', SITE)
series_csv = os.path.join(base_path, 'four_points_ws.csv.gz')
meta_csv = os.path.join(base_path, 'four_points_meta.csv')
map_nc = os.path.join(base_path, 'mean_ws_map.nc')

UTC_OFFSET = 7.0   # Indochina Time — cycles must be computed in local time
POINT = 'P3'       # mid-range point; try 'P1' (ridge) or 'P2' (valley) too

print("\n" + "#" * 26 + " Vortex F.d.C. 2026 " + "#" * 26 + "\n")

# =============================================================================
# 3. Read the Data Pack
# =============================================================================

df = read_four_points(series_csv)
meta = read_points_meta(meta_csv)
ds_map = read_mean_ws_map(map_nc)

df_local = convert_index_to_local(df, utc_offset=UTC_OFFSET)
print(df_local.describe().round(2))
print()
print(meta)

# =============================================================================
# 4. Spatial Variability of the Mean: the Map
# =============================================================================
# Within ~40 km the annual mean spans more than a factor of two. Since wind
# power goes with the cube of speed at moderate winds, siting on the wrong
# side of this map is not a small error.

plot_mean_ws_map(ds_map, meta,
                 title='Annual-mean wind speed across ~40 km — '
                       'a factor-of-two range')

# =============================================================================
# 5. Temporal Variability, Raw: One Year at One Point
# =============================================================================

ws = df_local[POINT].dropna()

fig, axes = plt.subplots(2, 1, figsize=(12, 6))
ws.plot(ax=axes[0], lw=0.3, color='steelblue')
ws.resample('MS').mean().plot(ax=axes[0], drawstyle='steps-post',
                              color='crimson', lw=2, label='monthly mean')
axes[0].axhline(ws.mean(), ls='--', color='navy',
                label=f'annual mean {ws.mean():.1f} m/s')
axes[0].legend()
axes[0].set_title(f'{POINT} — one year of half-hourly wind speed')
ws.loc['2014-12-01':'2014-12-11'].plot(ax=axes[1], color='steelblue')
axes[1].set_title('Ten days — the daily rhythm appears')
plt.tight_layout()
plt.show()

# =============================================================================
# 6. Three Temporal Summaries, Three Different Questions
# =============================================================================
# Histogram: how often is the wind at each strength? Daily cycle: at what
# time of day does the energy arrive? Yearly cycle: which months carry the
# year? None of them is "the" variability — they are projections of the
# same series onto different axes.

dc = daily_cycle(ws)
mc = monthly_cycle(ws)

fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(14, 4))
ax0.hist(ws.values, bins=np.arange(0, 20.01, 0.5), density=True,
         color='steelblue', alpha=0.8, edgecolor='white', linewidth=0.4)
ax0.set_xlabel('Wind speed (m/s)')
ax0.set_ylabel('Frequency')
ax0.set_title('Histogram')

ax1.fill_between(dc.index, dc['q_lo'], dc['q_hi'], alpha=0.25,
                 color='steelblue', label='interquartile range')
ax1.plot(dc.index, dc['mean'], color='navy', lw=2, label='mean')
ax1.set_xticks(range(0, 24, 3))
ax1.set_xlabel('Hour of day (local)')
ax1.set_ylabel('Wind speed (m/s)')
ax1.set_title(f"Daily cycle (peak {int(dc['mean'].idxmax()):02d}h)")
ax1.legend()

ax2.fill_between(mc.index, mc['q_lo'], mc['q_hi'], alpha=0.25,
                 color='steelblue')
ax2.plot(mc.index, mc['mean'], color='navy', lw=2, marker='o')
ax2.set_xticks(range(1, 13))
ax2.set_xlabel('Month')
ax2.set_ylabel('Wind speed (m/s)')
ratio = mc['mean'].max() / mc['mean'].min()
ax2.set_title(f'Yearly cycle (max/min ratio {ratio:.1f}x)')
plt.tight_layout()
plt.show()

# The two cycles interact: the 12 x 24 table shows the daily rhythm
# changing with the season.
table = hour_month_table(ws)
fig, ax = plt.subplots(figsize=(9, 4.5))
mesh = ax.pcolormesh(table.columns, table.index, table.values,
                     cmap='viridis', shading='auto')
fig.colorbar(mesh, ax=ax, label='mean wind speed (m/s)')
ax.set_xlabel('Hour of day (local)')
ax.set_ylabel('Month')
ax.set_title(f'{POINT} — 12 months x 24 hours mean wind speed')
plt.tight_layout()
plt.show()

# =============================================================================
# 7. The Map, Revisited: Spatial Variability of Everything Else
# =============================================================================
# Each temporal summary can be computed at every point — and each varies
# across the map as strongly as the mean does.

plot_map_with_insets(ds_map, meta, df_local, draw_histogram,
                     'Same map, four histogram shapes')
plot_map_with_insets(ds_map, meta, df_local, draw_daily_cycle,
                     'Same map, four daily-cycle shapes')
plot_map_with_insets(ds_map, meta, df_local, draw_monthly_cycle,
                     'Same map, four yearly-cycle shapes')

# =============================================================================
# 8. Summary Table
# =============================================================================
# P1 (ridge) and P2 (valley): opposite diurnal phase (02h vs 18h local)
# AND opposite monsoon (Dec vs Jul). P3 and P4: near-twin means, opposite
# seasonal peaks — the mean map alone would call them equivalent.

print(variability_summary(df_local))
