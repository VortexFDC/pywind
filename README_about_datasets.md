# PYWIND Sample Data
We present the sample data which is used within this repository to play with.

## Froya site

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3403362.svg)](https://doi.org/10.5281/zenodo.3403362) 


We are utilizing data for Froya site which are prepared for extraction in the `/data` folder. 
The data can be downloaded appart from this repository -->
[froya data pack](http://download.vortexfdc.com/froya.zip)  http://download.vortexfdc.com/froya.zip

To check the data integrity and autenticity the md5 checksun is:
md5sum froya.zip : 3ad4368eef6c8bb6ce8137448cdaaa1c

* updated on 2025-05-16 

After unpacking the folder structure should be created as follows under data folder:
``` 
├── froya
│   ├── measurements
│   │   ├── obs.nc
│   │   └── obs.txt
│   └── vortex
│       └── SERIE
│           ├── vortex.serie.era5.utc0.nc
│           └── vortex.serie.era5.utc0.100m.txt
│           └── vortex.remodeling.utc0.100m.txt  __added on 2025-05-16 __
```
        

### A. Observed data

Froya original data can be found [here](https://zenodo.org/records/3403362#.Y1eS5XZByUk).
The site represents an exposed coastal wind climate with open sea, land and mixed fetch from various directions. UTM-coordinates of the Met-mast: 8.34251 E and 63.66638. A post processing has been applied in order to obtain single boom and quality control standards for wind industry.

![View of the area for measurement site in Froya 
](images/Froya-map.png "Froya met mast")

### B. Modeled data
We are also using [Vortex f.d.c](http://www.vortexfdc.com) simulations. <br />

<b>SERIES</b> 20 year long time series computed using WRF at 3km final spatial rsolution. Heights from 30m to 300m  height. <br />

- Format netCDF with multiple heights. (data/froya/vortex/SERIE/vortex.serie.era5.utc0.nc). <br />

- Format txt @ 100m height (data/froya/vortex/SERIE/vortex.serie.era5.utc0.100m.txt) <br />

- Remodeled file in txt file. This has been generated using the measurements and Netcdf in this data pack. See full remodeling report in this pdf: [froya remodeling pdf](docs/vortex.remodeled.froya.utc0.100m.pdf)
<br /><br />


<div align="center"><img src="images/logo_VORTEX.png" width="200px"> </center>
## Southeast Asia variability pack (seasia)

Used by chapter 8 (Variability) — the reproducible companion of the Vortex
blog post "A better view of variables variability (without eating your tongue)".

The data can be downloaded apart from this repository -->
[seasia data pack](http://download.vortexfdc.com/seasia.zip)  http://download.vortexfdc.com/seasia.zip

To check the data integrity and authenticity the md5 checksum is:
md5sum seasia.zip : 8020fab4f1f22e11969466bc5f545296

* updated on 2026-07-10

After unpacking the folder structure should be created as follows under data folder:
```
├── seasia
│   ├── README.txt
│   ├── four_points_ws.csv.gz
│   ├── four_points_meta.csv
│   └── mean_ws_map.nc
```

### Contents

One year (half-hourly, timestamps UTC, local time UTC+7) of hub-height
(111 m) wind speed from a [Vortex f.d.c](http://www.vortexfdc.com) WRF
mesoscale simulation over a ~40 x 45 km domain in Southeast Asia. The pack
is anonymised: coordinates are km offsets from the domain SW corner, no
geographic coordinates are included.

- `four_points_ws.csv.gz` — wind speed (m/s) at four contrasting points of
  the domain: P1 ridge (windiest, 8.4 m/s), P2 valley (calmest, 3.8 m/s),
  P3 median (6.2 m/s), P4 far corner (5.7 m/s).
- `four_points_meta.csv` — per-point metadata (km offsets, mean, height).
- `mean_ws_map.nc` — annual-mean wind speed raster of the whole domain
  (netCDF, coordinates `x_km`/`y_km`).
