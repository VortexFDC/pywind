# PYWIND Sample Data
We present the sample data which is used within this repository to play with.

## Froya site

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3403362.svg)](https://doi.org/10.5281/zenodo.3403362) 


We are utilizing data for Froya site which are prepared for extraction in the `/data` folder. 
The data can be downloaded appart from this repository -->
[froya data pack](http://download.vortexfdc.com/froya.zip)  http://download.vortexfdc.com/froya.zip

To check the data integrity and autenticity the md5 checksun is:
md5sum froya.zip : 3ad4368eef6c8bb6ce8137448cdaaa1c

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

- Format txt @ 100m height (data/froya/vortex/SERIE/vortex.serie.era5.utc0.100m.txt)
<br /><br />


<div align="center"><img src="images/logo_VORTEX.png" width="200px"> </center>
