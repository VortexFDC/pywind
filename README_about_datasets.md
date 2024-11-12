# PYWIND Sample Data
We present the sample data which is used within this repository to play with.

## Lindenberg site
We are utilizing data for Lindenberg site which are prepared for extraction in the `/data` folder. 
The data can be downloaded appart from this repository -->
[lindenberg data pack](http://download.vortexfdc.com/lindenberg.zip)

After unpacking the folder structure should be created as follows under data folder:
``` 
├── lindenberg
│   ├── measurements
│   │   ├── lindenberg_obs.nc
│   │   └── lindenberg_obs.txt
│   └── vortex
│       └── SERIE
│           ├── vortex.serie.era5.100m.utc0.nc
│           └── vortex.serie.era5.100m.utc0.txt
```
        

### A. Observed data

[Lindenberg](data/lindenberg/measurements) original data can be found [here](https://www.cen.uni-hamburg.de/en/icdc/data/atmosphere/weathermast-lindenberg.html).
The met mast is located in a flat land area sorrounded by patches of forests and close to the German's city of 
Lindenberg. 
The location of the met mast is 52.170 latitude and 14.120 longitude in the EPSG:4326 projection(WGS84 lat lon).

![View of the boundary layer field measurement site in Lindenberg-Falkenberg with the 99 m measurement tower. 
Copyright:DWD/J.-P. Leps](images/pic-wettermast-lindenberg.jpg "Lindenebrg met mast")

### B. Modeled data
We are also using [Vortex f.d.c](http://www.vortexfdc.com) simulations. <br />

<b>SERIES</b> 20 year long time series computed using WRF at 3km final spatial rsolution. Heights from 30m to 300m  height. <br />

- Format netCDF with multiple heights. (data/lindenberg/vortex/SERIE/vortex.serie.era5.d02.nc). <br />

- Format txt @ 100m height (data/lindenberg/vortex/SERIE/vortex.serie.era5.d02.txt)
<br /><br />


<div align="center"><img src="images/logo_VORTEX.png" width="200px"> </center>