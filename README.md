# Pywind: Wind Data Analysis for Wind Energy Applications [](#top)

Welcome to the **Pywind**, a comprehensive repository about the use of Python  in the wind data analysis for wind energy 
applications. This repository provides tools and examples for reading, plotting, and analyzing wind data. Whether you 
are a student, developer, engineer, or a Python enthusiast interested in wind data, hope you will find valuable and usable 
resources. Find below the structure of this project:

- [Pywind: Wind Data Analysis for Wind Energy Applications ](#pywind-wind-data-analysis-for-wind-energy-applications-)
  - [1. Introduction and Main Goals ](#1-introduction-and-main-goals-)
  - [2. Authors ](#2-authors-)
  - [3. About ](#3-about-)
  - [4. Installation ](#4-installation-)
  - [5. Usage ](#5-usage-)
  - [6. Chapters ](#6-chapters-)
  - [7. License ](#7-license-)
  - [8. Collaboration ](#8-collaboration-)

 _Developed and Created by Vortex FdC._  

## 1. Introduction and Main Goals [](#1.introduction-and-main-goals)

The use of Python in Wind energy is a rapidly growing field, and the effective analysis and manipulation of wind data is crucial for its success. 
This repository provides wind data analysis solutions, from basic tasks to more technical methodologies. 

**Pywind** uses both public measurements and Vortex simulations, like the [SERIES](https://vortexfdc.com/windsite/wind-speed-time-series/) used in the first chapters. 

Feel free to browse, comment, share, reuse the code and ideas.

The structure of this repository is based in three main folders.

- **Data**: Sample wind data from public sources is facilitated for user testing. More in the corresponding [data sample documentation](README_about_datasets.md) which is used in this repository.
- **Examples**: Python scripts can be executed from terminals (bash). This repository has been tested under Linux.
- **Notebooks**: Jupyter Notebooks with extended comments and outputs from examples folder.


## 2. Authors [](#2-authors)

This repository is created and maintained by:
- __Oriol Lacave__, a member of the operational technical team at
[Vortex FDC](http://vortexfdc.com). With over 15 years of experience in the wind industry, Oriol specializes in data 
manipulation, analysis, and improvements. He is a scientific programmer dedicated to delivering added value from 
reanalysis, mesoscale models, and measurements to engineers. 
- __Arnau Toledano__ and Vortex technical team are also contributing to the development of the Pywind repository.
- Vortex team in general. Don't hesitate to contact us!

## 3. About [](#3-about)

[Vortex](http://vortexfdc.com) is a private company that started its technology development in 2005. Comprised of former Wind & Site 
engineers, atmospheric physicists, and computer experts, Vortex has developed its own methodology independently. <br />

Their work is based on the **Weather Research and Forecasting model** [(WRF)](https://www.mmm.ucar.edu/models/wrf), a state-of-the-art mesoscale model developed collaboratively by atmospheric research centers and a thriving community.

Some active groups we have been inspired are:
- The [WRAG](https://groups.io/g/wrag) and the Wind Resource Assessment Group.
- [Pywram](https://www.pywram.com/) stands for Python for Wind Resource Assessment & Metocean Analysis Forum. It originated for Python users 
within WRAG group.

## 4. Installation [](#4-installation)

Clone this repository in your local git environment:

`git clone https://github.com/VortexFDC/pywind.git`


## 5. Usage [](#5-usage)

Each section has its own example and notebook files, located in the [examples](examples) and [notebooks](notebooks) folders, respectively.

For each section a function file is provided, which is imported from the main notebook/example file of the section.

To execute the examples, you can run the following commands from the terminal or your prefered IDE.

`python example_1_read_netcdf.py`

## 6. Chapters [](#6-chapters)

- [Chapter 1](notebooks/example_1_read_netcdf.ipynb)
Read netcdf files with the xarray libraries. You will open and make basic operations. A quick overview of the data if done using pandas libraries.


## 7. License [](#7-license)

MIT License. Consult [LICENSE](/LICENSE)

Please, use, modify and share if you find it useful.

## 8. Collaboration [](#8-collaboration)

We encourage collaboration, proposals and new ideas. 

Plase follow the [collaboration guidelines](CONTRIBUTING.md) for work proposals. 

You can also use the discussions in this repository for new ideas, Q&A, questions, etc.

[Contact us](https://vortexfdc.com/contact/)

[Back to top.](#top)

<div align="center"><img src="images/logo_VORTEX.png" width="200px"> </center>

