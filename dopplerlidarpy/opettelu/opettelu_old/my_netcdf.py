#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:22:07 2019

@author: manninan
"""

#from netCDF4 import Dataset
from proggis import my_arg_parser

my_arg_parser()

#rootgrp = Dataset("test.nc", "w", format="NETCDF4_CLASSIC")


#level = rootgrp.createDimension("level", None)
#time = rootgrp.createDimension("dim_time", None)
#lat = rootgrp.createDimension("lat", None)
#lon = rootgrp.createDimension("lon", None)

#time = rootgrp.createVariable("time",'float',"dim_time")
#attrbts = rootgrp.createVariable("attributes")
#dimnsns = rootgrp.createVariable("dimensions")


#rootgrp.close()