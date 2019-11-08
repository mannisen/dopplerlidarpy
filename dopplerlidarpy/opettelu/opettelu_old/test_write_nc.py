#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:15:57 2019

@author: manninan
"""

#from utilities.my_nc_write_tools import create_my_nc_var
#from utilities.my_nc_write_tools import create_my_nc_att
from netCDF4 import Dataset
import numpy as np

from dopplerlidarpy.utilities import variables

#from utilities.variables import time
#from utilities.variables import unix_time
#from utilities.variables import height
#from utilities.variables import wind_speed

times = np.arange(0,100)
heights = np.arange(0,10)

# open netcdf file for writing
rootgrp = Dataset('sample.nc','w', format='NETCDF4_CLASSIC')

# create dimensions
rootgrp.createDimension("time",len(times))
rootgrp.createDimension("height",len(heights))

# Choose from predefined variables
vars_to_write = [unix_time,time,height,wind_speed]

nc_vars = {}
for var_info in vars_to_write:
        
    # create variable
    nc_vars[var_info["name"]] = create_my_nc_var(rootgrp,var_info["name"],
           var_info["dtype"],var_info["dims"])
        
    # create attributes
    nc_vars[var_info["name"]] = create_my_nc_att(nc_vars[var_info["name"]],
           var_info["att"])


def write_vars2nc(rootgrp, obs):
    """ Iterate over Cloudnet instances and write to given rootgrp """
    for var in obs:
        ncvar = rootgrp.createVariable(var.name, var.data_type, var.size, zlib=var.zlib, fill_value=var.fill_value)
        ncvar[:] = var.data
        ncvar.long_name = var.long_name
        if var.units : ncvar.units = var.units
        if var.error_variable : ncvar.error_variable = var.error_variable
        if var.bias_variable : ncvar.bias_variable = var.bias_variable
        if var.comment : ncvar.comment = var.comment
        if var.plot_range : ncvar.plot_range = var.plot_range
        if var.plot_scale : ncvar.plot_scale = var.plot_scale
        # iterate Dict of (possible) extra attributes
        if var.extra_attributes:
            for attr, value in var.extra_attributes.items():
                setattr(ncvar, attr, value)

# assign values -- from orignal files or measurements directly
len_time = 100
len_height = 10
nc_vars["time"][:] = np.arange(0,100)
nc_vars["unix_time"][:] = np.arange(0,100)
nc_vars["height"][:] = np.arange(0,10)
nc_vars["wind_speed"][:] = np.random.rand(len_time,len_height)

# set global attributes -- later on read from the database for site and date
rootgrp.title = "testi *.nc"
rootgrp.Convetions = "CF-1.7"
rootgrp.location = "toimisto"
rootgrp.institution = "FMI"
rootgrp.source = "polla"
rootgrp.history = "luotu omasta pollasta"
rootgrp.comment = "testi"

rootgrp.close()