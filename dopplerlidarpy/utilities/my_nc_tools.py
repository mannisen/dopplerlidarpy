#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:48:23 2019

@author: manninan
"""

from netCDF4 import Dataset
from datetime import datetime
import numpy as np
from dopplerlidarpy.utilities import dl_toolbox_version
import getpass
import uuid


def read_nc_fields(nc_file, names):
    """Reads selected variables from a netCDF file.
    Args:
        nc_file (str): netCDF file name.
        names (str/list): Variables to be read, e.g. 'temperature' or
            ['ldr', 'lwp'].
    Returns:
        ndarray/list: Array in case of one variable passed as a string.
        List of arrays otherwise.
    """
    names = [names] if isinstance(names, str) else names
    nc = Dataset(nc_file)
    data = [nc.variables[name][:] for name in names]
    nc.close()
    return data[0] if len(data) == 1 else data


def create_nc_dims(rootgrp, var, dimension_names):
    if var.dim_name is not None:
        for dname, dsize in zip(var.dim_name, var.dim_size):
            if dname not in dimension_names:
                print(dname)
                dimension_names.append(dname)
                rootgrp.createDimension(dname, dsize)
    return rootgrp, dimension_names


def create_nc_var(rootgrp, var):
    ncvar = rootgrp.createVariable(var.name, var.data_type, var.dim_name)
    return rootgrp, ncvar


def write_nc_(date_txt, file_name, obs, additional_gatts=None, title_=None, institution_=None, location_=None, source_=None):
    """Writes a netCDF file with name and full path specified by 'file_name' with variables as listed by 'obs'.

    Args:
        date_txt (str):
        file_name (str):
        obs (list):
        additional_gatts (dict):
        title_ (str):
        institution_ (str):
        location_ (str):
        source_ (str):

    """

    rootgrp = Dataset(file_name, 'w', format='NETCDF4')

    dimension_names = []
    for var in obs:
        if var.dim_name is not None:
            for dname, dsize in zip(var.dim_name, var.dim_size):
                if dname not in dimension_names:
                    print(dname)
                    dimension_names.append(dname)
                    rootgrp.createDimension(dname, dsize)
        ncvar = rootgrp.createVariable(var.name, var.data_type, var.dim_name)

        # rootgrp, dimension_names = create_nc_dims(rootgrp, var, dimension_names)
        # rootgrp, ncvar = create_nc_var(rootgrp, var)

        # Create netcdf variable
        #print("..variable '{}'".format(var.name))
        # if var.dim_name is not None:
        #     for dname, dsize in zip(var.dim_name, var.dim_size):
        #         if dname not in dimension_names:
        #             dimension_names.append(dname)
        #             rootgrp.createDimension(dname, dsize)
        #     ncvar = rootgrp.createVariable(var.name, var.data_type, var.dim_name)
        # else:  # dimensionless
        #     ncvar = rootgrp.createVariable(var.name, var.data_type)
        # print("ncvar {}".format(np.shape(ncvar)))
        # print("var.data {}".format(np.shape(var.data)))

        ncvar[:] = var.data[:]
        ncvar.long_name = var.long_name

        if var.units:
            ncvar.units = var.units
        if var.error_variable:
            ncvar.error_variable = var.error_variable
        if var.bias_variable:
            ncvar.bias_variable = var.bias_variable
        if var.comment:
            ncvar.comment = var.comment
        if var.plot_range:
            ncvar.plot_range = var.plot_range
        if var.plot_scale:
            ncvar.plot_scale = var.plot_scale
        # iterate Dict of (possible) extra attributes
        if var.extra_attributes:
            for attr, value in var.extra_attributes.items():
                setattr(ncvar, attr, value)

    # global attributes:
    print("..global attributes")
    rootgrp.Conventions = 'CF-1.7'
    rootgrp.title = title_
    rootgrp.institution = institution_
    rootgrp.location = location_
    rootgrp.source = source_
    rootgrp.year = int(date_txt[:4])
    rootgrp.month = int(date_txt[5:7])
    rootgrp.day = int(date_txt[8:10])
    rootgrp.software_version = dl_toolbox_version.__version__
#    rootgrp.git_version = git_version()
    rootgrp.file_uuid = str(uuid.uuid4().hex)
    rootgrp.references = ''
    user_name = getpass.getuser()
    now_time_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    history_msg = "NETCDF4 file created by user {} on {} UTC.".format(user_name, now_time_utc)
    rootgrp.history = history_msg
    # Additional global attributes
    if additional_gatts is not None:
        for key_, value_ in zip(additional_gatts.keys(), additional_gatts.values()):
            setattr(rootgrp, key_, value_)
    rootgrp.close()
    print(history_msg)
