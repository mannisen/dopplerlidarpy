#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:48:23 2019

@author: manninan
"""
import uuid
from netCDF4 import Dataset
from datetime import datetime
from dopplerlidarpy.utilities import dl_toolbox_version
import getpass


def write_nc_(date_txt, file_name, obs, add_gats=None, title_=None, institution_=None, location_=None, source_=None):
    """Writes a netCDF file with name and full path specified by 'file_name' with variables as listed by 'obs'.

    Args:
        date_txt (str):
        file_name (str):
        obs (list):
        add_gats (dict):
        title_ (str):
        institution_ (str):
        location_ (str):
        source_ (str):

    """

    rootgrp = Dataset(file_name, 'w', format='NETCDF4_CLASSIC')

    dimension_names = []
    for var in obs:
        print("..variable '{}'".format(var.name))
        if var.dim_name is not None:
            for i in range(len(var.dim_name)):
                if not var.dim_name[i] in dimension_names:
                    dimension_names.append(var.dim_name[i])
                    rootgrp.createDimension(var.dim_name[i], var.dim_size[i])
    
        # Create netcdf variable
        if var.dim_name is None:  # dimensionless
            ncvar = rootgrp.createVariable(var.name,
                                           var.data_type,
                                           zlib=var.zlib,
                                           fill_value=var.fill_value)
        else:
            ncvar = rootgrp.createVariable(var.name,
                                           var.data_type,
                                           var.dim_name,
                                           zlib=var.zlib,
                                           fill_value=var.fill_value)

        ncvar[:] = var.data
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
    rootgrp.year = int(date_txt[0:4])
    rootgrp.month = int(date_txt[4:6])
    rootgrp.day = int(date_txt[6:])
    rootgrp.software_version = dl_toolbox_version.__version__
#    rootgrp.git_version = git_version()
    rootgrp.file_uuid = str(uuid.uuid4().hex)
    rootgrp.references = ''
    user_name = getpass.getuser()
    now_time_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    history_msg = "NETCDF4 file created by user {} on {} UTC.".format(user_name, now_time_utc)
    rootgrp.history = history_msg
    # Additional global attributes
    for key_, value_ in zip(add_gats.keys(), add_gats.values()):
        setattr(rootgrp, key_, value_)
    rootgrp.close()
    print(history_msg)
