#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:48:23 2019

@author: manninan
"""

from netCDF4 import Dataset
from datetime import datetime
from dopplerlidarpy.utilities import dl_toolbox_version
from dopplerlidarpy.utilities import general_utils as gu
import getpass
import uuid


def copy_nc_attributes(obj_from, obj_to, att_names):
    for n in att_names:
        if hasattr(obj_from, n) and getattr(obj_from, n) is not None:
            v = getattr(obj_from, n)
            Dataset.setncattr(obj_to, n, v)


def read_nc_fields(nc_file, names):
    """Reads selected attributes from a netCDF file.
    Args:
        nc_file (str): full path to netCDF file, e.g. "~/data/product/20201231_dl_measurement.nc"
        names (str/list): Variables to be read, e.g. "temperature" or
            ["ldr", "lwp"].
    Returns:
        ndarray/list: Array in case of one variable passed as a string.
        List of arrays otherwise.
    """
    names = [names] if isinstance(names, str) else names
    nc = Dataset(nc_file)
    data = [nc.variables[name][:] for name in names]
    nc.close()
    return data[0] if len(data) == 1 else data


def create_nc_dim(rootgrp, dim_name, dim_size=None):
    """ Creates a new dimension 'dim_name' to 'rootgrp' instance. If exists, do nothing.

    Args:
        rootgrp (Dataset): collection of dimensions, groups, attributes and attributes.
        dim_name (str): name of the new dimension
        dim_size (int): size of the new dimension

    Returns:
        rootgrp (Dataset): collection of dimensions, groups, attributes and attributes.

    """
    if dim_name not in list(rootgrp.dimensions.keys()):
        return rootgrp.createDimension(dim_name, dim_size)


def create_nc_var(rootgrp, var):
    ncvar = rootgrp.createVariable(var.name, var.data_type, var.dim_name)
    return rootgrp, ncvar


def write_nc_(date_txt, file_name, obs, additional_gatts=None, title_="", institution_="", location_="", source_=""):
    """Writes a netCDF file with name and full path specified by 'file_name' with attributes as listed by 'obs'.

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

    rootgrp = Dataset(file_name, mode='w', format='NETCDF4', clobber=True)

    dimension_names = []
    for var in obs:
        if var.dim_name is not None:
            if isinstance(var.dim_name, tuple):
                for dname, dsize in zip(var.dim_name, var.dim_size):
                    if dname not in dimension_names:
                        print("Adding dimension {}".format(dname))
                        dimension_names.append(dname)
                        rootgrp.createDimension(dname, dsize)
            elif isinstance(var.dim_name, str):
                if var.dim_name not in dimension_names:
                    print("Adding dimension {}".format(var.dim_name))
                    dimension_names.append(var.dim_name)
                    rootgrp.createDimension(var.dim_name, var.dim_size[0])
            else:
                raise
        print("Adding variable {}".format(var.variable_name))
        ncvar = rootgrp.createVariable(var.variable_name, var.data_type, var.dim_name)

        # Copy attributes
        # copy_nc_attributes(var, ncvar, list(var.__dict__.keys()))

        attrs = list(var.__dict__.keys())
        for attr in attrs:
            if attr is "extra_attributes":
                if var.extra_attributes is not None:
                    for attr_extra, value_extra in var.extra_attributes.items():
                        value_extra = getattr(var, attr_extra)
                        if value_extra is not None:
                            setattr(ncvar, attr_extra, value_extra)
            elif attr is not "data":
                value = getattr(var, attr)
                if value is not None:
                    setattr(ncvar, attr, value)
        ncvar[:] = var.data

    # global attributes:
    print("Adding global attributes")
    rootgrp.Conventions = 'CF-1.7'
    rootgrp.title = title_
    rootgrp.institution = institution_
    rootgrp.location = location_
    rootgrp.source = source_
    rootgrp.year = int(date_txt[:4])
    rootgrp.month = int(date_txt[4:6])
    rootgrp.day = int(date_txt[6:8])
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
