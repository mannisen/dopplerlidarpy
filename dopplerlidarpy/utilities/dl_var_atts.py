#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:08:40 2019

@author: manninan
"""
import netCDF4
from dopplerlidarpy.attributes.common import COMMON_ATTRIBUTES
from dopplerlidarpy.attributes.products import PRODUCT_ATTRIBUTES


class NetcdfAttributeError(Exception):
    def __init__(self, message):
        super().__init__(message)


def validate_dims(dim_att=None, type_=None):
    if dim_att is not None and type_ is not None:  # not empty
        if type_ != str or type_ != int:
            raise NetcdfAttributeError("")
        if type(dim_att) != tuple:  # not tuple
            raise NetcdfAttributeError("Input has to be given as a tuple of strings.")
        else:  # is tuple
            if len(dim_att) == 1:  # one dimension
                if type_ == str and type(dim_att) != str:  # is string
                    raise NetcdfAttributeError("Input has to be given as a tuple of strings.")
                elif type_ == int and type(dim_att) != int:  # in int
                    raise NetcdfAttributeError("Input has to be given as a tuple of integers.")
            else:  # two dimensions
                if type_ == str and type(dim_att[0]) != str or type(dim_att[1]) != str:
                    raise NetcdfAttributeError("Input has to be given as a tuple of strings.")
                elif type_ == int and type(dim_att[0]) != int or type(dim_att[1]) != int:
                    raise NetcdfAttributeError("Input has to be given as a tuple of integers.")


class VarBlueprint:
    """Blueprint for Cloudnet-type Doppler lidar netcdf variable attributes"""

    def __init__(self, variable_name, data=None, dim_name=None, dim_size=None,
                 data_type="f8", zlib=False, fill_value=True, standard_name=None,
                 long_name=None, units=None, units_html=None, comment=None,
                 error_variable=None, bias_variable=None,
                 plot_scale=None, plot_range=None,
                 extra_attributes=None, calendar=None):
        self.variable_name = variable_name
        self.data = data
        self.data_type = data_type
        #self.zlib = zlib
        self.standard_name = standard_name
        self.long_name = long_name
        self.units = units
        self.units_html = units_html
        self.comment = comment
        self.plot_scale = plot_scale
        self.plot_range = plot_range
        self.extra_attributes = extra_attributes
        self.calendar = calendar
        self.error_variable = error_variable
        self.bias_variable = bias_variable
        try:
            validate_dims(dim_name)
        except NetcdfAttributeError:
            raise
        self.dim_name = dim_name
        try:
            validate_dims(dim_size)
        except NetcdfAttributeError:
            raise
        self.dim_size = dim_size
        # fill value:
        if fill_value and type(fill_value) == bool:  # True
            self.fill_value = netCDF4.default_fillvals[data_type]
        else:
            self.fill_value = fill_value


def dl_var_atts(var_name, data=None, dim_size=None):
    if var_name not in list_variables():
        raise Exception("Unknown variable name {:s}".format(var_name))
    else:
        if var_name in COMMON_ATTRIBUTES:
            att = COMMON_ATTRIBUTES
        elif var_name in PRODUCT_ATTRIBUTES:
            att = PRODUCT_ATTRIBUTES
        else:
            raise Exception("Unknown variable name {:s}".format(var_name))

        #elif var_name in LEVEL3_ATTRIBUTES:
        #    att = LEVEL3_ATTRIBUTES
        #elif var_name in DL_ATTRIBUTES:
        #    att = DL_ATTRIBUTES

        return VarBlueprint(var_name,
                            standard_name=att[var_name]["standard_name"],
                            long_name=att[var_name]["long_name"],
                            units=att[var_name]["units"],
                            comment=att[var_name]["comment"],
                            data=data,
                            dim_name=att[var_name]["dim_name"],
                            dim_size=dim_size)


def fill_in_atts(ncvar, atts):
    ncvar.standard_name = atts.standard_name
    ncvar.long_name = atts.long_name
    ncvar.units = atts.units
    ncvar.comment = atts.comment
    return ncvar


def list_variables():
    return list(COMMON_ATTRIBUTES.keys()) + list(PRODUCT_ATTRIBUTES.keys())  # + list(LEVEL3_ATTRIBUTES.keys()) + list(DL_ATTRIBUTES)


if __name__ == "__main__":
    print("'dl_attributes' can only be imported, not called directly.")
