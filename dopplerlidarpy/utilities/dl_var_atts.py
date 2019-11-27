#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:08:40 2019

@author: manninan
"""
import netCDF4


# from utilities.my_nc_write_tools import var_blueprint


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
    """Blueprint for Cloudnet-type netcdf variables"""

    def __init__(self, name, data=None, dim_name=None, dim_size=None,
                 data_type="f8", zlib=False, fill_value=True, standard_name="",
                 long_name="", units="", units_html="", comment="",
                 plot_scale=None, plot_range=None, bias_variable=None,
                 error_variable=None, extra_attributes=None, calendar=None):
        self.name = name
        self.data = data
        self.data_type = data_type
        self.zlib = zlib
        self.standard_name = standard_name
        self.long_name = long_name
        self.units = units
        self.units_html = units_html
        self.comment = comment
        self.plot_scale = plot_scale
        self.plot_range = plot_range
        self.extra_attributes = extra_attributes
        self.calendar = calendar
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
        # bias variable:
        if bias_variable and type(bias_variable) == bool:  # True
            self.bias_variable = name + '_bias'
        else:
            self.bias_variable = bias_variable
        # error variable:
        if error_variable and type(error_variable) == bool:  # True
            self.error_variable = name + '_error'
        else:
            self.error_variable = error_variable
        # fill value:
        if fill_value and type(fill_value) == bool:  # True
            self.fill_value = netCDF4.default_fillvals[data_type]
        else:
            self.fill_value = fill_value


def wind_speed_(data=None, dim_name=("unix_time", "range"), dim_size=None):
    return VarBlueprint("wind_speed",
                        standard_name="wind_speed",
                        long_name="Wind speed",
                        units="m s-1",
                        plot_range=[0, 20],
                        plot_scale="linear",
                        units_html="m s<sup>-1</sup>",
                        data=data,
                        dim_name=dim_name,
                        dim_size=dim_size)


def wind_direction_(data=None, dim_name=("unix_time", "range"), dim_size=None):
    return VarBlueprint("wind_direction",
                        standard_name="wind_from_direction",
                        long_name="Wind direction",
                        units="m s-1",
                        plot_range=[0, 360],
                        plot_scale="linear",
                        comment="Meteorological convention, the direction the wind is blowing from",
                        units_html="m s<sup>-1</sup>",
                        data=data,
                        dim_name=dim_name,
                        dim_size=dim_size)


def unix_time_(data=None, dim_name=("unix_time",), dim_size=None):
    return VarBlueprint("unix_time",
                        standard_name="unix_time",
                        long_name="UNIX Epoch time (seconds since 1970-01-01 00:00:00)",
                        units="seconds",
                        calendar="gregorian",
                        data=data,
                        dim_name=dim_name,
                        dim_size=dim_size)


def time_hrs_utc_(data=None, dim_name=("unix_time",), dim_size=None):
    return VarBlueprint("time_hrs_utc",
                        standard_name="time_hrs_utc",
                        long_name="Decimal hours since midnight UTC",
                        units="hours",
                        calendar="gregorian",
                        data=data,
                        dim_name=dim_name,
                        dim_size=dim_size)


def height_agl_(data=None, dim_name=("range",), dim_size=None):
    return VarBlueprint("height_agl",
                        standard_name="height_agl",
                        long_name="Height above ground level",
                        units="m",
                        comment="Not the same as range",
                        data=data,
                        dim_name=dim_name,
                        dim_size=dim_size)


def range_(data=None, dim_name=("range",), dim_size=None):
    return VarBlueprint("range",
                        standard_name="range",
                        long_name="Range from the instrument",
                        units="m",
                        comment="Not the same as height",
                        data=data,
                        dim_name=dim_name,
                        dim_size=dim_size)


if __name__ == "__main__":
    print("'dl_attributes' can only be imported, not called directly.")
