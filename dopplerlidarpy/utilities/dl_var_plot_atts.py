#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:08:40 2019

@author: manninan
"""

import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from dopplerlidarpy.utilities import dl_var_atts as vatts

# Defaults
_NORM = None
_CEXT = "both"
_CMAP = plt.get_cmap('pyart_HomeyerRainbow')
_VMIN = 0
_VMAX = 1


# Generates a blueprint for a variable plot attributes, when called
class VarPlotBlueprint:
    """Blueprint of variable attributes for plotting with pyplot"""

    # Initialize with defaults
    def __init__(self, norm=_NORM, cextend=_CEXT, cmap=_CMAP, vmin=_VMIN, vmax=_VMAX):
        self.norm = norm
        self.cextend = cextend
        self.cmap = cmap
        self.vmin = vmin
        self.vmax = vmax


# Generate plot attributes for Doppler lidar attributes. Only the ones that differ from defaults need to be specified!
def wind_speed_():
    var = vatts.wind_speed_()  # get default variable attributes
    var.cextend = "max"
    var.vmax = 20  # m s-1

    return VarPlotBlueprint(var)


def wind_speed_error_instrumental_():
    var = vatts.wind_speed_error_instr_()  # get default variable attributes
    var.cextend = "max"

    return VarPlotBlueprint(var)


def wind_speed_error_():
    var = vatts.wind_speed_error_()  # get default variable attributes
    var.cextend = "max"

    return VarPlotBlueprint(var)


def wind_direction_():
    var = vatts.wind_dir_()  # get default variable attributes
    var.vmax = 360  # degrees from North
    var.cmap = plt.get_cmap('pyart_erdc_iceFire')  # cyclic colormap

    return VarPlotBlueprint(var)


def wind_direction_error_instrumental_():
    var = vatts.wind_dir_error_instr_()  # get default variable attributes
    var.cextend = "max"

    return VarPlotBlueprint(var)


def wind_direction_error_():
    var = vatts.wind_dir_error_()  # get default variable attributes
    var.cextend = "max"

    return VarPlotBlueprint(var)


def signal_():
    return VarPlotBlueprint("signal0",
                             long_name="uncorrected signal",
                             units="SNR+1",
                                vmin=.99,
                             vmax=1.02,
                             ylabel="range (km)")


def signal0_error_():
    return VarPlotBlueprint("signal0_error",
                             long_name="uncorrected signal instrumental uncertainty",
                             units="%",
                             norm=mcolors.LogNorm(vmin=1, vmax=400),
                             ylabel="range (km)")


def radial_velo_():
    return VarPlotBlueprint("radial_velocity",
                             long_name="radial velocity",
                             units="m s-1",
                             vmin=-2,
                             vmax=2,
                             cmap=plt.get_cmap('pyart_balance'))


def radial_velo_():
    return VarPlotBlueprint("radial_velocity_error",
                             long_name="radial velocity instrumental uncertainty",
                             units="m s-1",
                             norm=mcolors.LogNorm(vmin=.001, vmax=10))


def epsilon_():
    return VarPlotBlueprint("epsilon",
                             long_name="TKE dissipation rate",
                             units="m2 s-3",
                             norm=mcolors.LogNorm(vmin=1e-6, vmax=1e-1))


def epsilon_error_():
    return VarPlotBlueprint("epsilon_error",
                             long_name="TKE dissipation rate retrieval uncertainty",
                             units="m s-1",
                             norm=mcolors.LogNorm(vmin=.001, vmax=10))


def wind_shear_():
    return VarPlotBlueprint("vector_wind_shear",
                             long_name="vector wind shear",
                             units="m2 s-3",
                             norm=mcolors.LogNorm(vmin=1e-6, vmax=1e-1))


def wind_shear_error_():
    return VarPlotBlueprint("vector_wind_shear_error",
                             long_name="vector wind shear retrieval uncertainty",
                             units="m s-1",
                             norm=mcolors.LogNorm(vmin=.001, vmax=10))





if __name__ == "__main__":
    print("'dl_attributes' can only be imported, not called directly.")
#     # stuff only to run when not called via 'import' here
#     def __init__(self, name, data=None, dim_name=("time", "range"), dim_size=None,
#                  data_type="f8", zlib=False, fill_value=True, standard_name="",
#                  long_name="", units="", units_html="", comment="", plot_scale=None,
#                  plot_range=None, bias_variable=None, error_variable=None,
#                  extra_attributes=None, calendar=None):
#         self.name = name
#         self.data = data
#         self.data_type = data_type
#         self.zlib = zlib
#         self.standard_name = standard_name
#         self.long_name = long_name
#         self.units = units
#         self.units_html = units_html
#         self.comment = comment
#         self.plot_scale = plot_scale
#         self.plot_range = plot_range
#         self.extra_attributes = extra_attributes
#         self.calendar = calendar
#         # Dimension names
#         if (dim_name != None and type(dim_name) != tuple):
#             raise NetcdfVariableError("'dim_name' has to be given as a tuple of strings.")
#         elif (type(dim_name) == tuple and
#               (type(dim_name[0]) != str or type(dim_name[1]) != str)):
#             raise NetcdfVariableError("'dim_name' has to be given as a tuple of strings.")
#         else:
#             self.dim_name = dim_name
#         # Dimension sizes
#         if (dim_size != None and type(dim_size) != tuple):
#             raise NetcdfVariableError("'dim_size' has to be given as a tuple object.")
#         else:
#             self.dim_size = dim_size
#
#         # bias variable:
#         if (bias_variable and type(bias_variable) == bool):  # True
#             self.bias_variable = name + '_bias'
#         else:
#             self.bias_variable = bias_variable
#         # error variable:
#         if (error_variable and type(error_variable) == bool):  # True
#             self.error_variable = name + '_error'
#         else:
#             self.error_variable = error_variable
#         # fill value:
#         if (fill_value and type(fill_value) == bool):  # True
#             self.fill_value = netCDF4.default_fillvals[data_type]
#         else:
#             self.fill_value = fill_value
