#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:48:23 2019

@author: manninan
"""

# Attributes for netcdf4 product files written by the Doppler lidar toolbox
# If new variables is added, please provide at least these:
# variable name, and attributes:
#   standard_name, based on CF 1.7 conventions
#   long_name, a more descriptive name
#   units, SI units
#   dim_name, for scalar - "dim_name": ()
#               for 1D variable - "dim_name": "time" or "dim_name": "range"
#               for 2D variable - "dim_name": ("time", "range")
#               for nD variable - "dim_name": ("time", "range", "foobar", ..., "month")
PRODUCT_ATTRIBUTES = {
    "wind_speed": {
        "standard_name": "wind_speed",
        "long_name": "wind speed",
        "units": "m s-1",
        "comment": "speed of two-dimensional air velocity vector",
        "dim_name": ("time", "range")},
    "wind_speed_precision": {
        "standard_name": "wind_speed_precision",
        "long_name": "precision of wind speed retrieval",
        "units": "m s-1",
        "comment": "See Eq. (8) in doi:10.5194/amt-10-1229-2017",
        "dim_name": ("time", "range")},
    "wind_speed_instrumental_precision": {
        "standard_name": "wind_speed_instrumental_precision",
        "long_name": "instrumental precision propagated into wind speed retrieval",
        "units": "m s-1",
        "comment": "See Eq. (10-11) in doi:10.5194/amt-8-2251-2015, and Eq. (8) in doi:10.5194/amt-10-1229-2017.",
        "dim_name": ("time", "range")},
    "wind_direction": {
        "standard_name": "wind_from_direction",
        "long_name": "direction from wind is blowing ",
        "units": "degree",
        "comment": "meteorological convention",
        "dim_name": ("time", "range")},
    "wind_direction_precision": {
        "standard_name": "wind_from_direction_precision",
        "long_name": "precision of wind direction retrieval",
        "units": "degree",
        "comment": "See Eq. (9) in doi:10.5194/amt-10-1229-2017",
        "dim_name": ("time", "range")},
    "wind_direction_instrumental_precision": {
        "standard_name": "wind_from_direction_instrumental_precision",
        "long_name": "instrumental precision propagated into wind direction retrieval",
        "units": "degree",
        "comment": "See Eq. (10-11) in doi:10.5194/amt-8-2251-2015, and Eq. (9) in doi:10.5194/amt-10-1229-2017.",
        "dim_name": ("time", "range")},
    "u": {
        "standard_name": "eastward_wind",
        "long_name": "Eastward wind vector component",
        "units": "m s-1",
        "comment": "Indicates a vector component which is positive when directed eastward (negative westward).",
        "dim_name": ("time", "range")},
    "u_precision": {
        "standard_name": "eastward_wind_precision",
        "long_name": "Precision of Eastward wind vector component retrieval",
        "units": "m s-1",
        "comment": "See Eq. (7) in doi:10.5194/amt-10-1229-2017",
        "dim_name": ("time", "range")},
    "u_instrumental_precision": {
        "standard_name": "eastward_wind_instrumental_precision",
        "long_name": "Instrumental precision propagated into Eastward wind vector component retrieval",
        "units": "m s-1",
        "comment": "See Eq. (10-11) in doi:10.5194/amt-8-2251-2015, and Eq. (7) in doi:10.5194/amt-10-1229-2017.",
        "dim_name": ("time", "range")},
    "v": {
        "standard_name": "northward_wind",
        "long_name": "Northward wind vector component",
        "units": "m s-1",
        "comment": "Indicates a vector component which is positive when directed northward (negative southward).",
        "dim_name": ("time", "range")},
    "v_precision": {
        "standard_name": "northward_wind_precision",
        "long_name": "Precision of Northward wind vector component retrieval",
        "units": "m s-1",
        "comment": "See Eq. (7) in doi:10.5194/amt-10-1229-2017",
        "dim_name": ("time", "range")},
    "v_instrumental_precision": {
        "standard_name": "northward_wind_instrumental_precision",
        "long_name": "Instrumental precision propagated into Northward wind vector component retrieval",
        "units": "m s-1",
        "comment": "See Eq. (10-11) in doi:10.5194/amt-8-2251-2015, and Eq. (7) in doi:10.5194/amt-10-1229-2017.",
        "dim_name": ("time", "range")},
    "w": {
        "standard_name": "upward_air_velocity",
        "long_name": "Upward wind vector component",
        "units": "m s-1",
        "comment": "Indicates a vector component which is positive when directed upward (negative downward).",
        "dim_name": ("time", "range")},
    "w_precision": {
        "standard_name": "upward_air_velocity_precision",
        "long_name": "Precision of upward wind vector component retrieval",
        "units": "m s-1",
        "comment": "See Eq. (7) in doi:10.5194/amt-10-1229-2017",
        "dim_name": ("time", "range")},
    "w_instrumental_precision": {
        "standard_name": "northward_wind_instrumental_precision",
        "long_name": "Instrumental precision propagated into upward wind vector component retrieval",
        "units": "m s-1",
        "comment": "See Eq. (10-11) in doi:10.5194/amt-8-2251-2015, and Eq. (7) in doi:10.5194/amt-10-1229-2017.",
        "dim_name": ("time", "range")},

}


# XXXXXX = {
#    "standard_name": "",
#    "long_name": "",
#    "units": "",
#    "comment": "",
#    "dim_name": ""}
