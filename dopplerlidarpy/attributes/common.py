#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:48:23 2019

@author: manninan
"""


# Attributes common to all netcdf4 files written by the Doppler lidar toolbox
COMMON_ATTRIBUTES = {
    "time_unix": {
        "standard_name": "time_unix",
        "long_name": "Epoch time UTC",
        "units": "seconds since 00:00:00 UTC 1 January 1970",
        "dim_name": "time",
        "calendar": "gregorian",
        "comment": ""},
    "time": {
        "standard_name": "time_hours_utc",
        "long_name": "Time UTC",
        "units": "hours since midnight UTC",
        "dim_name": "time",
        "comment": ""},
    "range": {
        "standard_name": "range",
        "long_name": "Range from the instrument",
        "units": "m",
        "dim_name": "range",
        "comment": ""},
    "height_agl": {
        "standard_name": "height",
        "long_name": "Height above ground level",
        "units": "m",
        "dim_name": "range",
        "comment": ""},
    "altitude": {
        "standard_name": "altitude",
        "long_name": "instrument height above the geoid",
        "units": "m",
        "dim_name": (),  # () = scalar
        "comment": ""},
    "ground_altitude": {
        "standard_name": "ground_level_altitude",
        "long_name": "height above the geoid",
        "units": "m",
        "dim_name": (),  # () = scalar
        "comment": ""},
    "latitude": {
        "standard_name": "latitude",
        "long_name": "latitude in degrees north",
        "units": "degree north",
        "comment": "can have values from -180 to 180 degrees north",
        "dim_name": ()},  # () = scalar
    "grid_latitude": {
        "standard_name": "grid_latitude",
        "long_name": "latitude in degrees",
        "units": "degree",
        "comment": "can have values from 0 to 360 degrees",
        "dim_name": ()},  # () = scalar
    "longitude": {
        "standard_name": "longitude",
        "long_name": "longitude in degrees east",
        "units": "degree east",
        "comment": "can have values from -180 to 180 degrees east",
        "dim_name": ()},  # () = scalar
    "grid_longitude": {
        "standard_name": "grid_longitude",
        "long_name": "longitude in degrees",
        "units": "degree",
        "comment": "can have values from 0 to 360 degrees",
        "dim_name": ()}  # () = scalar
}

# XXXXXX = {
#    "standard_name": "",
#    "long_name": "",
#    "units": "",
#    "comment": "",
#    "dim_name": ""}
