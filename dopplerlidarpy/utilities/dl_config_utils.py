#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:17:01 2019

@author: manninan
"""
from dopplerlidarpy.utilities import time_utils as tu
from dopplerlidarpy.utilities import general_utils as gu
from dopplerlidarpy.utilities.input_checks import validate_site
from dopplerlidarpy.utilities import input_checks as ic
import os


class ValidationError(Exception):
    """

    """
    def __init__(self, message):
        super().__init__(message)


class ConfigError(Exception):
    """

    """
    def __init__(self, message):
        super().__init__(message)


def get_dl_config(args):
    """Reads config parameters from a database or from a text file.
        
    Args:
        args (class): A populated namespace object

    Returns:
        config_info: (dict) Config parameters for the site

    """
    
    # Check if inputs are valid
    validate_site(args.site[0])

    dl_in = ic.get_dl_input_list()
    if args.i is not None and args.i in dl_in["instruments"]:
        path_to_config = "./" + args.i + "_config"
    else:
        path_to_config = "./halo_config"
    if os.path.isfile(path_to_config + ".txt"):
        config_info = get_dl_config_txt(args)
    elif os.path.isfile(path_to_config + ".db"):
        config_info = get_dl_config_db(args)
    else:
        raise ValidationError("No 'halo_config.txt' or 'halo_config.db' file found.")

    return config_info


def get_dl_config_db(args):
    """Reads the halo_config.db into a dictionary.

    Args:
        args (class): A populated namespace object

    Returns:
        config_info: (dict) Config parameters for the site

    """
    # Check if inputs are valid
    validate_site(args.site[0])

    config_info = {}
    return config_info


def get_dl_config_txt(args):
    """Reads the halo_config.txt into a dictionary.

    Args:
        args (class): A populated namespace object

    Returns:
        config_info: (dict) Config parameters for the site

    """

    # Check if inputs are valid
    validate_site(args.site)

    # open the file for reading
    file_handle = open('halo_config.txt', 'r')
    tmp_default_keys = []
    tmp_default_values = []
    tmp_site_keys = []
    tmp_site_values = []
    flag_site = False
    flag_defaults = True
    while True:
        # read a single line
        line = file_handle.readline()
        if not line:
            break
        # Skip comment lines
        if not(line.__contains__('DEFAULTS') or line.__contains__('SITE SPECIFIC')):
            # Detect when default parameters end
            if line.__contains__('site = '):
                flag_defaults = False
            if flag_defaults:
                # Default parameters
                tmp_default_keys.append(line[0:line.find("=")-1].replace("\n", ""))
                tmp_default_values.append(line[line.find("=")+2:len(line)].replace("\n", ""))
            if flag_site:
                # Site specific parameters
                tmp_site_keys.append(line[0:line.find("=")-1].replace("\n", ""))
                tmp_site_values.append(line[line.find("=")+2:len(line)].replace('\n', ""))
            # Detect when the section for the site in question ends
            if flag_site and line.__contains__("site = "):
                flag_site = False
            # Detect when the section for the site in question begins
            if line == "site = {}\n".format(args.site):
                flag_site = True

    # Close file
    file_handle.close()

    # Give error if site not found
    if not tmp_site_keys:
        raise ConfigError("No parameters found for site '{}' from halo_config.txt".format(args.site))

    # Filter empty elements
    tmp_site_keys = list(filter(None, tmp_site_keys))
    tmp_site_values = list(filter(None, tmp_site_values))

    # Parse redefined parameters from the halo_config.txt
    ival_begin = gu.look_for_from('parameters_valid_from_including', tmp_site_keys)

    # Initialize
    config_info = dict()
    config_info["default_parameters"] = {}
    config_info["site_parameters"] = {}

    # Get default parameters
    for key, value in zip(tmp_default_keys, tmp_default_values):
        config_info["default_parameters"][key] = value

    # Site parameters
    for i in range(len(ival_begin)):
        # Convert to posix time
        valid_posix = tu.date_txt2epoch(tmp_site_values[ival_begin[i]], '%Y-%m-%d_%H:%M:%S')

        # Set mark to separate the periods when parameters change, if there are any
        if len(ival_begin) > 1 and not i:
            mark = ival_begin[i+1]
        else:
            mark = len(tmp_site_keys)

        # Get site and date specific parameters
        valid_date_txt = tu.epoch2date_txt(valid_posix, "%Y-%m-%d_%H:%M:%S")
        config_info["site_parameters"][valid_date_txt] = dict()
        for key, value in zip(tmp_site_keys[:mark], tmp_site_values[:mark]):
            config_info["site_parameters"][valid_date_txt][key] = value

    return config_info
