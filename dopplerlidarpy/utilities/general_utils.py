#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:54:08 2019

@author: manninan
"""
from dopplerlidarpy.utilities import dl_config_utils as cu
from dopplerlidarpy.utilities import time_utils as tu
from datetime import datetime
import numpy as np
import re
import os


def normalize_between(values, actual_bounds, desired_bounds):
    """

    Args:
        values:
        actual_bounds:
        desired_bounds:

    Returns:

    """
    return [desired_bounds[0] +
            (x - actual_bounds[0]) *
            (desired_bounds[1] - desired_bounds[0]) /
            (actual_bounds[1] - actual_bounds[0]) for x in values]


def find_nearest(a, b):
    """Find nearest values of 'b' from 'a', adapted from HALO_lidar_toolbox MATLAB function look4nearest.m

    Args:
        a (ndarray): look from
        b (ndarray): look for

    Returns:
        ib (ndarray): nearest indices
        ab (ndarray): nearest values

    """
    a = np.asarray(a)
    b = np.asarray(b)
    p = np.arange(0, len(b), 1)
    c = np.sort(b)
    ref = np.hstack((np.double(-np.inf), (c[:-1] + c[1:]) / 2, np.double(np.inf)))
    ic = np.digitize([a], ref).squeeze() - 1
    ib = p[ic]
    ab = b[ib]

    return ib.astype(int), ab.astype(float)


def look_for_from(look_for, look_from):
    """Looks for a string from list of strings.

    Args:
        look_for (str): string to look for
        look_from (list): list of strings (str) to look from

    Returns:
        indices (list): indices (int) where found

    """

    try:
        isinstance(look_from, list)
    except False:
        raise ValueError('First input must be a list (of strings).')
    try:
        isinstance(look_for, str)
    except False:
        raise ValueError('Second input must a string.')
        
    indices = [i for i, x in enumerate(look_from) if look_for in x]
    
    return indices


def camel2snake(str_in):
    """Converts str from CamelFormat to snake_format.

    Args:
        str_in: (str) String in CamelFormat

    Returns:
        str_out: (str) String in snake_format

    """

    words = str_in.splitlines()

    reg_ = r"(.+?)([A-Z])"

    def snake(match):
        return match.group(1).lower() + "_" + match.group(2).lower()

    results = [re.sub(reg_, snake, w, 0) for w in words]
    str_out = results[0]

    return str_out


def list2str(in_list):
    """Converts list of strings to string preserving quotation marks
    """
    ret_list = "'" + "', '".join(in_list) + "'"

    return ret_list


def get_dl_file_list(args):
    """Retrieves list of files with full paths according to given inputs and as specified in the config file.

    Args:
        args (class): A populated namespace object

    Returns:
        files_info (dict): Dictionary containing
                            - full paths to files matching the inputs (list)
                            - number of files total (int)
                            - number of full days when data is available (int)

    """

    if args.pol is not None:
        last_part = "_" + args.pol
    elif args.e is not None:
        last_part = "_ele" + str(args.e)
    elif args.a is not None:
        last_part = "_azi" + str(args.a)
    elif args.b is not None:
        last_part = "_" + str(args.b) + "beams"
    else:
        last_part = None

    # Get config info on the site
    config_info = cu.get_dl_config(args)
    valid_dates = list(config_info["site_parameters"].keys())

    # Get date format and convert to epoch time
    valid_format = tu.check_date_format(valid_dates[0])
    date_format = tu.check_date_format(args.start_date)
    start_date_epoch = tu.date_txt2epoch(args.start_date, date_format)
    valid_date_0_epoch = tu.date_txt2epoch(valid_dates[0], valid_format)

    # Check that parameters are valid for the start_date
    try:
        start_date_epoch >= valid_date_0_epoch
    except False:
        ValueError("Parameters are not valid for '{}' earlier than '{}'.".format(args.site, valid_dates[0]))

    # Get paths to files while taking into account the parameters_valid_from_including from config file
    start_date = datetime.strptime(args.start_date, date_format).date()

    full_paths = []
    file_names = []
    number_of_days = 0

    for i in range(len(valid_dates)):

        # Extract valid site parameters
        sp = config_info["site_parameters"][valid_dates[i]]

        # In case paths changes over the date range
        if len(valid_dates) > 1 and not i:
            end_date = datetime.strptime(valid_dates[i+1], valid_format).date()  # - timedelta(seconds=1)
        else:
            end_date = datetime.strptime(args.end_date, date_format).date()

        # Iterate for extracting the paths from the config file
        for time_step in tu.daterange(start_date, end_date):
            year_ = time_step.strftime("%Y")
            month_ = time_step.strftime("%m")
            day_ = time_step.strftime("%d")

            # abc = "dir_" + args.processing_level + "_" + args.observation_type + "_"
            if last_part is not None:
                path_ = sp["dir_" + args.processing_level + "_" + args.observation_type + last_part]
            else:
                path_ = sp["dir_" + args.processing_level + "_" + args.observation_type]

            # Parse the correct date to the path string
            path_ = path_.replace("+YYYY+", year_)
            path_ = path_.replace("+mm+", month_)
            path_ = path_.replace("+dd+", day_)

            # Determine the file type
            if args.processing_level == "raw":
                file_type = "." + args.observation_type
            else:
                file_type = ".nc"

            # List files and get only files for which parameters are valid
            for entry in os.scandir(path_):
                if entry.name.endswith(file_type) and entry.name.startswith(time_step.strftime("%Y%m%d")):
                    full_paths.append(os.path.join(path_, entry.name))
                    file_names.append(entry.name)

        number_of_days += 1
        start_date = end_date

    files_info = {
        'file_names': file_names,
        'full_paths': full_paths,
        'number_of_files': len(full_paths),
        'number_of_days': number_of_days
    }

    return files_info


def rreplace(s, old, new, occurrence):
    """Replaces the last occurrence of an expression in a string.
    https://stackoverflow.com/a/2556252

    Args:
        s (str):
        old (str):
        new (str):
        occurrence (int):

    Returns:
        li (str):

    """
    li = s.rsplit(old, occurrence)
    return new.join(li)
