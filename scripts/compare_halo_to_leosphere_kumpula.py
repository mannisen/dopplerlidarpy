#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:48:23 2019

@author: manninan
"""

import os
from netCDF4 import Dataset
from datetime import datetime
from dopplerlidarpy.utilities.time_utils import datetime_range
import numpy as np

path_to_leo = ""

full_paths = []
file_names = []

start_date = datetime(2019, 12, 31, 23, 59, 59)
end_date = datetime(2020, 1, 1, 0, 0, 2)
for i in datetime_range(start_date, end_date):
    # List files and get only files for which parameters are valid
    try:
        for entry in os.scandir(path_to_leo):
            if entry.name.endswith(".nc"):
                full_paths.append(os.path.join(path_to_leo, entry.name))
                file_names.append(entry.name)
    except FileNotFoundError:
        raise  # print("FileNotFoundError: {0}".format(err))

    f = Dataset(file_name_leo, "r")