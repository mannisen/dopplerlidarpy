#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module converts original Halo Photonics files from hpl file format to netcdf4.

Required inputs are site name and paths to read and write files. The date written on the output netcdf4 file is read
from the data. By default the *.hpl files are assumed to have 'hpl' ending, encoding 'latin' and 17 header lines.

Examples:
        $ python3 halo_hpl2nc(site_name, path_in, path_out)
        $ python3 halo_hpl2nc(site_name, path_in, path_out, file_type='hpl', file_encoding='latin', header_lines=17)

Created on 2019-07-09
Antti Manninen
Finnish Meteorological Institute
dopplerlidarpy@fmi.fi
"""

import os
import re
import pathlib
import pandas as pd
import numpy as np
from datetime import datetime
from dopplerlidarpy.utilities import dl_var_atts as dl_atts
from dopplerlidarpy.utilities.my_nc_tools import write_nc_
from dopplerlidarpy.utilities.general_utils import look_for_from
pd.options.mode.chained_assignment = None  # default='warn'

def halo_hpl2nc(site_name, measurement_mode, path_in, path_out,
                file_type='hpl', file_encoding='latin', header_lines=17,
                elevation_angle=None, azimuth_angle=None, polarization=None):
    """Reads original Halo files and writes them into netcdf4 files.


    """

