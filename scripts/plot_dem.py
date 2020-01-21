#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dopplerlidarpy.utilities import arg_parser_tools as ap
from dopplerlidarpy.utilities.dem_tools import get_dem

args = ap.my_args_parser()

dem_data = get_dem(args)
