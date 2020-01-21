#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dopplerlidarpy.utilities.arg_parser_tools import ap
from dopplerlidarpy.utilities import general_utils as gu

args = ap.my_args_parser()
gu.get_dl_file_list(args)
