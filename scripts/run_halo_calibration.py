#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dopplerlidarpy.utilities import my_args_parser as ap
from dopplerlidarpy.utilities import general_utils as gu

args = ap.my_args_parser()

full_paths = gu.get_dl_file_list(args)


