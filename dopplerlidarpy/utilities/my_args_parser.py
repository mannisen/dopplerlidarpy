#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module parses given arguments.

Created on 2019-07-18
Antti Manninen
Finnish Meteorological Institute
dopplerlidarpy(at)fmi.fi
"""
import argparse
from operator import xor
from dopplerlidarpy.utilities.general_utils import list2str
from dopplerlidarpy.utilities.time_utils import validate_date
from dopplerlidarpy.utilities.input_checks import get_dl_input_list


class ArgsBlueprint(object):
    """Blueprint for args"""

    # Initialize with defaults
    def __init__(self, site=None, start_date=None, end_date=None, processing_level=None, observation_type=None,
                 pol=None, e=None, b=None, a=None, i=None, dpi=None):
        self.site = site
        self.start_date = start_date
        self.end_date = end_date
        self.processing_level = processing_level
        self.observation_type = observation_type
        self.pol = pol
        self.e = e
        self.b = b
        self.a = a
        self.i = i
        self.dpi = dpi


def meas_or_prod(args):
    """Checks processing level and respective dependent arguments

    Args:
        args (class): A populated namespace object

    """

    dl_in = get_dl_input_list()

    msg_body = "When processing level '{}' choose following argument from {}."

    if (args.processing_level in ["uncalibrated", "calibrated"]) and args.observation_type not in dl_in["measurements"]:
        raise NameError(msg_body.format(args.processing_level, list2str(dl_in["measurements"])))
    elif args.processing_level == 'product' and args.observation_type not in dl_in["products"]:
        raise NameError(msg_body.format(args.processing_level, list2str(dl_in["products"])))
    elif args.processing_level == 'raw' and args.observation_type not in dl_in["background_file"]:
        raise NameError(msg_body.format(args.observation_type, list2str(dl_in["background_file"])))


def meas_arg_parser(args):
    """Checks dependent options for measurements

    Args:
        args (class): A populated namespace object

    """

    dl_in = get_dl_input_list()

    msg_body = "'{}' requires argument {} {} to be given also."
    if args.observation_type == 'stare' and args.pol is None:
        raise NameError(msg_body.format(args.observation_type, '-pol', list2str(dl_in["polarization"])))
    elif args.observation_type == 'vad' and args.e is None:
        raise NameError(msg_body.format(args.observation_type, '-e', '[0-90]'))
    elif args.observation_type == 'rhi' and args.a is None:
        raise NameError(msg_body.format(args.observation_type, '-a', '[0-180]'))


def prod_arg_parser(args):
    """Checks dependent options for products

    Args:
        args (class): A populated namespace object

    """
    msg_body = "'{}' requires argument {} {} to be given also."
    if args.observation_type == 'wind_vad' and args.e is None:
        raise NameError(msg_body.format(args.observation_type, '-e', '[0-90]'))
    elif args.observation_type == 'wind_dbs' and args.b is None:
        raise NameError(msg_body.format(args.observation_type, '-b', '[3-5]'))
    elif args.observation_type == 'wind_shear' and xor(args.e is None, args.b is None):
        raise NameError(msg_body.format(args.observation_type, '-b | -e', '[0-90] | [3-5] (respectively)'))
    elif args.observation_type == 'epsilon' and xor(args.e is None, args.b is None):
        raise NameError(msg_body.format(args.observation_type, '-b | -e', '[0-90] | [3-5] (respectively)'))
    elif args.observation_type == 'sigma2_vad' and args.e is None:
        raise NameError(msg_body.format(args.observation_type, '-e', '[0-90]'))


def optional_arg_parser(args):
    """Check optional arguments

    Args:
        args (class): A populated namespace object

    """
    msg_body = "Optional argument {} is accepted only with '{}', not with '{}'..."
    if args.observation_type != 'stare' and args.pol is not None:
        raise NameError(msg_body.format('-pol', 'stare', args.observation_type))
    elif args.observation_type != 'rhi' and args.a is not None:
        raise NameError(msg_body.format('-a', 'rhi', args.observation_type))
    elif (args.observation_type not in ['vad', 'windvad', 'epsilon']) and args.e is not None:
        raise NameError(msg_body.format('-e', 'vad', args.observation_type))
    elif (args.observation_type not in ['epsilon', 'winddbs']) and args.b is not None:
        raise NameError(msg_body.format('-b', "'epsilon' or 'winddbs'", args.observation_type))
    elif (args.observation_type not in ['epsilon', 'windvad', 'sigma2vad']) and args.e is not None:
        raise NameError(msg_body.format('-e', "'epsilon', 'windvad', or 'sigma2vad'", args.observation_type))
    elif args.processing_level == "raw":
        raise NameError("Optional arguments are not accepted with '{}'".format(args.observation_type))


def my_args_parser():
    """Argument parser, checks that correct inputs were given.

    Returns:
        args (class): A populated namespace object

    """

    dl_in = get_dl_input_list()

    # Parse inputs
    parser = argparse.ArgumentParser()

    # site
    s_msg = "e.g. 'kuopio'"
    parser.add_argument('site', type=str, help=s_msg, metavar="site")

    # date(s)
    d_format = "'YYYY-mm-dd', 'YYYY-mm-dd_HH', 'YYYY-mm-dd_HH:MM', 'YYYY-mm-dd_HH:MM:SS'"
    parser.add_argument('start_date', type=str, help=d_format)
    parser.add_argument('end_date', type=str, help=d_format)

    # processing levels
    l_msg = list2str(dl_in["levels"])
    parser.add_argument('processing_level', type=str, help=l_msg, choices=dl_in["levels"],
                        metavar="level_of_processing")

    # measurement mode or product name
    obs_type_msg = list2str(dl_in["measurements"]) + list2str(dl_in["products"]) + list2str(dl_in["background_file"])
    obs_type_choices = dl_in["measurements"] + dl_in["products"]
    parser.add_argument('observation_type', type=str, help=obs_type_msg, choices=obs_type_choices,
                        metavar='e.g. stare, windvad, or epsilon')
    # polarization
    pol_msg = list2str(dl_in["polarization"])
    parser.add_argument('-pol', type=str, help=pol_msg,
                        choices=dl_in["polarization"], metavar='polarization')
    # elevation
    ele_msg = "degrees from the horizon, e.g. -e 75"
    parser.add_argument('-e', type=int, help=ele_msg,
                        choices=range(0, 90), metavar='elevation')
    # beams
    beams_msg = "number of dbs beams, e.g. -b 3"
    parser.add_argument('-b', type=int, help=beams_msg,
                        choices=range(3, 5), metavar='beams')
    # azimuth
    azi_msg = "degrees from North, e.g. -a 180"
    parser.add_argument('-a', type=int, help=azi_msg,
                        choices=range(0, 360), metavar='azimuth')

    # instrument
    ins_msg = "instrument name"
    parser.add_argument('-i', type=str, help=ins_msg, choices=dl_in["instruments"], metavar='instrument')

    # plot dpi
    dpi_msg = "dpi in plots"
    parser.add_argument('-dpi', type=int, help=dpi_msg, choices=range(50, 1000), metavar='dpi')

    args = parser.parse_args()

    # Check processing level and respective dependent arguments
    try:
        meas_or_prod(args)
    except NameError as err:
        raise err

    # Check dependent options of measurements
    if args.observation_type in dl_in["measurements"]:
        try:
            meas_arg_parser(args)
        except NameError as err:
            raise err

    # Check dependent options of products
    elif args.observation_type in dl_in["products"]:
        try:
            prod_arg_parser(args)
        except NameError as err:
            raise err

    # Check optional arguments
    try:
        optional_arg_parser(args)
    except NameError as err:
        raise err

    # Check date formats
    try:
        date_format_s = validate_date(args.start_date)
    except ValueError as err:
        raise err
    try:
        date_format_e = validate_date(args.end_date)
    except ValueError as err:
        raise err
    if date_format_e is not date_format_s:
        raise ValueError("Dates should be given in matching format: {}.".format(d_format))

    return args


if __name__ == '__main__':
    my_args_parser()
