#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module reads Doppler lidar netcdf nc files and generates standard plots in .png format.

Examples:
        $ ...
        $ ...

Created on 2019-07-18
Antti Manninen
Finnish Meteorological Institute
dopplerlidarpy(at)fmi.fi
"""
from dopplerlidarpy.utilities import general_utils as gu
from dopplerlidarpy.utilities import dl_var_plot_atts as patts
import matplotlib.pylab as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset
import numpy as np

_YMIN = 0
_YMAX = 3
_XMIN = 0
_XMAX = 24
_YTICK = np.arange(_YMIN, _YMAX+1, 1)
_XTICK = np.arange(_XMIN, _XMAX+3, 3)
_XLABEL = "time UTC (hrs)"
_YLABEL = "range (km)"
_PLOT_XRESO = 3000  # pixels
_RANGE_UNITS = "km"
_STARE_OBS = ['signal0', 'signal', 'signal0_error', 'signal_error', 'v_raw', 'v_error']


def read_dl_vars(file_name, obs):

    # Get netcdf handle
    f = Dataset(file_name, "r")

    # Extract slices of attributes
    step = int(np.ceil(len(f.dimensions["time"]) / _PLOT_XRESO))

    d_out = dict()

    # Read time and range
    d_out["time"] = f.variables["time"][::step]
    if _RANGE_UNITS is "km":
        d_out["range"] = f.variables["range"][:] / 1000  # in km
    elif _RANGE_UNITS is "m":
        d_out["range"] = f.variables["range"][:]  # in m
    else:
        raise ValueError("Range units must be 'km' or 'm'!")

    # Read attributes in time-range dimensions
    for var in obs:
        d_out[var] = f.variables[var][::step, :].transpose()

    f.close()

    return d_out


def create_axis_(ax, txt_=" "):
    ax.set_ylim([_YMIN, _YMAX])
    ax.set_xlim([_XMIN, _XMAX])
    ax.set_xlabel(_XLABEL)
    ax.set_ylabel(_YLABEL)
    ax.set_xticks(_XTICK)
    ax.set_yticks(_YTICK)
    ax.text(0, 3.1, "a) " + txt_)

    return ax


def create_pcolor_plot(fig_=None, ax_=None, x_=None, y_=None, Z=None):
    """

    Args:
        fig_ (Figure): figure handle
        ax_ (AxesSubplot): axis handle
        x_ (1-D ndarray): x-data, e.g. time
        y_ (1-D ndarray): y-data, e.g. range
        Z (2-D ndarray): color data, e.g. v_raw

    """

    # Extract slices of attributes
    step = int(np.ceil(len(f.dimensions["time"]) / _PLOT_XRESO))
    # Get time
    time_ = f.variables[x_][::step]
    # Get y axis values: "range", "height_agl", or "height_asl"
    range_ = f.variables[y_][:] / 1000

    var_ = getattr(patts, var_name)

    im = ax_.pcolormesh(time_, range_, x, vmin=var_.vmin, vmax=var_.vmax, cmap=var_.cmap)

    fig_.colorbar(im, ax=ax_, use_gridspec=True, extend=var_.cextend, label=var_.units)


def plot_stare(args):
    """Generates a plot of calibrated Doppler lidar nc measured in vertical stare pointing mode.

    Args:
        args (class): A populated namespace object, output of my_args_parser

    """
    # Set constants
    plot_time_reso = 3000
    cmap = plt.get_cmap('pyart_HomeyerRainbow')

    # Get paths
    files_info = gu.get_dl_file_list(args)

    for file_name in files_info['full_paths']:

        # Get netcdf handle
        print("Loading {}".format(file_name))
        data = read_dl_vars(file_name, _STARE_OBS)

        # Get figure handle, set size
        fig = plt.figure()
        fig.set_size_inches(10, 10)

        # signal0
        ax00 = plt.subplot2grid((5, 2), (0, 0), rowspan=1, colspan=1)
        create_axis_(ax00, "a) uncorrected signal")
        create_pcolor_plot(fig, ax00, data["time"], data["range"], data["signal0"])


        # signal0_error
        ax10 = plt.subplot2grid((5, 2), (0, 1), rowspan=1, colspan=1)
        im10 = ax10.pcolormesh(time_, range_, signal0_error, norm=mcolors.LogNorm(vmin=1, vmax=400), cmap=cmap)
        ax10.set_ylim([0, 3])
        ax10.set_xlim([0, 24])
        plt.yticks(np.arange(0, 4, 1))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax10.set_xlabel("time UTC (hrs)")
        ax10.set_ylabel("range (km)")
        fig.colorbar(im10, ax=ax10, use_gridspec=True, extend="max", label="(%)")
        plt.text(0, 3.1, "b) uncorrected signal fractional error")

        # signal
        ax01 = plt.subplot2grid((5, 2), (1, 0), rowspan=1, colspan=1)
        im01 = ax01.pcolormesh(time_, range_, signal, vmin=.99, vmax=1.02, cmap=cmap)
        ax01.set_ylim([0, 3])
        ax01.set_xlim([0, 24])
        plt.yticks(np.arange(0, 4, 1))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax01.set_xlabel("time UTC (hrs)")
        ax01.set_ylabel("range (km)")
        fig.colorbar(im01, ax=ax01, use_gridspec=True, extend="both", label="(SNR+1)")
        plt.text(0, 3.1, "c) corrected signal")

        # signal_error
        ax11 = plt.subplot2grid((5, 2), (1, 1), rowspan=1, colspan=1)
        im11 = ax11.pcolormesh(time_, range_, signal_error, norm=mcolors.LogNorm(vmin=1, vmax=400), cmap=cmap)
        ax11.set_ylim([0, 3])
        ax11.set_xlim([0, 24])
        plt.yticks(np.arange(0, 4, 1))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax11.set_xlabel("time UTC (hrs)")
        ax11.set_ylabel("range (km)")
        fig.colorbar(im11, ax=ax11, use_gridspec=True, extend="max", label="(%)")
        plt.text(0, 3.1, "d) corrected signal fractional error")

        # v
        ax01 = plt.subplot2grid((5, 2), (2, 0), rowspan=1, colspan=1)
        im01 = ax01.pcolormesh(time_, range_, v, vmin=-2, vmax=2, cmap=plt.get_cmap('pyart_balance'))
        ax01.set_ylim([0, 3])
        ax01.set_xlim([0, 24])
        plt.yticks(np.arange(0, 4, 1))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax01.set_xlabel("time UTC (hrs)")
        ax01.set_ylabel("range (km)")
        fig.colorbar(im01, ax=ax01, use_gridspec=True, extend="both", label="(m s-1)", ticks=[-2, -1, 0, 1, 2])
        plt.text(0, 3.1, "e) radial velocity")

        # v_error
        ax11 = plt.subplot2grid((5, 2), (2, 1), rowspan=1, colspan=1)
        im11 = ax11.pcolormesh(time_, range_, v_error, norm=mcolors.LogNorm(vmin=.001, vmax=10), cmap=cmap)
        ax11.set_ylim([0, 3])
        ax11.set_xlim([0, 24])
        plt.yticks(np.arange(0, 4, 1))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax11.set_xlabel("time UTC (hrs)")
        ax11.set_ylabel("range (km)")
        fig.colorbar(im11, ax=ax11, use_gridspec=True, extend="both", label="(m s-1)", ticks=[.001, .01, .1, 1, 10])  #ticks=[0, .25, .5, .75, 1]
        plt.text(0, 3.1, "f) radial velocity measurement uncertainty")

        # v vs signal0
        ax20 = plt.subplot2grid((5, 2), (3, 0), rowspan=2, colspan=1)
        # norm = mcolors.LogNorm(vmin=1, vmax=1000)
        norm = "asgfa"
        im20 = ax20.hist2d(signal0.ravel(), v.ravel(), bins=bins, cmap=cmap, cmin=0, cmax=1000, norm=norm)
        ax20.set_ylim([-20, 20])
        ax20.set_xlim([.99, 1.015])
        ax20.set_xlabel("signal0 (SNR+1)")
        ax20.set_ylabel("v (m s-1)")
        ax20.vlines(1, 0, 1, transform=ax20.get_xaxis_transform(), colors='k', linestyles='dashed')
        cbar20 = fig.colorbar(im20[3], ax=ax20, norm=norm, use_gridspec=True, extend="max", label="occurrence")
        plt.text(.99, 20.5, "g) radial velocity vs uncorrected signal")

        # v vs signal
        ax21 = plt.subplot2grid((5, 2), (3, 1), rowspan=2, colspan=1)
        im21 = ax21.hist2d(signal.ravel(), v.ravel(), bins=bins, cmap=cmap, cmin=0, cmax=1000,  norm=norm)
        ax21.set_ylim([-20, 20])
        ax21.set_xlim([.99, 1.015])
        ax21.set_xlabel("signal (SNR+1)")
        ax21.set_ylabel("v (m s-1)")
        ax21.vlines(1, 0, 1, transform=ax21.get_xaxis_transform(), colors='k', linestyles='dashed')
        cbar21 = fig.colorbar(im21[3], ax=ax21, norm=norm, use_gridspec=True, extend="max", label="occurrence")
        plt.text(.99, 20.5, "h) radial velocity vs corrected signal")

        fig.tight_layout()

        fname_out = gu.rreplace(file_name, "nc", "png", 1)
        print("Saving {}".format(fname_out))
        plt.savefig(fname_out, dpi=args.dpi, facecolor='w', edgecolor='w',
                    format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close()


def plot_wstats(args):
    """Generates a plot of vertical velocity statistics (wstats) estimated from Doppler lidar measurements.

    Args:
        args (class): A populated namespace object, output of my_args_parser

    """

    # Set constants
    cmap = plt.get_cmap('cubehelix')


def plot_epsilon(args):
    """Generates a plot of turbulent kinetic energy dissipation rate estimated from Doppler lidar measurements.
        Args:
            args (class): A populated namespace object, output of my_args_parser

        """
    # Set constants
    cmap = plt.get_cmap('pyart_HomeyerRainbow')

    # Get paths
    files_info = gu.get_dl_file_list(args)

    for file_name in files_info['full_paths']:
        # Get netcdf handle
        print("Loading {}".format(file_name))
        f = Dataset(file_name, "r")

        time_ = f.variables["time_3min"][:]
        height_ = f.variables["height"][:]/1000
        epsilon = f.variables["epsilon_3min"][:].transpose()
        epsilon_error = f.variables["epsilon_error_3min"][:].transpose() * 100

        f.close()

        # Get figure handle, set size
        fig = plt.figure()
        fig.set_size_inches(10, 2.133)

        # epsilon
        ax00 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
        norm = mcolors.LogNorm(1e-6, 1e-1)
        im00 = ax00.pcolormesh(time_, height_, epsilon, cmap=cmap, norm=norm, vmin=1e-6, vmax=1e-1)
        cbar00 = fig.colorbar(im00, ax=ax00, norm=norm, use_gridspec=True, extend="both", label="(m2 s-3)",
                              ticks=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
        plt.text(0, 3.1, "a) TKE dissipation rate")
        ax00.set_ylim([0, 3])
        ax00.set_xlim([0, 24])
        plt.yticks(np.arange(0, 3.5, .5))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax00.set_xlabel("time UTC (hrs)")
        ax00.set_ylabel("height (km agl)")

        # epsilon_error
        ax10 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
        im10 = ax10.pcolormesh(time_, height_, epsilon_error, cmap=cmap, vmin=0, vmax=300)
        ax10.set_ylim([0, 3])
        ax10.set_xlim([0, 24])
        plt.yticks(np.arange(0, 3.5, .5))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax10.set_xlabel("time UTC (hrs)")
        ax10.set_ylabel("height (km agl)")
        cbar10 = fig.colorbar(im10, ax=ax10, use_gridspec=True, extend="max",
                              label="(%)", ticks=[0, 100, 200, 300])
        plt.text(0, 3.1, "b) TKE dissipation rate fractional uncertainty")

        fig.tight_layout()

        fname_out = gu.rreplace(file_name, "nc", "png", 1)
        print("Saving {}".format(fname_out))
        plt.savefig(fname_out, dpi=args.dpi, facecolor='w', edgecolor='w',
                    format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close()


def plot_windshear(args):
    """Generates a plot of turbulent kinetic energy dissipation rate estimated from Doppler lidar measurements.
        Args:
            args (class): A populated namespace object, output of my_args_parser

        """
    # Set constants
    cmap = plt.get_cmap('pyart_HomeyerRainbow')

    # Get paths
    files_info = gu.get_dl_file_list(args)

    for file_name in files_info['full_paths']:

        # Get netcdf handle
        print("Loading {}".format(file_name))
        f = Dataset(file_name, "r")

        time_ = f.variables["time_3min"][:]
        height_ = f.variables["height"][:]/1000
        shear = f.variables["vector_wind_shear_3min"][:].transpose()
        shear_error = f.variables["vector_wind_shear_error_3min"][:].transpose()

        f.close()

        # Get figure handle, set size
        fig = plt.figure()
        fig.set_size_inches(10, 2.133)

        # wind shear
        ax00 = plt.subplot2grid((1, 2), (0, 0), rowspan=1, colspan=1)
        im00 = ax00.pcolormesh(time_, height_, shear, cmap=cmap, vmin=0, vmax=.075)
        cbar00 = fig.colorbar(im00, ax=ax00, use_gridspec=True, extend="max", label="(m s-1 m-1)",
                              ticks=np.arange(0, .125, .025))
        plt.text(0, 3.1, "a) vector wind shear")
        ax00.set_ylim([0, 3])
        ax00.set_xlim([0, 24])
        plt.yticks(np.arange(0, 3.5, .5))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax00.set_xlabel("time UTC (hrs)")
        ax00.set_ylabel("height (km agl)")

        # wind shear error
        ax10 = plt.subplot2grid((1, 2), (0, 1), rowspan=1, colspan=1)
        im10 = ax10.pcolormesh(time_, height_, shear_error, cmap=cmap, vmin=0, vmax=.02)
        ax10.set_ylim([0, 3])
        ax10.set_xlim([0, 24])
        plt.yticks(np.arange(0, 3.5, .5))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax10.set_xlabel("time UTC (hrs)")
        ax10.set_ylabel("height (km agl)")
        cbar10 = fig.colorbar(im10, ax=ax10, use_gridspec=True, extend="max",
                              label="(m s-1 m-1)", ticks=np.arange(0, .025, .005))
        plt.text(0, 3.1, "b) vector wind shear error")

        fig.tight_layout()

        fname_out = gu.rreplace(file_name, "nc", "png", 1)
        print("Saving {}".format(fname_out))
        plt.savefig(fname_out, dpi=args.dpi, facecolor='w', edgecolor='w',
                    format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close()


def plot_windvad(args):
    """Generates a plot of turbulent kinetic energy dissipation rate estimated from Doppler lidar measurements.
        Args:
            args (class): A populated namespace object

    """
    # Set constants
    cmap = plt.get_cmap('pyart_HomeyerRainbow')

    # Get paths
    files_info = gu.get_dl_file_list(args)

    for file_name in files_info['full_paths']:
        # Get netcdf handle
        print("Loading {}".format(file_name))
        f = Dataset(file_name, "r")

        time_ = f.variables["time"][:]
        height_ = f.variables["height"][:]/1000
        ws = f.variables["wind_speed"][:].transpose()
        ws_e = f.variables["wind_speed_error"][:].transpose()
        wd = f.variables["wind_direction"][:].transpose()
        wd_e = f.variables["wind_direction_error"][:].transpose()

        # Get figure handle, set size
        fig = plt.figure()
        fig.set_size_inches(10, 4.133)

        # ws
        att = patts.wind_speed()
        ax00 = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
        im00 = ax00.pcolormesh(time_, height_, ws, vmin=att.vmin, vmax=att.vmax, cmap=att.cmap)
        ax00.set_ylim(_XLIM)
        ax00.set_xlim(_YLIM)
        plt.yticks(_YTICKS)
        plt.xticks(_XTICKS)
        ax00.set_xlabel(att.xlabel)
        ax00.set_ylabel(att.ylabel)
        fig.colorbar(im00, ax=ax00, use_gridspec=True, extend="max", label="(m s-1)")
        plt.text(0, 3.1, "a) wind speed")

        # ws_e
        ax10 = plt.subplot2grid((2, 2), (0, 1), rowspan=1, colspan=1)
        im10 = ax10.pcolormesh(time_, height_, ws_e, vmin=0, vmax=2, cmap=cmap)
        ax10.set_ylim([0, 3])
        ax10.set_xlim([0, 24])
        plt.yticks(np.arange(0, 3.5, .5))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax10.set_xlabel("time UTC (hrs)")
        ax10.set_ylabel("height (km agl)")
        fig.colorbar(im10, ax=ax10, use_gridspec=True, extend="max", label="(m s-1)", ticks=np.arange(0, 2.5, .5))
        plt.text(0, 3.1, "b) wind speed error")

        # wd
        ax01 = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
        im01 = ax01.pcolormesh(time_, height_, wd, vmin=0, vmax=360, cmap=d)
        ax01.set_ylim([0, 3])
        ax01.set_xlim([0, 24])
        plt.yticks(np.arange(0, 3.5, .5))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax01.set_xlabel("time UTC (hrs)")
        ax01.set_ylabel("height (km agl)")
        fig.colorbar(im01, ax=ax01, use_gridspec=True, extend="both", label="(degrees)", ticks=np.arange(0, 420, 60))
        plt.text(0, 3.1, "c) wind direction")

        # wd_e
        ax11 = plt.subplot2grid((2, 2), (1, 1), rowspan=1, colspan=1)
        im11 = ax11.pcolormesh(time_, height_, wd_e, vmin=0, vmax=.2, cmap=cmap)
        ax11.set_ylim([0, 3])
        ax11.set_xlim([0, 24])
        plt.yticks(np.arange(0, 3.5, .5))
        plt.xticks(np.arange(0, 24 + 3, 3))
        ax11.set_xlabel("time UTC (hrs)")
        ax11.set_ylabel("range (km)")
        fig.colorbar(im11, ax=ax11, use_gridspec=True, extend="max", label="(degrees)", ticks=np.arange(0, .25, 0.05))
        plt.text(0, 3.1, "d) wind direction error")

        fig.tight_layout()

        fname_out = gu.rreplace(file_name, "nc", "png", 1)
        print("Saving {}".format(fname_out))
        plt.savefig(fname_out, dpi=args.dpi, facecolor='w', edgecolor='w',
                    format="png", bbox_inches="tight", pad_inches=0.1)
        plt.close()


def plot_dl(args):
    """

    Args:
        args: (class): A populated namespace object, output of my_args_parser

    Returns:

    """

    print_msg = ["Plotting {} {}...{} ".format(args.site, args.start_date, args.end_date)]
    if args.processing_level == "calibrated":
        print_msg.append("calibrated ")
        if args.observation_type == "stare":
            print_msg.append("stare ")
            print_msg.append("co")
            print(" ".join(print_msg))

            # Generate plot
            plot_stare(args)

        else:
            raise ValueError

    elif args.processing_level == "product":
        print_msg.append("product")

        if args.observation_type == "epsilon":
            print_msg.append("epsilon ")
            print(" ".join(print_msg))

            # Generate plot
            plot_epsilon(args)

        elif args.observation_type == "windshear":
            print_msg.append("windshear ")
            print(" ".join(print_msg))

            # Generate plot
            plot_windshear(args)

        elif args.observation_type == "windvad":
            print_msg.append("windvad ")
            print(" ".join(print_msg))

            # Generate plot
            plot_windvad(args)

        else:
            raise ValueError

    else:
        raise ValueError
