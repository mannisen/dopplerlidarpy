#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:08:40 2019

@author: manninan
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from netCDF4 import Dataset
from dopplerlidarpy.utilities import my_args_parser as ap
from dopplerlidarpy.utilities import general_utils as gu

_FONTSIZE = 16
plt.rcParams.update({'font.size': _FONTSIZE})
_YLIMS = [0, 2200]
_XLIMS = [9, 24]
_YTICKS = np.arange(0, 2500, 500)
_XTICKS = np.arange(_XLIMS[0], _XLIMS[1]+3, 3)
_YLABEL = "Height agl (m)"
_XLABEL = "Time UTC (hours)"
_CMAP = plt.get_cmap('pyart_HomeyerRainbow')

# Path to write
_PATH_OUT = "/home/manninan/Documents/data/halo/kumpula/products/turbulence_length_scale-lidar/"

# inputs to read data
site = 'kumpula'
start_date = '2019-08-24_00:00:00'
end_date = '2019-08-25_00:00:00'
processing_level = 'products'
# observation_type = 'epsilon'
# args_epsilon = ap.ArgsBlueprint(site=site, start_date=start_date, end_date=end_date, processing_level=processing_level,
#                                observation_type=observation_type)
observation_type = 'turbulence_length_scales'
args_scales = ap.ArgsBlueprint(site=site, start_date=start_date, end_date=end_date, processing_level=processing_level,
                               observation_type=observation_type)

# Get paths
#files_info_epsilon = gu.get_dl_file_list(args_epsilon)
files_info_scales = gu.get_dl_file_list(args_scales)

# iterate over files
for i in range(files_info_scales["number_of_files"]):  # assumed one per day and same number of files each product

    # print("Loading {}".format(files_info_epsilon["full_paths"][i]))
    # f = Dataset(files_info_epsilon["full_paths"][i], "r")
    # time_eps = np.array(f.variables["time_hrs_UTC"][:])
    # height_eps = np.array(f.variables["height_agl"][:])
    # epsilon = np.array(f.variables["epsilon"][:])
    # f.close()

    time_eps = np.linspace(0, 24, 1000)
    height_eps = np.linspace(0, 2500, 100)
    epsilon = np.random.rand(np.size(time_eps), np.size(height_eps))

    print("Loading {}".format(files_info_scales["full_paths"][i]))
    f = Dataset(files_info_scales["full_paths"][i], "r")
    time_scl = np.array(f.variables["time_hrs_UTC"][:])
    height_scl = np.array(f.variables["height_agl"][:])
    lambda0 = np.array(f.variables["lambda0"][:])
    l_w = np.array(f.variables["l_w"][:])
    f.close()

    # mask arrays
    lambda0[lambda0 == 0] = np.nan
    l_w[l_w == 0] = np.nan

    # Set figure specs
    fig = plt.figure()
    fig.set_size_inches(8, 10)

    # First panel
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1)
    ax0.set_xlim([9, 21])
    epsilon_m = np.ma.masked_invalid(epsilon)
    im0 = ax0.pcolormesh(time_eps, height_eps, epsilon_m.T, norm=mcolors.LogNorm(vmin=1e-6, vmax=1e-1), cmap=_CMAP)
    fig.colorbar(im0, ax=ax0, use_gridspec=True, extend="both", label="$\\epsilon$ (m2 s-3)",
                 ticks=np.logspace(-6, -1, 6))

    ax0.set_ylim(_YLIMS)
    ax0.set_ylabel(_YLABEL)
    ax0.set_xticks(_XTICKS)
    ax0.set_yticks(_YTICKS)
    ax0.text(_XLIMS[0]*1.025, _YLIMS[1]*.9, "a) TKE dissipation rate")

    # Second panel
    ax1 = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1)
    lambda0_m = np.ma.masked_invalid(lambda0)
    im1 = ax1.pcolormesh(time_scl, height_scl, lambda0_m.T, vmin=0, vmax=2000, cmap=_CMAP)
    fig.colorbar(im1, ax=ax1, use_gridspec=True, extend="max", label="$\\lambda_0$ (m)",
                 ticks=np.arange(0, 2500, 500))
    ax1.set_xlim(_XLIMS)
    ax1.set_ylim(_YLIMS)
    ax1.set_ylabel(_YLABEL)
    ax1.set_xticks(_XTICKS)
    ax1.set_yticks(_YTICKS)
    ax1.text(_XLIMS[0]*1.025, _YLIMS[1]*.9, "b) Transition wavelength")

    # Third panel
    ax2 = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1)
    l_w_m = np.ma.masked_invalid(l_w)
    im2 = ax2.pcolormesh(time_scl, height_scl, l_w_m.T, vmin=0, vmax=2000, cmap=_CMAP)
    fig.colorbar(im2, ax=ax2, use_gridspec=True, extend="max", label="$l_w$ (m)",
                 ticks=np.arange(0, 2500, 500))
    ax2.set_xlim(_XLIMS)
    ax2.set_ylim(_YLIMS)
    ax2.set_xlabel(_XLABEL)
    ax2.set_ylabel(_YLABEL)
    ax2.set_xticks(_XTICKS)
    ax2.set_yticks(_YTICKS)
    ax2.text(_XLIMS[0]*1.025, _YLIMS[1]*.9, "c) Along-wind integral scale")

    fig.tight_layout()
    # plt.show()

    file_name = files_info_scales["full_paths"][i]
    fname_out = gu.rreplace(file_name, "nc", "png", 1)
    print("Saving {}".format(fname_out))
    plt.savefig(fname_out, dpi=200, facecolor='w', edgecolor='w',
                format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close()
