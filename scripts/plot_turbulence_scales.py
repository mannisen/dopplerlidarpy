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
from dopplerlidarpy.utilities import arg_parser_tools as ap
from dopplerlidarpy.utilities import general_utils as gu
from dopplerlidarpy.bottleneck.bottleneck.slow import move
from scipy.interpolate import interp1d
from scipy.signal import medfilt2d

_FONTSIZE = 14
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
#observation_type = 'windvad'
#elevation = '70'
#args_windvad70 = ap.ArgsBlueprint(site=site, start_date=start_date, end_date=end_date, processing_level=processing_level,
#                                  observation_type=observation_type, e=elevation)

observation_type = 'epsilon'
args_epsilon = ap.ArgsBlueprint(site=site, start_date=start_date, end_date=end_date, processing_level=processing_level,
                                observation_type=observation_type)
observation_type = 'turbulence_length_scales'
args_scales = ap.ArgsBlueprint(site=site, start_date=start_date, end_date=end_date, processing_level=processing_level,
                               observation_type=observation_type)

# Get paths
#files_info_vad70 = gu.get_dl_file_list(args_windvad70)
files_info_epsilon = gu.get_dl_file_list(args_epsilon)
files_info_scales = gu.get_dl_file_list(args_scales)

# iterate over files
for i in range(files_info_scales["number_of_files"]):  # assumed one per day and same number of files each product

    #print("Loading {}".format(files_info_vad70["full_paths"][i]))
    #f_ws70 = Dataset(files_info_vad70["full_paths"][i], "r")
    #time_ws70 = np.array(f_ws70.variables["time"][:])
    #height_AGLws70 = np.array(f_ws70.variables["height_agl"][:])
    #ws70 = np.array(f_ws70.variables["wind_speed"][:])
    #f_ws70.close()

    print("Loading {}".format(files_info_epsilon["full_paths"][i]))
    f = Dataset(files_info_epsilon["full_paths"][i], "r")
    time_eps = np.array(f.variables["time_3min"][:])
    height_eps = np.array(f.variables["height_agl"][:])
    epsilon = np.array(f.variables["epsilon_3min"][:])
    epsilon_error = np.array(f.variables["epsilon_error_3min"][:])
    f.close()

    print("Loading {}".format(files_info_scales["full_paths"][i]))
    f = Dataset(files_info_scales["full_paths"][i], "r")
    time_scl = np.array(f.variables["time_hrs_UTC"][:])
    height_scl = np.array(f.variables["height_agl"][:])
    lambda0 = np.array(f.variables["lambda0"][:])
    ws_wmean = np.array(f.variables["wind_speed_weighted_mean"][:])
    l_w = np.array(f.variables["l_w"][:])
    f.close()

    # mask arrays
    ws_wmean[ws_wmean == -9999] = np.nan
    ws_wmean[ws_wmean <= 0] = np.nan

    ws_wmean_m = np.ma.masked_invalid(ws_wmean)

    lambda0[lambda0 == 0] = np.nan
    lambda0_smooth = medfilt2d(lambda0, [3, 3])
    lambda0_smooth[np.isnan(lambda0)] = np.nan
    lambda0_m = np.ma.masked_invalid(lambda0_smooth)

    l_w[l_w == 0] = np.nan
    l_w[l_w > 2000] = np.nan
    l_w_smooth = medfilt2d(l_w, [3, 3])
    l_w_smooth[np.isnan(l_w)] = np.nan
    l_w_m = np.ma.masked_invalid(l_w_smooth)

    epsilon[epsilon_error > 300] = np.nan
    epsilon[:, :4] = np.nan
    epsilon_smooth = medfilt2d(epsilon, [15, 3])
    epsilon_smooth[np.isnan(epsilon)] = np.nan
    epsilon_m = np.ma.masked_invalid(epsilon_smooth)

    # Set figure specs
    fig = plt.figure()
    fig.set_size_inches(10, 10)

    # Zeroth panel
    ax00 = plt.subplot2grid((4, 1), (0, 0), rowspan=1, colspan=1)
    im00 = ax00.pcolormesh(time_scl, height_scl, ws_wmean_m.T, vmin=0, vmax=15, cmap=_CMAP)
    fig.colorbar(im00, ax=ax00, use_gridspec=True, extend="max", label="(m s$^{-1}$)",
                 ticks=np.arange(0, 17.5, 2.5))

    ax00.set_xlim(_XLIMS)
    ax00.set_ylim(_YLIMS)
    ax00.set_ylabel(_YLABEL)
    ax00.set_xticks(_XTICKS)
    ax00.set_yticks(_YTICKS)
    ax00.text(_XLIMS[0] * 1.025, _YLIMS[1] * .875, "a) Horizontal error weighted wind speed")

    # First panel
    ax0 = plt.subplot2grid((4, 1), (1, 0), rowspan=1, colspan=1)
    im0 = ax0.pcolormesh(time_eps, height_eps, epsilon_m.T, norm=mcolors.LogNorm(vmin=1e-7, vmax=1e-2), cmap=_CMAP)
    fig.colorbar(im0, ax=ax0, use_gridspec=True, extend="both", label="$\\epsilon$ (m$^2$ s$^{-3}$)",
                 ticks=np.logspace(-7, -2, 6))

    ax0.set_xlim(_XLIMS)
    ax0.set_ylim(_YLIMS)
    ax0.set_ylabel(_YLABEL)
    ax0.set_xticks(_XTICKS)
    ax0.set_yticks(_YTICKS)
    ax0.text(_XLIMS[0]*1.025, _YLIMS[1]*.875, "b) TKE dissipation rate")

    # Second panel
    ax1 = plt.subplot2grid((4, 1), (2, 0), rowspan=1, colspan=1)
    im1 = ax1.pcolormesh(time_scl, height_scl, l_w_m.T, norm=mcolors.LogNorm(vmin=1e0, vmax=2e3), cmap=_CMAP)
    fig.colorbar(im1, ax=ax1, use_gridspec=True, extend="both", label="$l_w$ (m)",
                 ticks=np.logspace(0, 4, 5))
    ax1.set_xlim(_XLIMS)
    ax1.set_ylim(_YLIMS)
    ax1.set_ylabel(_YLABEL)
    ax1.set_xticks(_XTICKS)
    ax1.set_yticks(_YTICKS)
    ax1.text(_XLIMS[0] * 1.025, _YLIMS[1] * .875, "c) vertical wind integral scale")

    # Third panel
    ax2 = plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1)
    im1 = ax2.pcolormesh(time_scl, height_scl, lambda0_m.T, vmin=0, vmax=1200, cmap=_CMAP)
    fig.colorbar(im1, ax=ax2, use_gridspec=True, extend="max", label="$\\lambda_0$ (m)",
                 ticks=np.arange(0, 1600, 200))
    ax2.set_xlim(_XLIMS)
    ax2.set_ylim(_YLIMS)
    ax2.set_ylabel(_YLABEL)
    ax2.set_xticks(_XTICKS)
    ax2.set_yticks(_YTICKS)
    ax2.set_xlabel(_XLABEL)
    ax2.text(_XLIMS[0] * 1.025, _YLIMS[1] * .875, "d) transition wavelength")

    fig.tight_layout()
    # plt.show()

    file_name = files_info_scales["full_paths"][i]
    fname_out = gu.rreplace(file_name, "nc", "png", 1)
    print("Saving {}".format(fname_out))
    plt.savefig(fname_out, dpi=200, facecolor='w', edgecolor='w',
                format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close()


    # Set figure specs

    # Zeroth panel
    over_750 = np.argwhere(height_eps > 750)

    y_eps = np.array(epsilon_m[:, over_750[0]])
    y_eps[y_eps == 0] = np.nan

    y_lambda = lambda0_m[:, over_750[0]]
    y_lambda[y_lambda == 0] = np.nan

    y_lw = l_w_m[:, over_750[0]]
    y_lw[y_lw == 0] = np.nan

    fig1, host = plt.subplots()
    par1 = host.twinx()
    fig1.set_size_inches(8, 11)

    p1, = host.plot(time_eps, y_eps, marker=".", ls="None", color="darkorange", markersize=15)
    p2, = par1.plot(time_scl, y_lw, marker="s", ls="None", color="royalblue", markersize=10)
    p3, = par1.plot(time_scl, y_lambda, marker="^", linestyle="None", color="darkmagenta", markersize=10)

    host.set_yscale("log")
    par1.set_yscale("log")
    host.set_xlim(_XLIMS)
    host.set_ylim(1e-8, 1e-1)
    host.grid()
    par1.set_ylim(5e-1, 3e3)

    host.set_xlabel(_XLABEL)
    host.set_ylabel("(m$^2$ s$^{-3}$)")
    par1.set_ylabel("(m)")

    plt.legend([p1, p2, p3], ['$\\epsilon$ (left y-axis)', '$l_w$ (right y-axis)', '$\\lambda_{0}$ (right y-axis)'],
               fontsize=12)
    print(height_eps[over_750[0]])
    hh = int(height_eps[over_750[0]])
    host.text(_XLIMS[0] * 1.025, 1.2e-1, "Height agl {} m".format(hh))

    file_name = files_info_scales["full_paths"][i]
    fname_out = gu.rreplace(file_name, "nc", "png", 1)
    fname_out = gu.rreplace(fname_out, "turbulence-length-scale", "epsilon-lw-lambda0", 1)
    print("Saving {}".format(fname_out))
    plt.savefig(fname_out, dpi=200, facecolor='w', edgecolor='w',
                format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close()


