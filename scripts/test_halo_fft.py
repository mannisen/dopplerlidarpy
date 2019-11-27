#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:08:40 2019

@author: manninan
"""

from netCDF4 import Dataset
from datetime import datetime
import getpass
import uuid
from dopplerlidarpy.utilities import my_args_parser as ap
from dopplerlidarpy.utilities import general_utils as gu
from dopplerlidarpy.utilities import plot_utils as pu
from dopplerlidarpy.equations import conversions
from dopplerlidarpy.equations import spectrum
from dopplerlidarpy.bottleneck.bottleneck.slow import move
# from dopplerlidarpy.utilities import dl_var_atts
from dopplerlidarpy.utilities import time_utils
# from dopplerlidarpy.utilities import my_nc_tools
from dopplerlidarpy.utilities import dl_toolbox_version
# from dopplerlidarpy.equations import acf
import pathlib
import cmocean
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from astropy.timeseries import LombScargle
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from scipy.optimize import curve_fit

# - Wavelength (lambda_w) is the distance over which the wave's shape (a cycle) repeats
# - Wavenumber (k) is the number of full cycles in a unit distance (i.e. lambda_w).
# -- k = (2*pi) / λ_w
# - Frequency (f) is the number of full cycles per unit time (sec).
# -- f = wave speed / λ_w
# --> λ_w = wave speed / f
# wave speed = wind speed (m s-1) = the distance a wave crest travels per unit time (units of distance / time)

# defaults
_FONTSIZE = 16
_TSTEP = 15/60  # hrs
_TWINDOW = 60/60  # hrs -- half width!!
_RWINDOW = 1  # has to be uneven!
ref_freq = np.arange(1e-3, 2e-1, 1e-4)
ref_freq_plot = np.arange(1e-2, 2e-1, 1e-4)
lambda_guess = np.array(np.arange(100, 3e3, 5e1))
_MISSING_VALUE = -9999
mu_ = 1

plt.rcParams.update({'font.size': _FONTSIZE})

_PATH_OUT = "/home/manninan/Documents/data/halo/kumpula/products/turbulence_length_scale-lidar/"

# generate inputs
site = 'kumpula'
start_date = '2019-08-24_00:00:00'
end_date = '2019-08-25_00:00:00'
processing_level = 'calibrated'
observation_type = 'stare'
pol = 'co'
args = ap.ArgsBlueprint(site=site, start_date=start_date, end_date=end_date, processing_level=processing_level,
                        observation_type=observation_type, pol=pol)

processing_level = 'products'
observation_type = 'windvad'
ele = '70'
args_winds = ap.ArgsBlueprint(site=site, start_date=start_date, end_date=end_date, processing_level=processing_level,
                              observation_type=observation_type, e=ele)

# Get paths
files_info = gu.get_dl_file_list(args)
files_info_winds = gu.get_dl_file_list(args_winds)
# iterate files

for i in range(files_info["number_of_files"]):
    if i == 0:

        # read from netcdf file
        print("Loading {}".format(files_info["full_paths"][i]))
        f = Dataset(files_info["full_paths"][i], "r")

        # get time and range
        time_ = np.array(f.variables["time"][:])
        height_AGL = np.array(f.variables["height_agl"][:])
        range_out = np.array(f.variables["range"][:])
        velo = np.array(f.variables["v_raw"][:])
        velo_error = np.array(f.variables["v_error"][:])
        beta_ = np.array(f.variables["beta_raw"][:])
        day_ = "{:1g}".format(getattr(f, "day"))
        month_ = "{:1g}".format(getattr(f, "month"))
        year_ = "{:1g}".format(getattr(f, "year"))
        f.close()

        f_ws = Dataset(files_info_winds["full_paths"][i], "r")
        # get time and range
        time_ws = np.array(f_ws.variables["time"][:])
        height_AGLws = np.array(f_ws.variables["height_agl"][:])
        ws = np.array(f_ws.variables["wind_speed"][:])
        ws_e = np.array(f_ws.variables["wind_speed_error"][:])
        f_ws.close()

        # interpolate ws to velo resolution
        model_interp2_ws = interp2d(height_AGLws[:], time_ws[:], ws, kind='linear', bounds_error=False)
        ws_interp = model_interp2_ws(height_AGL[:], time_[:])
        model_interp2_ws_e = interp2d(height_AGLws[:], time_ws[:], ws_e, kind='linear', bounds_error=False)
        ws_e_interp = model_interp2_ws_e(height_AGL[:], time_[:])

        time_filled, velo_filled = pu.fill_gaps_with_nans(time_[:], velo)
        _, beta_filled = pu.fill_gaps_with_nans(time_[:], beta_)
        _, ws_interp_filled = pu.fill_gaps_with_nans(time_[:], ws_interp)
        _, ws_e_interp_filled = pu.fill_gaps_with_nans(time_[:], ws_e_interp)

        velo_filled[:, :3] = np.nan
        beta_filled[:, :3] = np.nan
        ws_interp_filled[:, :3] = np.nan
        ws_e_interp_filled[:, :3] = np.nan

        # initialize
        lambda_0_out = np.empty([len(np.arange(0+_TWINDOW, 24, _TSTEP)), len(height_AGL)])
        lambda_0_out[:] = 0
        SE_out = np.empty([len(np.arange(0 + _TWINDOW, 24, _TSTEP)), len(height_AGL)])
        SE_out[:] = np.nan
        time_out = np.arange(0+_TWINDOW, 24, _TSTEP)

        # slide a window with size of _TWINDOW over time with steps _TSTEP
        jj = 0
        kk = 0
        for j in np.arange(0+_TWINDOW, 24, _TSTEP):
            kk = 0
            for k in range(len(height_AGL)):
                if 0 < j <= 24 and 105 < height_AGL[k] < 2000:  # 14.9 < j <= 15 and

                    # print("time: {}, range: {}".format(j, height_AGL[k]))

                    # Select a sample to calculate the FFT and LS spectrum
                    idx = np.array(np.where((time_ >= j-_TWINDOW) & (time_ <= j+_TWINDOW))).squeeze()
                    time_sel = time_[idx]*3600  # to seconds
                    uniques, counts = np.unique(np.diff(time_sel[:]), return_counts=True)
                    # print("uniques {}".format(uniques))
                    # print("counts  {}".format(counts))
                    delta_time = np.nanmedian(np.diff(time_sel[:]))
                    velo_sel = velo[idx, k]
                    velo_error_sel = velo_error[idx, k]
                    ws_sel = ws_interp[idx, k]
                    ws_e_sel = ws_e_interp[idx, k]
                    range_width = np.median(np.diff(height_AGL))

                    # Collect spectrum results
                    ii = 1

                    # Calculate wavelength and wavenumber using wind speed weighted by respective retrieval errors
                    ws_wmean = np.average(ws_sel[:], 0, 1/ws_e_sel[:]**2)
                    if np.isnan(ws_wmean):
                        kk += 1
                        continue
                    lambda_w = conversions.f2lambda(ws_wmean, ref_freq)

                    th_min_lambda = 2.5 * delta_time * ws_wmean
                    th_max_freq = conversions.lambda2f(ws_wmean, th_min_lambda)
                    # print("th min lambda {}".format(th_min_lambda))
                    # print("th max freq {}".format(th_max_freq))

                    ref_freq_valid = ref_freq[ref_freq < th_max_freq]
                    lambda_w_valid = conversions.f2lambda(ws_wmean, ref_freq_valid[:])
                    wavenumber = conversions.lambda2k(lambda_w_valid[:])
                    wavenumber_4model = conversions.lambda2k(lambda_w[:])

                    # -- estimate sigma2_w -- #
                    sigma2_w = np.nanvar(velo_sel[:])  # - np.nanvar(velo_error_sel[:])

                    # astropy implementation of Lomb-Scargle periodogram
                    ls_astro_power = LombScargle(time_sel[:], velo_sel[:]).power(ref_freq_valid, normalization='psd')
                    ls_astro_power_med = move.move_median(ls_astro_power, 5, 3)
                    model_interp = interp1d(ref_freq_valid[3:], ls_astro_power_med[3:], fill_value="extrapolate")
                    ls_astro_power_med_interp = model_interp(ref_freq_valid)

                    SE = np.array(np.empty([len(lambda_guess)]))
                    SE[:] = np.nan
                    mu_est = np.array(np.empty([len(lambda_guess)]))
                    mu_est[:] = np.nan
                    lambda_0_est = np.array(np.empty([len(lambda_guess)]))
                    lambda_0_est[:] = np.nan

                    i_SE = 0
                    Sk = np.empty(len(wavenumber))
                    Sk[:] = np.nan
                    func = spectrum.kristensen_spectral_intensity
                    for lambda_0 in lambda_guess:
                        if lambda_0 is not 0 and np.isfinite(lambda_0) and np.isfinite(sigma2_w):

                            # calculate standard error SE of Kristensen model to the observed spectra
                            # print("set: lambda0 {}, sigma2_w {}".format(lambda_0, sigma2_w))
                            try:
                                popt, pcov = curve_fit(func, wavenumber, 2e2*np.multiply(wavenumber, ls_astro_power_med_interp),
                                                       p0=[sigma2_w, mu_, lambda_0],
                                                       bounds=((sigma2_w*.99, mu_*.15, lambda_0*.9),
                                                               (sigma2_w*1.01, mu_*1.75, lambda_0*1.1)),
                                                       maxfev=10000)

                                variance_ = np.diagonal(pcov)

                                if np.all(np.isfinite(variance_)):
                                    SE[i_SE] = np.sqrt(variance_[-1])
                                    lambda_0_est[i_SE] = popt[-1]
                                    mu_est[i_SE] = popt[1]

                            except ValueError:
                                continue
                        i_SE += 1

                        # print("sigma2w {:.3g}, SE {:.2g}; mu {:.2g}, SE {:.2g}; lambda {:.2g}, SE {:.4g}".format(popt[0], np.sqrt(variance_[0]), popt[1], np.sqrt(variance_[1]), popt[2], np.sqrt(variance_[2])))
                    if np.all(np.isnan(SE)):
                        kk += 1
                        continue
                    idx_min_SE = np.nanargmin(SE)
                    SE_out[jj, kk] = SE[idx_min_SE]
                    lambda_0_out[jj, kk] = lambda_0_est[idx_min_SE]
                    mu_out = mu_est[idx_min_SE]

                    #print("time resolution {}".format(delta_time))
                    print("lambda_0 {} m".format(lambda_0_out[jj, kk]))

                    fig = plt.figure()
                    fig.set_size_inches(10, 10)
                    ax00 = plt.subplot2grid((2, 2), (0, 0))

                    wavenumber_plot = conversions.lambda2k(conversions.f2lambda(ws_wmean, ref_freq_plot))
                    h2 = plt.plot(wavenumber, 2e2*np.multiply(wavenumber, ls_astro_power), color=".75")
                    h21 = plt.plot(wavenumber, 2e2*np.multiply(wavenumber, ls_astro_power_med))
                    yy = spectrum.kristensen_spectral_intensity(wavenumber_4model, sigma2_w, mu_out,lambda_0_out[jj, kk])
                    h3 = plt.plot(wavenumber_4model, yy, linewidth=3)
                    h_l = plt.plot(conversions.lambda2k(np.repeat(lambda_0_out[jj, kk], 10)),
                                   np.linspace(.00001, 10, 10), color="black", linewidth=2.5, linestyle="--")

                    ax00.legend([h2[0], h21[0], h3[0], h_l[0]],
                                ['periodogram', 'running median', 'spectral model', '$\\lambda_{0}$'],
                                loc="upper right", bbox_to_anchor=(1.15, .35), fontsize=12)
                    ax00.set_zorder(100)
                    plt.yscale("log")
                    plt.xscale("log")
                    plt.grid()
                    ax00.set_xlabel("$k$ (rad m-1)")
                    ax00.set_ylabel("$k$ $S*(k)$")

                    ax00_2 = ax00.secondary_xaxis('top', functions=(conversions.k2lambda, conversions.lambda2k),
                                                  xlabel="Wavelength (m)")
                    ax00.set_ylim([2e-5, 8e0])
                    ax00.set_xlim([5e-4, 5e-1])

                    cmap1 = plt.get_cmap('pyart_HomeyerRainbow')
                    cmap2 = cmocean.cm.balance

                    ax02 = plt.subplot2grid((2, 2), (0, 1))
                    ax02.set_xlim([0, 15])
                    ax02.set_ylim([0, 2200])
                    ws_plot = ws_interp[idx, :]  # select data
                    ws_e_plot = ws_e_interp[idx, :]  # select data
                    ws_wmean_plot = np.average(ws_plot, 0, 1/ws_e_plot**2)  # calculate error weighted mean
                    for iw in range(np.size(ws_plot, 0)):
                        plt.plot(ws_plot[iw, :], height_AGLws[:], color=".75", linestyle="None", marker=".")
                    plt.plot(ws_wmean_plot, height_AGLws[:], linestyle="None", marker=".")
                    plt.plot(ws_wmean_plot[k], height_AGLws[k], linestyle="None", marker=".", markersize="20")
                    ax02.yaxis.tick_right()
                    ax02.yaxis.set_label_position("right")
                    ax02.set_xticks(np.arange(0, 18, 3))
                    ax02.set_ylabel("Height agl (m)")
                    ax02.set_xlabel("Wind speed (m s-1)")
                    ax02.grid()
                    ax02.set_title("Height range agl {}-{} m".format(height_AGL[k]-range_width/2,
                                                                     height_AGL[k]+range_width/2), fontsize=_FONTSIZE)

                    ax11 = plt.subplot2grid((2, 2), (1, 0))
                    ax11.set_xlim([np.min(time_sel/3600), np.max(time_sel/3600)])
                    ax11.set_ylim([0, 2200])
                    beta_filledm = np.ma.masked_invalid(beta_filled)
                    im1 = ax11.pcolormesh(time_filled, height_AGL, beta_filledm.T,
                                          norm=mcolors.LogNorm(vmin=1e-7, vmax=1e-4), cmap=cmap1)
                    P = patches.Rectangle((time_sel[0]/3600, height_AGL[int(k-np.floor(_RWINDOW/2))]),
                                          time_sel[-1]/3600, np.float(range_width), alpha=.5, color="black", fill=False,
                                          linestyle="--")
                    ax11.add_patch(P)
                    fig.colorbar(im1, ax=ax11, use_gridspec=True, extend="both",
                                 label='attenuated $\\beta*$ (m-1 sr-1)',
                                 orientation="horizontal", pad=0.2)
                    ax11.set_xlabel("time UTC (hrs)")
                    ax11.set_ylabel("Height agl (m)")

                    ax12 = plt.subplot2grid((2, 2), (1, 1))
                    ax12.set_xlim([np.min(time_sel/3600), np.max(time_sel/3600)])
                    ax12.set_ylim([0, 2200])
                    velo_filledm = np.ma.masked_invalid(velo_filled)
                    im2 = ax12.pcolormesh(time_filled, height_AGL, velo_filledm.T, vmin=-3, vmax=3, cmap=cmap2)
                    P = patches.Rectangle((time_sel[0]/3600, height_AGL[int(k-np.floor(_RWINDOW/2))]),
                                          time_sel[-1]/3600, float(range_width), alpha=.5, color="black", fill=False,
                                          linestyle="--")
                    ax12.add_patch(P)
                    fig.colorbar(im2, ax=ax12, use_gridspec=True, extend="both", label="vertical $v_r$ (m s-1)",
                                 orientation="horizontal", pad=0.2)
                    ax12.set_xlabel("time UTC (hrs)")

                    # plt.show()

                    hrs = np.floor(np.nanmedian(time_sel/3600))
                    mins = np.floor((np.nanmedian(time_sel/3600) - hrs) * 60)
                    if len("{:.0f}".format(hrs)) < 2:
                        hrs = "0" + "{:.0f}".format(hrs)  # e.g. 8 --> 08
                    else:
                        hrs = "{:.0f}".format(hrs)
                    if len("{:.0f}".format(mins)) < 2:
                        mins = "0" + "{:.0f}".format(mins)  # e.g. 5 --> 05
                    else:
                        mins = "{:.0f}".format(mins)

                    folders = "{}/{}{}{}/{}{}/".format(args.start_date[:4], args.start_date[:4], args.start_date[5:7],
                                                       args.start_date[8:10], hrs, mins)
                    if len("{:.0f}".format(k)) < 2:
                        k_str = "00" + "{:.0f}".format(k)  # e.g. 8 --> 008
                    elif len("{:.0f}".format(k)) < 3:
                        k_str = "0" + "{:.0f}".format(k)  # e.g. 8 --> 08
                    else:
                        k_str = "{:.0f}".format(k)
                    fname_out = "FFT-vs-LS-spectrum_" \
                                "win-{:.0f}min-{:.0f}gates_" \
                                "t{}{}UTC_gate{}_r{:.0f}m".format(int(_TWINDOW*2*60), int(_RWINDOW),
                                                                  hrs, mins, k_str, height_AGL[k])
                    fname_out = gu.rreplace(fname_out, ".", "_", 1)
                    full_path = _PATH_OUT + folders + fname_out + ".png"

                    # Create a folder if not exist
                    pathlib.Path(_PATH_OUT + folders).mkdir(parents=True, exist_ok=True)

                    print("Saving {}".format(full_path))
                    plt.savefig(full_path, dpi=80, facecolor='w', edgecolor='w',
                                format="png", bbox_inches="tight", pad_inches=0.1)
                    plt.close()

                kk += 1
            jj += 1

        # range_ = dl_var_atts.range_(data=range_out[:], dim_size=(len(range_out[:]), ))
        # height_agl = dl_var_atts.height_agl_(data=height_AGL[:], dim_size=(len(range_.data[:]), ))
        # time_hrs_utc = dl_var_atts.time_hrs_utc_(data=time_out[:], dim_size=(len(time_out[:]), ))

        unix_time_ = time_utils.time_hrs_utc2epoch(year_, month_, day_, time_out[:])
        # unix_time = dl_var_atts.unix_time_(data=unix_time_[:], dim_size=(len(time_out[:]), ))

        # lambda_zero = dl_var_atts.VarBlueprint("lambda_0", data=np.float64(lambda_0_out[:]),
        #                                       dim_name=("unix_time", "height_agl", ),
        #                                       dim_size=(len(time_out), len(range_out), ),
        #                                       standard_name="turbulence_transition_wavelength",
        #                                       long_name="transition wavelength", units="m", fill_value=False)
        # Prepare and write
        #obs = [unix_time, range_, height_agl, time_hrs_utc, lambda_zero]
        fname = files_info["file_names"][i][:8]
        file_name_ = _PATH_OUT + year_ + "/" + fname + "_" + args.site + '_' + "halo-doppler-lidar" + "_turbulence-length-scale.nc"
        date_txt = "{}-{:02d}-{:02d}".format(year_, int(month_), int(day_))
        print("Writing " + file_name_)

        rootgrp = Dataset(file_name_, 'w', format='NETCDF4')
        rootgrp.createDimension("unix_time", len(unix_time_))
        rootgrp.createDimension("range", len(range_out))

        nc_unix_time = rootgrp.createVariable("unix_time", "f8", "unix_time")
        nc_height_agl = rootgrp.createVariable("height_agl", "f8", "range")
        nc_time_hrs_UTC = rootgrp.createVariable("time_hrs_UTC", "f8", "unix_time")
        nc_range = rootgrp.createVariable("range", "f8", "range")
        nc_lambda0 = rootgrp.createVariable("lambda0", "f8", ("unix_time", "range"))

        nc_unix_time[:] = unix_time_[:]
        nc_height_agl[:] = height_AGL[:]
        nc_time_hrs_UTC[:] = time_out[:]
        nc_range[:] = range_out[:]
        nc_lambda0[:] = lambda_0_out[:]

        title_ = "HALO Doppler lidar retrievals",
        institution_ = "HALO Doppler lidar deployed on the roof of FMI at Helsinki, Finland",
        location_ = "Kumpula, Helsinki, Finland",
        source_ = "ground-based remote sensing"
        rootgrp.Conventions = 'CF-1.7'
        rootgrp.title = title_
        rootgrp.institution = institution_
        rootgrp.location = location_
        rootgrp.source = source_
        rootgrp.year = int(date_txt[:4])
        rootgrp.month = int(date_txt[5:7])
        rootgrp.day = int(date_txt[8:10])
        rootgrp.software_version = dl_toolbox_version.__version__
        rootgrp.file_uuid = str(uuid.uuid4().hex)
        rootgrp.references = ''
        user_name = getpass.getuser()
        now_time_utc = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        history_msg = "NETCDF4 file created by user {} on {} UTC.".format(user_name, now_time_utc)
        rootgrp.history = history_msg
        rootgrp.close()
        print(history_msg)

        # my_nc_tools.write_nc_(date_txt, file_name_, obs, title_="HALO Doppler lidar retrievals",
        #                      institution_="HALO Doppler lidar deployed on the roof of FMI at Helsinki, Finland",
        #                      location_="Kumpula, Helsinki, Finland",
        #                      source_="ground-based remote sensing")

