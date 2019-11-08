#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:08:40 2019

@author: manninan
"""

from netCDF4 import Dataset
from dopplerlidarpy.utilities import my_args_parser as ap
from dopplerlidarpy.utilities import general_utils as gu
from dopplerlidarpy.utilities import plot_utils as pu
import pathlib
import cmocean
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from astropy.timeseries import LombScargle
from scipy.interpolate import interp2d

# - Wavelength (lambda_w) is the distance over which the wave's shape (a cycle) repeats
# - Wavenumber (k) is the number of full cycles in a unit distance (i.e. lambda_w).
# -- k = 1 / λ_w
# - Frequency (f) is the number of full cycles per unit time (sec).
# -- f = 1 / sec = wave speed / λ_w
# --> λ_w = wave speed / f
# wave speed = wind speed (m s-1) = the distance a wave crest travels per unit time (units of distance / time)

# defaults
_TSTEP = 10/60  # hrs
_TWINDOW = 30/60  # hrs -- half width!!
_RWINDOW = 5  # has to be uneven!

plt.rcParams.update({'font.size': 16})

path_out = "/home/manninan/Documents/data/halo/kumpula/products/turbulence_length_scale-lidar/"

# generate inputs
site = 'kumpula'
start_date = '2019-08-01_00:00:00'
end_date = '2019-08-02_00:00:00'
processing_level = 'calibrated'
observation_type = 'stare'
pol = 'co'
args = ap.ArgsBlueprint(site=site, start_date=start_date, end_date=end_date, processing_level=processing_level,
                        observation_type=observation_type, pol=pol)

processing_level = 'product'
observation_type = 'windvad'
ele = '15'
args_winds = ap.ArgsBlueprint(site=site, start_date=start_date, end_date=end_date, processing_level=processing_level,
                              observation_type=observation_type, e=ele)

# Get paths
files_info = gu.get_dl_file_list(args)
files_info_winds = gu.get_dl_file_list(args_winds)
# iterate files

for i in np.arange(0, files_info["number_of_files"]):

    if i == 1:

        # read from netcdf file
        f = Dataset(files_info["full_paths"][i], "r")

        # get time and range
        time_ = np.array(f.variables["time"][:])
        height_agl_ = np.array(f.variables["height_agl"][:])
        velo = np.array(f.variables["v_raw"][:])
        velo_error = np.array(f.variables["v_error"][:])
        beta_ = np.array(f.variables["beta_raw"][:])
        f.close()

        f_ws = Dataset(files_info_winds["full_paths"][i], "r")

        # get time and range
        time_ws = np.array(f_ws.variables["time"][:])
        height_agl_ws = np.array(f_ws.variables["height_agl"][:])
        ws = np.array(f_ws.variables["wind_speed"][:])
        f_ws.close()

        # interpolate ws to velo resolution
        model_interp2 = interp2d(time_ws[:], height_agl_ws[:], ws, bounds_error=False, fill_value=np.NaN)
        ws_interp = model_interp2(time_[:], height_agl_[:])

        time_filled, velo_filled = pu.fill_gaps_with_nans(time_[:], velo)
        _, beta_filled = pu.fill_gaps_with_nans(time_[:], beta_)
        _, ws_interp_filled = pu.fill_gaps_with_nans(time_[:], ws_interp)

        # slide a window with size of _TWINDOW over time with steps _TSTEP
        for j in np.arange(0+_TWINDOW, 24, _TSTEP):
            for k in range(len(height_agl_)):
                if 15.9 < j <= 16 and 100 < height_agl_[k] < 2000:
                    # print("time: {}, range: {}".format(j, height_agl_[k]))

                    # Select a sample to calculate the FFT and LS spectrum
                    idx = np.array(np.where((time_ >= j-_TWINDOW) & (time_ <= j+_TWINDOW))).squeeze()
                    time_sel = time_[idx]*3600  # to seconds
                    velo_sel = velo[idx, int(k-np.floor(_RWINDOW/2)):int(k+np.floor(_RWINDOW/2))+1]
                    velo_error_sel = velo_error[idx, int(k-np.floor(_RWINDOW/2)):int(k+np.floor(_RWINDOW/2))+1]

                    range_width = _RWINDOW * np.median(np.diff(height_agl_))

                    # initialize
                    fft_psd_sum = 0
                    ls_astro_power_sum = 0

                    fft_freq_sum = 0
                    ls_astro_freq_sum = 0

                    # Collect spectrum results
                    ii = 1
                    for l in range(_RWINDOW):
                        # 0) rfft: real inputs, no add. info on the neg. freqs.
                        fft_asd = np.abs(np.fft.rfft(velo_sel[:, l]))
                        fft_psd = np.abs(fft_asd) ** 2  # PSD
                        fft_freq = np.fft.rfftfreq(time_sel.shape[-1], np.float(np.median(np.diff(time_sel))))

                        # astropy implementation of Lomb-Scargle periodogram
                        ls_astro_freq = np.arange(1e-4, 2e-2, 1e-4)
                        ls_astro_power = LombScargle(time_sel[:], velo_sel[:, l], velo_error_sel[:, l],
                                                     fit_mean=True, center_data=True, nterms=1).power(ls_astro_freq)

                        # Sum up, divide by N later to get the mean
                        fft_freq_sum += fft_freq[fft_freq > 0]
                        fft_psd_sum += fft_psd[fft_freq > 0]
                        ls_astro_freq_sum += ls_astro_freq
                        ls_astro_power_sum += ls_astro_power

                    fig = plt.figure()
                    fig.set_size_inches(10, 10)

                    ax00 = plt.subplot2grid((2, 2), (0, 0))
                    ax00.set_ylim([1e-4, 1e4])
                    ax00.set_xlim([2e-5, 1e-1])
                    h1 = plt.plot(fft_freq[fft_freq > 0], fft_psd_sum/_RWINDOW, marker=".")
                    h2 = plt.plot(ls_astro_freq, ls_astro_power_sum/_RWINDOW, marker=".")

                    xref = np.linspace(1e-3, 5e-2, 100)
                    for ix in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]:
                        h0 = plt.plot(xref, ix*xref**(-2/3), color="gray", linestyle=":")

                    ax00.legend([h1[0], h2[0], h0[0]],
                                ['FFT power spectral density', 'L-S periodogram',
                                 '-2/3 slope (inertial subrange)'],
                                loc="upper right", bbox_to_anchor=(2.2, 1))
                    plt.yscale("log")
                    plt.xscale("log")
                    plt.grid()
                    ax00.text(2e-5, 1.5e4, "range {}-{} m".format(height_agl_[int(k-np.floor(_RWINDOW/2))],
                                                                  height_agl_[int(k-np.floor(_RWINDOW/2))] + range_width))

                    ax00.set_xlabel("Frequency (Hz)")
                    ax00.set_ylabel("Power (au)")

                    cmap1 = plt.get_cmap('pyart_HomeyerRainbow')
                    cmap2 = cmocean.cm.balance

                    ax11 = plt.subplot2grid((2, 2), (1, 0))
                    ax11.set_xlim([np.min(time_sel/3600), np.max(time_sel/3600)])
                    ax11.set_ylim([0, 2200])
                    beta_filledm = np.ma.masked_invalid(beta_filled)
                    im1 = ax11.pcolormesh(time_filled, height_agl_, beta_filledm.T,
                                          norm=mcolors.LogNorm(vmin=1e-7, vmax=1e-4), cmap=cmap1)
                    P = patches.Rectangle((time_sel[0]/3600, height_agl_[int(k-np.floor(_RWINDOW/2))]),
                                          time_sel[-1]/3600, range_width, alpha=.5, color="black", fill=False,
                                          linestyle="--")
                    ax11.add_patch(P)
                    fig.colorbar(im1, ax=ax11, use_gridspec=True, extend="both", label="att. beta (m-1 sr-1)",
                                 orientation="horizontal", pad=0.2)
                    ax11.set_xlabel("time UTC (hrs)")
                    ax11.set_ylabel("range (m)")

                    ax12 = plt.subplot2grid((2, 2), (1, 1))
                    ax12.set_xlim([np.min(time_sel/3600), np.max(time_sel/3600)])
                    ax12.set_ylim([0, 2200])
                    velo_filledm = np.ma.masked_invalid(velo_filled)
                    im2 = ax12.pcolormesh(time_filled, height_agl_, velo_filledm.T, vmin=-3, vmax=3, cmap=cmap2)
                    P = patches.Rectangle((time_sel[0]/3600, height_agl_[int(k-np.floor(_RWINDOW/2))]),
                                          time_sel[-1]/3600, range_width, alpha=.5, color="black", fill=False,
                                          linestyle="--")
                    ax12.add_patch(P)
                    fig.colorbar(im2, ax=ax12, use_gridspec=True, extend="both", label="velocity (m s-1)",
                                 orientation="horizontal", pad=0.2)
                    ax12.set_xlabel("time UTC (hrs)")

                    # plt.show()

                    # fig.tight_layout()

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
                                                                  hrs, mins, k_str, height_agl_[k])
                    fname_out = gu.rreplace(fname_out, ".", "_", 1)
                    full_path = path_out + folders + fname_out + ".png"

                    # Create a folder if not exist
                    pathlib.Path(path_out + folders).mkdir(parents=True, exist_ok=True)

                    print("Saving {}".format(full_path))
                    plt.savefig(full_path, dpi=80, facecolor='w', edgecolor='w',
                                format="png", bbox_inches="tight", pad_inches=0.1)
                    plt.close()




