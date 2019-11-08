# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import cmocean

path_to = '/home/manninan/Documents/ARM-Stratus/data_transfer/'
file_name = '20140502_arm-darwin_halo-doppler-lidar_wind-vad-ele24.nc'
f = Dataset(path_to+file_name,'r',format = 'NETCDF4_CLASSIC')

time = f.variables['time'][:]
height = f.variables['height'][:]
ws = f.variables['wind_speed'][:]
ws = np.transpose(ws)
wd = f.variables['wind_direction'][:]
wd = np.transpose(wd)
ws_e = f.variables['wind_speed_error'][:]
ws_e = np.transpose(ws_e)
wd_e = f.variables['wind_direction_error'][:]
wd_e = np.transpose(wd_e)
snr = f.variables['mean_snr'][:]
snr = np.transpose(snr)
f.close()

snr_mask = snr < 1.005

#ws[snr_mask] = np.NaN
#ws_e[snr_mask] = np.NaN
#wd[snr_mask] = np.NaN
#wd_e[snr_mask] = np.NaN

plt.figure(num=None, figsize=(18, 4), facecolor='w', edgecolor='k')
plt.subplot(221)
plt.pcolor(time,height,ws, cmap = cmocean.cm.thermal, vmin=0, vmax=20)
plt.colorbar(ticks = np.arange(0,20,5), )

plt.subplot(222)
plt.pcolor(time,height,ws_e, cmap = cmocean.cm.thermal, vmin=0, vmax=2)
plt.colorbar(ticks = np.arange(0,2,.5), )

plt.subplot(223)
plt.pcolor(time,height,wd, cmap = cmocean.cm.phase, vmin=0, vmax=360)
plt.colorbar(ticks = np.arange(0,360,60), )

plt.subplot(224)
plt.pcolor(time,height,wd_e, cmap = cmocean.cm.thermal, vmin=0, vmax=3)
plt.colorbar(ticks = np.arange(0,3,.5), )

plt.show()
