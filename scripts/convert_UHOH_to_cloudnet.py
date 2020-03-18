#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np
from dopplerlidarpy.utilities import nc_tools
from dopplerlidarpy.utilities import general_utils as gu
from dopplerlidarpy.utilities.dl_var_atts import dl_var_atts as vatts

_SAVE_NC = True  # True
path_in = "/home/manninan/Documents/data/halo/arm-sgpC1/product/windvad-UHOH/2017/original_files/"
path_out = "/home/manninan/Documents/data/halo/arm-sgpC1/product/windvad-UHOH/2017/"

full_paths = []
file_names = []

files_info = gu.list_files(path_in, ".nc", "VADprof")
names_in = ["time_offset", "heights", "speed", "direction", "decTimeVec"]
names_out = ["time_unix", "height_agl", "wind_speed", "wind_direction", "time"]

for i in range(files_info["number_of_files"]):  # assumed one per day
    print("Loading {}".format(files_info["full_paths"][i]))
    data_in = nc_tools.read_nc_fields(files_info["full_paths"][i], names_in)

    # Rename variables and assign attributes
    data_out = [vatts(names_out[j], data=data_in[j], dim_size=np.shape(data_in[j])) for j in range(len(data_in)-1)]

    # Add time hours UTC
    t_hrs = data_in[-1][3] + (data_in[-1][4] + data_in[-1][5]/60)/60
    data_out.append(vatts(names_out[-1], data=t_hrs, dim_size=len(t_hrs)))

    # Fix times, sometimes last time stamp from the next day
    a2 = np.append(0, data_out[0].data)
    data_out[0].data[np.diff(a2) < 0] = data_out[0].data[np.diff(a2) < 0] + 24 * 3600
    a2 = np.append(0, data_out[-1].data)
    data_out[-1].data[np.diff(a2) < 0] = data_out[-1].data[np.diff(a2) < 0] + 24

    # Calculate u- and v-components
    u_wind = np.multiply(np.abs(data_out[2].data), np.sin(np.divide(np.multiply(np.pi, 270-data_out[3].data), 180)))
    v_wind = np.multiply(np.abs(data_out[2].data), np.cos(np.divide(np.multiply(np.pi, 270-data_out[3].data), 180)))
    data_out.append(vatts("u", data=u_wind, dim_size=data_out[2].dim_size))
    data_out.append(vatts("v", data=v_wind, dim_size=data_out[2].dim_size))

    if _SAVE_NC is True:

        # Prepare and write
        fname_in = files_info["file_names"][i]
        date_str = fname_in[7:11] + fname_in[12:14] + fname_in[15:17]
        file_name = path_out + date_str + "_arm-sgpC1_halo-doppler-lidar-UHOH_wind-vad-eleXX.nc"
        print("Writing " + file_name)
        nc_tools.write_nc_(date_str, file_name, data_out)
