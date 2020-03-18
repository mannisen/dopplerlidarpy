#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:48:23 2019

@author: manninan
"""


import os
from netCDF4 import Dataset
from datetime import datetime
from dopplerlidarpy.utilities import nc_tools
from dopplerlidarpy.utilities.time_utils import datetime_range
from dopplerlidarpy.equations.velocity_statistics import sigma2w_lenschow
from dopplerlidarpy.utilities.dl_var_atts import VarBlueprint
import numpy as np

leo_name = "WLS400s-113"
leo_path = ""

start_date = datetime(2019, 12, 31, 23, 59, 59)
end_date = datetime(2020, 1, 1, 0, 0, 2)
dt = 3 * 60  # 3 mins
from_ = start_date.timestamp()
to_ = dt
tres = "{:1g}min".format(round(dt / 60))

vars_leo = ["time", "range", "cnr", "radial_wind_speed", "relative_beta"]
vars_halo = ["time", "range", "snr", "v_raw", "beta_raw"]
leo = dict()

# create dimensions
rootgrp = Dataset(file_name_, 'w', format='NETCDF4')
for dname in SIGMA2W["dimenion"]:
    rootgrp = nc_tools.create_nc_dim(rootgrp, dname)

nc_sigma2w = nc_tools.create_nc_var(rootgrp, VarBlueprint("sigma2_w",
                                                          dim_name=("unix_time", "height_agl"),
                                                          long_name="observed velocity variance",
                                                          units="m2 s-2",
                                                          comment="Biased observed variance."))


nc_sigma2_w_unbiased = rootgrp.createVariable("sigma2_w_unbiased", "f8", ("unix_time", "range"))
nc_sigma2_w_unbiased.standard_name = "sigma2_w_unbiased"
nc_sigma2_w_unbiased.long_name = "unbiased vertical velocity variance"
nc_sigma2_w_unbiased.units = "m2 s-2"
nc_sigma2_w_unbiased.comment = "true air motion variance"

for idate_ in datetime_range(start_date, end_date):
    try:
        while to_ <= end_date.timestamp():
            for entry in os.scandir(leo_path):
                if entry.name.startswith(leo_name) and entry.name.endswith(".nc"):
                    full_path = os.path.join(leo_path, entry.name)
                    print("Loading {}".format(full_path))
                    nc_leo = Dataset(full_path, "r")
                    leo_pick = from_ <= nc_leo.variables["time"] < to_

                    for var_leo in vars_leo:
                        leo["time"] = np.mean(nc_leo.variables["time"][leo_pick])
                        leo["height_agl"] = nc_leo.variables["range"]

                        if var_leo not in ["time", "range"]:
                            # Select values
                            velo_sel = nc_leo.variables[var_leo][leo_pick, :]
                            # mean
                            vname = var_leo + "_mean_" + tres
                            leo[vname].append(np.mean(velo_sel, 0))
                            # variance
                            vname = var_leo + "_variance_" + tres
                            leo[vname].append(np.var(velo_sel, 0))
                            # unbiased variance
                            vname = "radial_wind_speed_unbiased_variance_" + tres
                            for j in range(len(leo["range"])):
                                leo[vname][j].append(sigma2w_lenschow(velo_sel[:, j]))

        from_ = to_
        to_ = to_ + dt

    except FileNotFoundError:
        raise  # print("FileNotFoundError: {0}".format(err))

