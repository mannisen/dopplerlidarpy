#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from netCDF4 import Dataset
import numpy as np

leosphere_file_name = ''

# read from netcdf file
print("Loading {}".format(leosphere_file_name))
f = Dataset(leosphere_file_name, "r")
data = f.groups["Sweep_246335"]

time_ = np.array(f.variables[afield][:])
range_ = np.array(f.variables["range"][:])
velo = np.array(f.variables["v_raw"][:])
velo_error = np.array(f.variables["v_error"][:])
beta_ = np.array(f.variables["beta_raw"][:])
day_ = "{:1g}".format(getattr(f, "day"))
month_ = "{:1g}".format(getattr(f, "month"))
year_ = "{:1g}".format(getattr(f, "year"))

    f.close()