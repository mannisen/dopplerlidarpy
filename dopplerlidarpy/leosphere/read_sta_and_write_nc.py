#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module converts original Leosphere files from sta file format to netcdf4.

Required inputs are paths to read and write files and site name. Only one file per day is assumed to exist and the date
written on the output netcdf4 file is read from the data. By default the *.sta files are assumed to have 'sta' ending,
encoding 'latin' and 39 header lines.

Examples:
        $ python3 leosphere_sta2nc(site_name, path_in, path_out)
        $ python3 leosphere_sta2nc(site_name, path_in, path_out, file_type='sta', file_encoding='latin', header_lines=39)

Created on 2019-07-05
Antti Manninen
Finnish Meteorological Institute
dopplerlidarpy@fmi.fi
"""

import os
import re
import pathlib
import pandas as pd
import numpy as np
from datetime import datetime
from dopplerlidarpy.utilities import dl_var_atts as dl_atts
from dopplerlidarpy.utilities.nc_tools import write_nc_
from dopplerlidarpy.utilities.general_utils import look_for_from
pd.options.mode.chained_assignment = None  # default='warn'


def leosphere_sta2nc(site_name, path_in, path_out, file_type="sta", file_encoding="latin", header_lines=39):
    """Reads original Leosphere files and writes them into netcdf4 files.

    :param site_name:  (str)
    :param path_in:  (str)
    :param path_out:  (str)
    :param file_type:  (str)
    :param file_encoding:  (str)
    :param header_lines:  (int)
    :return:
    """

    # get pathlib object
    path_to = pathlib.Path(path_in)

    # search files in path
    for file_name in path_to.glob("*." + file_type):
        # Check that file is not empty
        if os.stat(file_name).st_size != 0:
            # get headers
            with open(file_name, encoding=file_encoding) as f:
                # initialize
                keys_ = []
                values_ = []
                # iterate over header lines
                for i in range(header_lines):
                    # read line at a time
                    line = f.readline()
                    # collect keys and values, separated with '=' in file
                    keys_.append(line[0:line.find('=')].replace('\n', ''))
                    values_.append(line[line.find('=') + 1:len(line)].replace('\n', ''))
                # close file
                f.close()

                # Remove values and keys that match, i.e. are comments in data file
                empties_ = [i for i, j in zip(keys_, values_) if i == j]
                for irm in empties_:
                    keys_.remove(irm)
                    values_.remove(irm)

                # parse lat and lon
                igps = look_for_from('GPS Location', keys_)
                ll = values_[igps[0]]
                if ll.find('N') != -1:
                    latitude = float(ll[ll.find('Lat:') + 4:ll.find('N') - 1])
                    lat_NS = "N"
                else:
                    latitude = float(ll[ll.find('Lat:') + 4:ll.find('S') - 1])
                    lat_NS = "S"
                if ll.find('E') != -1:
                    longitude = float(ll[ll.find('Long:') + 5:ll.find('E') - 1])
                    lon_WE = "E"
                else:
                    longitude = float(ll[ll.find('Long:') + 5:ll.find('W') - 1])
                    lon_WE = "W"
                keys_.__delitem__(igps[0])
                values_.__delitem__(igps[0])

                # parse height
                iheight = look_for_from('Altitudes (m)', keys_)
                heights = np.array(values_[iheight[0]][1:].replace('\t', ',').split(','), dtype=np.float64)
                keys_.__delitem__(iheight[0])
                values_.__delitem__(iheight[0])

                # Remove units from names and put them in values (as str)
                keys_clean = []
                values_clean = []
                for ikey_, ivalue_ in zip(keys_, values_):
                    # Clean empty space and special characters
                    ikey_ = ikey_.replace(" / ", "_per_")
                    ikey_ = ikey_.replace(" ", "")
                    # print(ikey_)
                    if "(" in ikey_:
                        ifrom = re.search(r'\(.*\)', ikey_).start()
                        ito = re.search(r'\(.*\)', ikey_).end()
                        keys_clean.append(str(ikey_[:ifrom]))
                        values_clean.append(str(ivalue_) + " " + str(ikey_[ifrom:ito]))
                    else:
                        keys_clean.append(ikey_)
                        values_clean.append(ivalue_)

                # read data from file skipping header lines
                df = pd.read_csv(file_name, sep="\t", header=header_lines+1, encoding='latin')

                # parse times
                unix_time_all = []
                time_hrs_utc_all = []
                for i in range(len(df[df.columns[0]])):
                    # extract string
                    time_str = df[df.columns[0]][i]
                    # parse into datetime object
                    time_dt_obj = datetime.strptime(time_str, "%Y/%m/%d %H:%M")
                    # convert to unix time
                    unix_time_all.append(time_dt_obj.timestamp())
                    # convert to time_hrs_utc
                    time_hrs_utc_all.append(time_dt_obj.hour + time_dt_obj.minute / 60)
                if time_hrs_utc_all[-1] < time_hrs_utc_all[-1 - 1]:
                    time_hrs_utc_all[-1] += 24
                df["unix_time"] = unix_time_all
                df["time_hrs_utc"] = time_hrs_utc_all

                # Initialize predefined common Doppler lidar attributes
                time_unix = dl_atts.unix_time_(data=df["unix_time"].values, dim_size=(len(df["unix_time"]),))
                time = dl_atts.time_hrs_utc_(data=df["time_hrs_utc"].values, dim_size=(len(df["unix_time"]),))
                height = dl_atts.height_agl_(data=heights, dim_size=(len(heights),))
                find_ws = [i for i in df.columns.tolist() if "Wind Speed (m/s)" in i]
                ws = dl_atts.wind_speed_(data=df[find_ws].values, dim_size=(len(df["unix_time"]), len(heights)))
                find_wd = [i for i in df.columns.tolist() if "Wind Direction (°)" in i]
                wd = dl_atts.wind_direction_(data=df[find_wd].values, dim_size=(len(df["unix_time"]), len(heights)))

                # Initialize the rest of the attributes
                temp_int = dl_atts.VarBlueprint("temperature_internal", data=df["Int Temp (°C)"].tolist(),
                                                dim_name=("time",), dim_size=(len(df["unix_time"]),),
                                                standard_name="temperature_internal",
                                                long_name="Internal temperature",
                                                units="degrees Celsius")
                temp_ext = dl_atts.VarBlueprint("temperature_external", data=df["Ext Temp (°C)"].tolist(),
                                                dim_name=("time",), dim_size=(len(df["unix_time"]),),
                                                standard_name="temperature_external",
                                                long_name="External temperature",
                                                units="degrees Celsius")
                pressure = dl_atts.VarBlueprint("pressure", data=df["Pressure (hPa)"].tolist(),
                                                dim_name=("time",), dim_size=(len(df["unix_time"]),),
                                                standard_name="pressure",
                                                long_name="Pressure",
                                                units="hPa")
                relative_humidity = dl_atts.VarBlueprint("relative_humidity", data=df["Rel Humidity (%)"].tolist(),
                                                         dim_name=("time",), dim_size=(len(df["unix_time"]),),
                                                         standard_name="relative_humidity",
                                                         long_name="Relative humidity",
                                                         units="percent")
                wiper_count = dl_atts.VarBlueprint("wiper_count", data=df["Wiper count"].tolist(),
                                                   dim_name=("time",), dim_size=(len(df["unix_time"]),),
                                                   standard_name="wiper_count",
                                                   long_name="Wiper count")
                battery_voltage = dl_atts.VarBlueprint("battery_voltage", data=df["Vbatt (V)"].tolist(),
                                                       dim_name=("time",), dim_size=(len(df["unix_time"]),),
                                                       standard_name="battery_voltage",
                                                       long_name="Battery voltage",
                                                       units="V")
                find_cnr = [i for i in df.columns.tolist() if "CNR (dB)" in i]
                cnr = dl_atts.VarBlueprint("cnr", data=df[find_cnr].values, dim_name=("time", "height"),
                                           dim_size=(len(df["unix_time"]), len(heights)), standard_name="cnr",
                                           long_name="Carrier-to-noise ratio", units="dB")
                find_cnr_min = [i for i in df.columns.tolist() if "CNR min (dB)" in i]
                cnr_min = dl_atts.VarBlueprint("cnr_min", data=df[find_cnr_min].values,
                                               dim_name=("time", "height"),
                                               dim_size=(len(df["unix_time"]), len(heights)),
                                               standard_name="cnr_min", long_name="Carrier-to-noise ratio min",
                                               units="dB")
                find_w = [i for i in df.columns.tolist() if "Z-wind (m/s)" in i]
                w = dl_atts.VarBlueprint("w", data=df[find_w].values, dim_name=("time", "height"),
                                         dim_size=(len(df["unix_time"]), len(heights)), standard_name="w",
                                         long_name="w-wind", units="m s-1")
                find_w_std = [i for i in df.columns.tolist() if "Z-wind Dispersion (m/s)" in i]
                w_std = dl_atts.VarBlueprint("w_std", data=df[find_w_std].values,
                                             dim_name=("time", "height"),
                                             dim_size=(len(df["unix_time"]), len(heights)), standard_name="w_std",
                                             long_name="w-wind standard deviation", units="m s-1")
                find_dsb = [i for i in df.columns.tolist() if "Dopp Spect Broad (m/s)" in i]
                dopp_spec_broad = dl_atts.VarBlueprint("doppler_spectral_broadening",
                                                       data=df[find_dsb].values,
                                                       dim_name=("time", "height"),
                                                       dim_size=(len(df["unix_time"]), len(heights)),
                                                       standard_name="doppler_spectral_broadening",
                                                       long_name="Doppler spectral broadening",
                                                       units="m s-1")
                find_da = [i for i in df.columns.tolist() if "Data Availability (%)" in i]
                data_avail = dl_atts.VarBlueprint("data_availability", data=df[find_da].values,
                                                  dim_name=("time", "height"),
                                                  dim_size=(len(df["unix_time"]), len(heights)),
                                                  standard_name="data_availability", long_name="Data availability",
                                                  units="percent")
                find_ws_std = [i for i in df.columns.tolist() if "Wind Speed Dispersion (m/s)" in i]
                ws_std = dl_atts.VarBlueprint("wind_speed_std", data=df[find_ws_std].values,
                                              dim_name=("time", "height"),
                                              dim_size=(len(df["unix_time"]), len(heights)),
                                              standard_name="wind_speed_std",
                                              long_name="Wind speed standard deviation", units="m s-1")
                find_ws_min = [i for i in df.columns.tolist() if "Wind Speed min (m/s)" in i]
                ws_min = dl_atts.VarBlueprint("wind_speed_min", data=df[find_ws_min].values,
                                              dim_name=("time", "height"),
                                              dim_size=(len(df["unix_time"]), len(heights)),
                                              standard_name="wind_speed_min",
                                              long_name="Wind speed min", units="m s-1")
                find_ws_max = [i for i in df.columns.tolist() if "Wind Speed max (m/s)" in i]
                ws_max = dl_atts.VarBlueprint("wind_speed_max", data=df[find_ws_max].values,
                                              dim_name=("time", "height"),
                                              dim_size=(len(df["unix_time"]), len(heights)),
                                              standard_name="wind_speed_max",
                                              long_name="Wind speed max", units="m s-1")
                lat = dl_atts.VarBlueprint("latitude", data=latitude, standard_name="latitude",
                                           long_name="Latitude", units="degrees {}".format(lat_NS))
                lon = dl_atts.VarBlueprint("longitude", data=longitude, standard_name="longitude",
                                           long_name="Longitude", units="degrees {}".format(lon_WE))

                # Attributes passed from Windcube files
                additional_gatts = {}
                for key_, value_ in zip(keys_clean, values_clean):
                    additional_gatts[key_] = value_

                # Prepare and write
                obs = [time, height, time_unix, ws, wd, temp_int, temp_ext, pressure, relative_humidity,
                       wiper_count, battery_voltage, cnr, cnr_min, w, w_std, dopp_spec_broad, data_avail, ws_std,
                       ws_min, ws_max, lat, lon]
                date_txt = time_dt_obj.strftime("%Y%m%d")
                file_name_ = path_out + date_txt + "_" + site_name + "_leosphere.nc"
                print("Writing " + path_out + date_txt + "_" + site_name + "_leosphere.nc")
                write_nc_(date_txt, file_name_, obs, add_gats=additional_gatts,
                          title_="Leosphere Windcube retrievals",
                          institution_="Windcube deployed at FMI in Helsinki, Finland",
                          location_="Kumpula, Helsinki, Finland",
                          source_="ground-based remote sensing",
                          )

        else:
            print("Skipping empty file: {}\n".format(path_to + file_name))
