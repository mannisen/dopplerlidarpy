#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:25:50 2019

@author: manninan
"""

from dopplerlidarpy.leosphere.read_sta_and_write_nc import leosphere_sta2nc

site_name = "kumpula"
year_txt = "2018"
path_in = "/home/manninan/Documents/data/leosphere/" + site_name + "/" + year_txt + "/"
path_out = path_in

leosphere_sta2nc(site_name, path_in, path_out)

