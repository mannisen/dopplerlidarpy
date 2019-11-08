#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:18:21 2019

@author: manninan
"""

from utilities.config_utils import getconfig

site_name = 'arm-graciosa'
date_txt = '2016-12-31_23:59:59'

df = getconfig(site_name,date_txt)

print(df)
