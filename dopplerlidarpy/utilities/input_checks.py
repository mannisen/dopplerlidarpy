#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 12:34:14 2019

@author: manninan
"""


def validate_site(site_name):
    """
    Checks is input 'site_name' a string.
    """
    error_msg = "The input 'site_name' must be a string."
    try:
        isinstance(site_name, str)
    except ValueError:
        raise ValueError(error_msg)


def get_dl_input_list():
    """

    Returns:
        dl_inputs (dict): Dictionary listing of all possible command line inputs for Doppler lidar toolbox

    """

    dl_inputs = {
        'instruments': [
            'halo',
            'windcube'],
        'levels': [
            'raw',
            'uncalibrated',
            'calibrated',
            'product'],
        'measurements': [
            'stare',
            'vad',
            'rhi'],
        'products': [
            'windvad',
            'winddbs',
            'wstats',
            'windshear',
            'windshear_vad',
            'windshear_dbs',
            'epsilon',
            'epsilon_vad',
            'epsilon_dbs',
            'cloud_precip',
            'attbeta_velo_covar',
            'sigma2vad',
            'ABLclass',
            'wind_merged',
            'turbulence_length_scale'],
        'background_file': [
            'nc',
            'txt'],
        'polarization': [
            'co',
            'cross']
    }

    return dl_inputs
