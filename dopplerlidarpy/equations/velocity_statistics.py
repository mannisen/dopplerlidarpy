#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:48:23 2019

@author: manninan
"""

from dopplerlidarpy.equations.acf import acf_fast_normalized
from sklearn import linear_model
import numpy as np


def sigma2w_lenschow(velo_sel):
    """Estimates radial velocity unbiased variance by usindg the Lenschow et al. (2000) method

    Args:
        velo_sel (1D array): a vector of velocity values

    Returns:
        velo_var (scalar): unbiased variance

    """

    sigma2_w = np.nanvar(velo_sel[:])  # - np.nanvar(velo_error_sel[:])  --> negative values, TBD
    my_acf_norm = acf_fast_normalized(velo_sel[:])

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor()
    x = np.empty([6, 2])
    x[:, 0] = 1
    x[:, 1] = np.arange(1, 7, 1)
    ransac.fit(x, np.multiply(my_acf_norm[1:7], sigma2_w))
    line_x = np.empty([7, 2])
    line_x[:, 0] = 1
    line_x[:, 1] = np.arange(0, 7, 1)
    line_y_ransac = ransac.predict(line_x)
    # if unbiased variance less than 'biased' variance, no significant amount of noise --> used 'biased' variance
    if line_y_ransac[0] < sigma2_w:
        velo_var = line_y_ransac[0]
    else:
        velo_var = sigma2_w

    return velo_var
