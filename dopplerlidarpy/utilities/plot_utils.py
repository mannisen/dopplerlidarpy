#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:54:08 2019

@author: manninan
"""

import numpy as np


def fill_gaps_with_nans(t_in, x_in, dt=1):
    """

    Args:
        t_in:
        x_in:
        dt:

    Returns:

    """

    # x_out = np.empty([1, np.size(x_in[0, :])])
    x_out = np.empty([1, np.size(x_in, 1)])
    x_out[:] = np.nan
    x_out = np.vstack((x_out, x_in[0, :]))
    t_out = np.nan
    t_out = np.vstack((t_out, t_in[0]))
    i_out = int(1)
    i_in = int(1)

    while int(i_in) < np.size(x_in, 0):
        test_ = abs((t_in[i_in] - t_in[i_in - 1])) > (np.nanmedian(np.diff(t_in)) * dt)
        if test_ == 0:

            x_out = np.vstack((x_out, x_in[i_in, :]))
            t_out = np.vstack((t_out, t_in[i_in]))
            i_in += int(1)
            i_out += int(1)
        elif test_ == 1:
            t_out = np.vstack((t_out, t_in[i_in - 1] + np.nanmedian(np.diff(t_in)) * dt))
            tmp = np.empty([1, np.size(x_in, 1)])
            tmp[:] = np.nan
            x_out = np.vstack((x_out, tmp))
            i_out += int(1)
            x_out = np.vstack((x_out, x_in[i_in, :]))
            t_out = np.vstack((t_out, t_in[i_in]))
            i_in += int(1)
            i_out += int(1)

    t_out = np.delete(t_out, 0, 0).squeeze()
    x_out = np.delete(x_out, 0, 0)

    return t_out, x_out
