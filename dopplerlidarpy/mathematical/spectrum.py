#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:54:08 2019

Created on 2019-10-24
Antti Manninen
Finnish Meteorological Institute
dopplerlidarpy(at)fmi.fi
"""

import numpy as np
from math import atan2


def lomb_scargle_periodogram(t, x, ofac=4, hifac=1):
    """ calculates the Lomb-Scargle (LS) based power spectrum of a signal unevenly-spaced in time domain.
    http://mres.uni-potsdam.de/index.php/2017/08/22/data-voids-and-spectral-analysis-dont-be-afraid-of-gaps/

    Args:
        t (ndarray, float): time vector
        x (ndarray, float): variable vector
        ofac ():
        hifac ():

    Returns:
        fg (ndarray, float): frequencies
        pxg (ndarray, float): power spectrum 'x'

    """
    # median sampling interval
    int_ = np.nanmedian(np.diff(t))
    # ofac_ = 4  #  oversampling parameter
    # hifac = 1
    fg = np.arange(((2*int_)**(-1)) / (len(x)*ofac), hifac*(2*int_)**(-1), ((2*int_)**-1) / (len(x)*ofac))

    x = x - np.nanmean(x)
    pxg = np.empty(np.size(fg))
    pxg[:] = np.nan

    for k in range(len(fg)-1):

        wrun = 2*np.pi*fg[k]
        pxg[k] = 1 / (2*np.var(x)) * \
                 ((np.sum(np.multiply(x, np.cos(wrun*t -
                                                atan2(np.sum(np.sin(2*wrun*t)),
                                                      np.sum(np.cos(2*wrun*t))) / 2))))**2) / \
                 (np.sum((np.cos(wrun*t - atan2(np.sum(np.sin(2*wrun*t)),
                                                np.sum(np.cos(2*wrun*t))) / 2))**2)) + \
                 ((np.sum(np.multiply(x, np.sin(wrun*t -
                                                atan2(np.sum(np.sin(2*wrun*t)),
                                                      np.sum(np.cos(2*wrun*t))) / 2))))**2) / \
                 (np.sum((np.sin(wrun*t - atan2(np.sum(np.sin(2*wrun*t)),
                                                np.sum(np.cos(2*wrun*t))) / 2))**2))
    return fg, pxg
