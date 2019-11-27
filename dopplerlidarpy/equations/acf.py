#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:54:08 2019

@author: manninan
"""


import numpy as np


def acf_newsom(x, num_lags=40):
    """Slow autocorrelation function calculated using equation by Newsom et al. (2019), accessed from
    https://www.arm.gov/publications/tech_reports/doe-sc-arm-tr-149.pdf

    Args:
        x (array like): input signal
        num_lags (int): number of lags

    Returns:
        acf (array like): autocorrelation functions

    """
    # initialize as nans
    acf = np.empty([num_lags, 1])
    acf[:] = np.nan
    for i in range(num_lags):
        # initialize as nans
        x_all = np.empty([len(x)-1-i, 1])
        x_all[:] = np.nan
        for j in range(len(x)-1-i):
            x_all[j] = np.multiply(x[j], x[j+i])
        acf[i] = np.divide(np.nansum(x_all[:]), (len(x)-1))

    return np.array(acf)


def acf_slow_normalized_partial(x, lags=range(40)):
    """Slow "partial" autocorrelation using numpy.corrcoef, partial = calculates mean and var at each lag
    https://stackoverflow.com/a/51168178

    Args:
        x (array like): input signal, time-series
        lags (array like): number of lags, default 0,1,...,39,40

    Returns:
        acf (array like): autocorrelation function

    """

    acf = np.array([1. if lag == 0 else np.corrcoef(x[lag:], x[:-lag])[0][1] for lag in lags])
    return acf


def acf_fast_unnormalized(x):
    """Fast autocorrelation using FFT, unnormalized, non-partial
    https://stackoverflow.com/a/52361803

    Args:
        x (array like): input signal

    Returns:
        r2 (array like): autocorrelation as a function of lag

    """
    r2 = np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    return r2[:len(x)//2]


def acf_fast_normalized(x):
    """Fast autocorrelation using FFT, normalized, non-partial
    https://stackoverflow.com/a/52361803

    Args:
        x (array like): input signal

    Returns:
        c (array like): autocorrelation as a function of lag

    """
    r2 = np.fft.ifft(np.abs(np.fft.fft(x))**2).real
    c = (r2/x.shape-np.mean(x)**2)/np.std(x)**2
    return c[:len(x)//2]