#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:10:04 2019

@author: manninan
"""

import numpy as np


def f2lambda(wave_speed_, f_):
    """

    Args:
        wave_speed_ (scalar or array-like): Wave speed (m s-1)
        f_ (scalar or array-like): Frequency (Hz)

    Returns:
        lambda (scalar or array-like): Wavelength (m)

    """
    return np.divide(wave_speed_, f_)


def lambda2f(wave_speed_, lambda_):
    """

    Args:
        wave_speed_ (scalar or array-like): Wave speed (m s-1)
        lambda_ (scalar or array-like): Wavelength (m)

    Returns:
        f (scalar or array-like): Frequency (Hz)

    """
    return np.divide(wave_speed_, lambda_)


def lambda2k(lambda_):
    """

    Args:
        lambda_ (scalar or array-like): Wavelength (m)

    Returns:
        k  (scalar or array-like): Wavenumber (rad m-1)

    """
    return np.divide((2*np.pi), lambda_)


def k2lambda(k):
    """

    Args:
        k (scalar or array-like): Wavenumber (rad m-1)

    Returns:
        lambda (scalar or array-like): Wavelength (m)

    """
    return np.divide((2*np.pi), k)


