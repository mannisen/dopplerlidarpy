#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:10:04 2019

@author: manninan
"""

from codecs import decode
import numpy as np
import struct


def wswd2vr(ws, wd, azi):
    """

    Args:
        ws (float): wind speed
        wd (float): wind from direction
        azi (ndarray): azimuth directions

    Returns:
        v_r (ndarray): radial velocities

    """
    # Flip and phase shift
    dir_rad = np.deg2rad(azi) - np.deg2rad(wd - 270)

    # Check folding
    dir_rad = np.where(dir_rad > (2*np.pi), dir_rad - 2 * np.pi, dir_rad)
    dir_rad = np.where(dir_rad < 0, dir_rad + 2 * np.pi, dir_rad)

    return np.sin(dir_rad) * ws


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


def bin_to_float(b):
    """ Convert binary string to a float. """
    bf = int_to_bytes(int(b, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]


def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]


def float_to_bin(value):  # For testing.
    """ Convert float to 64-bit binary string. """
    [d] = struct.unpack(">Q", struct.pack(">d", value))
    return '{:064b}'.format(d)
