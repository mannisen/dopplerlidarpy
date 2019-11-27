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
from scipy.special import gamma


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


def kristensen_spectral_intensity(k_, sigma2_w, mu_, lambda_0):
    """Calculates Kristensen spectral intensity model based on inputs.

    Args:
        k_ (scalar or array like): wave number (rad m-1)
        sigma2_w (scalar):
        lambda_0 (scalar): transition wavelength (m)

    Returns:
        kSk (array like): model-based spectral intensity multiplied with wavenumber, Kristensen et al. (1989)

    """
    # mu_ (scalar): parameters controlling curvature of the spectrum
    # a_ (scalar): see Lothon et al. (2009) Eq. (4)
    # mu_ = 1
    # a_ = .69

    # calculate a
    a_ = np.pi * (mu_ * gamma(5/(6*mu_)) / gamma(1/(2*mu_)) * gamma(1/(3*mu_)))

    # calculate integral length scale from inverse of Eq. (3) in Lothon et al. (2009)
    l_w = lambda_0 / (((5/3)*np.sqrt(mu_**2+(6/5)*mu_+1)-((5/3)*mu_+1))**(1/(2*mu_))*((2*np.pi)/a_))
    # calculate model-based spectral intensity
    s_ = np.empty(np.shape(k_))
    s_[:] = np.nan
    for ik in range(len(k_)):
        s_[ik] = (sigma2_w*l_w)/(2*np.pi) * ((3+8*((l_w*k_[ik])/a_)**(2*mu_))/(3*(1+((l_w*k_[ik])/a_)**(2*mu_))**(5/(6*mu_)+1)))

    kSk = np.multiply(k_, s_)

    return kSk
