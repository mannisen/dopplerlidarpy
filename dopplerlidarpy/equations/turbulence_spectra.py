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
    """Estimates the Lomb-Scargle (LS) based power spectrum of a signal unevenly-spaced in time domain.

    Args:
        t (ndarray, float): time vector
        x (ndarray, float): variable vector
        ofac ():
        hifac ():

    Returns:
        fg (ndarray, float): frequencies
        pxg (ndarray, float): power spectrum 'x'

    References:
        http://mres.uni-potsdam.de/index.php/2017/08/22/data-voids-and-spectral-analysis-dont-be-afraid-of-gaps/
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


def kristensen_model_a_parameter(mu_):
    """Calculate the 'a' parameter for Kristensen spectral intensity model

    Args:
        mu_ (scalar): parameter controlling curvature of the spectrum across the transition from zero to −5/3 slope

    Returns:
        a_ (scalar): see Lothon et al. (2009) Eq. (4)

    References:
        Lothon et al. (2009), https://doi.org/10.1007/s10546-009-9398-y
    """

    # a_ (scalar): see Lothon et al. (2009) Eq. (4)
    a_ = np.pi * (mu_ * gamma(5 / (6 * mu_)) / gamma(1 / (2 * mu_)) * gamma(1 / (3 * mu_)))

    return a_


def integral_scale_l_w(mu_, lambda_0):
    """Calculate integral scale

    Args:
        mu_ (scalar): parameter controlling curvature of the spectrum across the transition from zero to −5/3 slope
        lambda_0 (scalar): transition wavelength (m)

    Returns:
        l_w (scalar): characteristic scale over which the vertical velocity is significantly correlated with itself,
                      thus it represents a characteristic size of the individual eddies (m)

    References:
        Lenschow and Stankov (1986), https://doi.org/10.1175/1520-0469(1986)043<1198:LSITCB>2.0.CO;2
        Lothon et al. (2009), https://doi.org/10.1007/s10546-009-9398-y

    """

    # a_ (scalar): see Lothon et al. (2009) Eq. (4)
    # a_ = np.pi * (mu_ * gamma(5/(6*mu_)) / gamma(1/(2*mu_)) * gamma(1/(3*mu_)))
    a_ = kristensen_model_a_parameter(mu_)

    # calculate integral length scale from inverse of Eq. (3) in Lothon et al. (2009)
    l_w = lambda_0 / (((5/3)*np.sqrt(mu_**2+(6/5)*mu_+1)-((5/3)*mu_+1))**(1/(2*mu_))*((2*np.pi)/a_))

    return l_w


def kristensen_spectral_intensity(k_, sigma2_w, mu_, lambda_0):
    """Calculates Kristensen spectral intensity model based on inputs.

    Args:
        k_ (scalar or array like): wave number (rad m-1)
        sigma2_w (scalar):
        mu_ (scalar): parameter controlling curvature of the spectrum across the transition from zero to −5/3 slope
        lambda_0 (scalar): transition wavelength (m)

    Returns:
        k_sk (array like): model-based spectral intensity multiplied with wavenumber, Kristensen et al. (1989)

    References:
        Kristensen et al. (1989), https://doi.org/10.1007/BF00122327
        Lothon et al. (2009), https://doi.org/10.1007/s10546-009-9398-y
        Tonttila et al. (2015), https://doi.org/10.5194/acp-15-5873-2015
    """
    s_ = np.empty(np.shape(k_))
    s_[:] = np.nan
    if np.all(np.isfinite([sigma2_w, mu_, lambda_0])):
        # a_ (scalar): see Lothon et al. (2009) Eq. (4)
        # a_ = np.pi * (mu_ * gamma(5/(6*mu_)) / gamma(1/(2*mu_)) * gamma(1/(3*mu_)))
        a_ = kristensen_model_a_parameter(mu_)

        # calculate integral length scale from inverse of Eq. (3) in Lothon et al. (2009)
        # l_w = lambda_0 / (((5/3)*np.sqrt(mu_**2+(6/5)*mu_+1)-((5/3)*mu_+1))**(1/(2*mu_))*((2*np.pi)/a_))
        l_w = integral_scale_l_w(mu_, lambda_0)

        # calculate model-based spectral intensity
        for ik in range(len(k_)):
            s_[ik] = (sigma2_w*l_w)/(2*np.pi)*((3+8*((l_w*k_[ik])/a_)**(2*mu_))/(3*(1+((l_w*k_[ik])/a_)**(2*mu_))**(5/(6*mu_)+1)))

        k_sk = np.multiply(k_, s_)

    else:
        k_sk = np.empty(np.shape(k_))
        k_sk[:] = 0

    return k_sk
