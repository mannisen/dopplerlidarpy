#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np

# Known constants
_DELTA_RANGE = 30
_POWER_OUT_LAMBDA_ON = 1
_POWER_OUT_LAMBDA_OFF = 1
_PLANCK_CONSTANT = 1
_BOLTZMANN_CONSTANT = 1
_SPEED_OF_LIGHT = 1
_RECEIVER_BANDWIDTH = 1
_ROO_ZERO = 1


def effective_receiver_area(range_, D_eff, f_eff, lambda_):
    """

    Args:
        range_ (array-like, float): range to instrument
        D_eff (float): has to be estimated with using Pentikainen et al. (2020) method
        f_eff (float): has to be estimated with using Pentikainen et al. (2020) method
        lambda_ (float): wavelength of the laser

    Returns:
        A_e (array-like, float): effective receiver area as a function of range

    """

    return np.divide(np.pi * D_eff**2, 4 * (1 + (np.divide(np.pi * D_eff**2, 4 * lambda_ * range_))**2) *
                    (1 - range_ / f_eff)**2 +
                    (D_eff / (2 * _ROO_ZERO))**2)


def optical_frequency(lambda_):
    """

    Args:
        lambda_:

    Returns:

    """
    return _SPEED_OF_LIGHT / lambda_


def focus_function(range_, D_eff, f_eff, lambda_):
    """

    Args:
        range_:
        D_eff:
        f_eff:
        lambda_:

    Returns:

    """
    return np.divide(effective_receiver_area(range_, D_eff, f_eff, lambda_), range_**2)


def beta_att(snr_, T_f):


def xco2(range_, delta_sigma_abs, beta_att_lambda_on, beta_att_lambda_off, power_background):

    n_c = np.empty([len(range_), 1])
    n_c[:] = 0
    power_in_lambda_on = _POWER_OUT_LAMBDA_ON * _DELTA_RANGE * beta_att_lambda_on + power_background
    power_in_lambda_off = _POWER_OUT_LAMBDA_OFF * _DELTA_RANGE * beta_att_lambda_off + power_background

    for i in range(len(range_)):

        n_c[i] = 1 / (2 * delta_sigma_abs * _DELTA_RANGE) * \
            np.log((power_in_lambda_off[i+1] - power_background) / (power_in_lambda_on[i+1] - power_background) *
                   (power_in_lambda_on[i] - power_background) / (power_in_lambda_off[i] - power_background))

    return n_c


