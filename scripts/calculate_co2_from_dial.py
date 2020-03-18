#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

import numpy as np

N_co2 = 1 / (2 * (abs_cross_section_co2(lambda_on) - abs_cross_section_co2(lambda_off) * delta_r)) * np.log((E_lambda_off(r_top) * E_lambda_on(r_bottom)) / (E_lambda_on(r_top) * E_lambda_off(r_bottom)))

# Symbols:
# sigma: absorption cross section

xco2 = np.log((E_lambda_off(r_top) * E_lambda_on(r_bottom)) / (E_lambda_on(r_top) * E_lambda_off(r_bottom))) / \
       (2 * delta_r * (sigma_co2_lambda_on(range_) - sigma_co2_lambda_off(range_)))

# measured differential absorption optical depth (DAOD),
tau = np.polynomial.chebyshev(x, y, 1)