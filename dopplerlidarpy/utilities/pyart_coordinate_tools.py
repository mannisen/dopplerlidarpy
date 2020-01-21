#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2013, UChicago Argonne, LLC
# All rights reserved.
#
# Copyright 2013 UChicago Argonne, LLC. This software was produced under U.S.
# Government contract DE-AC02-06CH11357 for Argonne National Laboratory (ANL),
# which is operated by UChicago Argonne, LLC for the U.S. Department of Energy.
# The U.S. Government has rights to use, reproduce, and distribute this
# software.  NEITHER THE GOVERNMENT NOR UCHICAGO ARGONNE, LLC MAKES ANY
# WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS
# SOFTWARE.  If software is modified to produce derivative works, such modified
# software should be clearly marked, so as not to confuse it with the version
# available from ANL.
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of UChicago Argonne, LLC, Argonne National
#       Laboratory, ANL, the U.S. Government, nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY UCHICAGO ARGONNE, LLC AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL UCHICAGO ARGONNE, LLC OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import numpy as np
import pyproj


def antenna_to_cartesian(ranges_, azimuths_, elevations_):
    """
    Return Cartesian coordinates from antenna coordinates.
    Parameters
    ----------
    ranges_ : array
        Distances to the center of the radar gates (bins) in kilometers.
    azimuths_ : array
        Azimuth angle of the radar in degrees.
    elevations_ : array
        Elevation angle of the radar in degrees.
    Returns
    -------
    x, y, z : array
        Cartesian coordinates in meters from the radar.
    Notes
    -----
    The calculation for Cartesian coordinate is adapted from equations
    2.28(b) and 2.28(c) of Doviak and Zrnic [1]_ assuming a
    standard atmosphere (4/3 Earth's radius model).
    .. math::
        z = \\sqrt{r^2+R^2+2*r*R*sin(\\theta_e)} - R
        s = R * arcsin(\\frac{r*cos(\\theta_e)}{R+z})
        x = s * sin(\\theta_a)
        y = s * cos(\\theta_a)
    Where r is the distance from the radar to the center of the gate,
    :math:`\\theta_a` is the azimuth angle, :math:`\\theta_e` is the
    elevation angle, s is the arc length, and R is the effective radius
    of the earth, taken to be 4/3 the mean radius of earth (6371 km).
    References
    ----------
    .. [1] Doviak and Zrnic, Doppler Radar and Weather Observations, Second
        Edition, 1993, p. 21.
    """
    theta_e = np.multiply(elevations_, np.pi) / 180.0    # elevation angle in radians.
    theta_a = np.multiply(azimuths_, np.pi) / 180.0      # azimuth angle in radians.
    R = 6371.0 * 1000.0 * 4.0 / 3.0     # effective radius of earth in meters.
    r = ranges_ * 1000.0                 # distances to gates in meters.

    z = (r ** 2 + R ** 2 + 2.0 * r * R * np.sin(theta_e)) ** 0.5 - R
    s = R * np.arcsin(r * np.cos(theta_e) / (R + z))  # arc length in m.
    x = s * np.sin(theta_a)
    y = s * np.cos(theta_a)
    return x, y, z


def cartesian_to_geographic(x, y, projparams):
    """
    Cartesian to Geographic coordinate transform.
    Transform a set of Cartesian/Cartographic coordinates (x, y) to a
    geographic coordinate system (lat, lon) using pyproj or a build in
    Azimuthal equidistant projection.
    Parameters
    ----------
    x, y : array-like
        Cartesian coordinates in meters unless R is defined in different units
        in the projparams parameter.
    projparams : dict or str
        Projection parameters passed to pyproj.Proj. If this parameter is a
        dictionary with a 'proj' key equal to 'pyart_aeqd' then a azimuthal
        equidistant projection will be used that is native to Py-ART and
        does not require pyproj to be installed. In this case a non-default
        value of R can be specified by setting the 'R' key to the desired
        value.
    Returns
    -------
    lon, lat : array
        Longitude and latitude of the Cartesian coordinates in degrees.
    """
    # if isinstance(projparams, dict) and projparams.get('proj') == 'pyart_aeqd':
    #     # Use Py-ART's Azimuthal equidistance projection
    #     lon_0 = projparams['lon_0']
    #     lat_0 = projparams['lat_0']
    #     if 'R' in projparams:
    #         R = projparams['R']
    #         lon, lat = cartesian_to_geographic_aeqd(x, y, lon_0, lat_0, R)
    #     else:
    #         lon, lat = cartesian_to_geographic_aeqd(x, y, lon_0, lat_0)
    # else:
        # Use pyproj for the projection
        # check that pyproj is available
        # if not _PYPROJ_AVAILABLE:
        #     raise MissingOptionalDependency(
        #         "PyProj is required to use cartesian_to_geographic "
        #         "with a projection other than pyart_aeqd but it is not "
        #         "installed")

    proj = pyproj.Proj(projparams)
    lon, lat = proj(x, y, inverse=True)

    return lon, lat
