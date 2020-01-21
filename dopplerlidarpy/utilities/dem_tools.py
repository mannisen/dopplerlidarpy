#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from osgeo import gdal
# from dopplerlidarpy.utilities import my_args_parser as ap
from dopplerlidarpy.utilities import general_utils as gu
from dopplerlidarpy.utilities import dl_config_utils as cu
from dopplerlidarpy.utilities import FIN_coordinate_tools


def get_dem(args):

    # Get paths
    files_info = gu.get_dl_file_list(args)

    # Get required info from config file
    config_params = cu.get_dl_config(args)

    print(files_info["number_of_files"])
    print(files_info["number_of_files"])

    # Load one at a time
    for i in range(files_info["number_of_files"]):  # assumed one per day

        print("Loading {}".format(files_info["full_paths"][i]))
        raster = gdal.Open(files_info["full_paths"][i])

        if config_params["site_parameters"]




        # Check type of the variable 'raster'
        print("Driver: {}/{}".format(raster.GetDriver().ShortName,
                                raster.GetDriver().LongName))
        print("Size is {} x {} x {}".format(raster.RasterXSize,
                                            raster.RasterYSize,
                                            raster.RasterCount))
        print("Projection is {}".format(raster.GetProjection()))
        geotransform = raster.GetGeoTransform()
        if geotransform:
            print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
            print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))