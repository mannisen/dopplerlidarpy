#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:24:23 2019

@author: manninan
"""

in_put = {
        "i": {"raw":["txt","nc"],
                    "uncalibrated":{
                            "stare":["co","cross"],
                            "vad":"-e",
                            "rhi":"-a"
                            },
                    "calibrated":{
                            "stare":["co","cross"],
                            "vad":"-e",
                            "rhi":"-a"
                            },
                    "product":{
                            "wind_vad":"-e",
                            "wind_dbs":"-b",
                            "wind_shear":["-e","-b"],
                            "epsilon":["-e","-b"],
                            "sigma2_vad":"-e",
                            "wstats":None,
                            "cloud_precip":None,
                            "attbeta_velo_covar":None,
                            "abl_classification":None
                            }
                    }
}
                    
#print(in_put)
print(in_put["i"].keys())
a = list(in_put["i"]["product"].keys())
print(a)