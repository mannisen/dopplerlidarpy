#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 09:28:31 2019

@author: manninan
"""



# Impbort required libraries
import argparse
from operator import xor
from time_utilities import validate_date

# 
levels = ['raw','uncalibrated','calibrated','product']
measurements = ['stare', 'vad', 'rhi']
products = ['wind_vad', 'wind_dbs', 'wstats','wind_shear_vad', 'wind_shear_dbs', 
            'epsilon_vad', 'epsilon_dbs', 'cloud_precip', 'attbeta_velo_covar',
            'sigma2_vad', 'ABL_class', 'wind_merged']
background_file = ['txt','nc']
polarization = ['co', 'cross']

def strlist2str(in_list):
    '''Converts list of strings to string preserving quotation marks'''
    ret_list = "'" + "', '".join(in_list) + "'"
    return ret_list

def meas_or_prod(args,measurements,products,background_file):
    '''Checks processing level and respective dependent arguments'''
    msg_body = "When processing level {} choose following argument from {}."
    if (args.l[0] == 'uncalibrated' or args.l[0] == 'calibrated') and \
        args.mpb[0] not in measurements:
        raise NameError(msg_body.format(args.l[0],strlist2str(measurements)))
    elif args.l[0] == 'product' and args.mpb[0] not in products_1st_lev+products_2nd_lev:
        raise NameError(msg_body.format(args.l[0],strlist2str(products_1st_lev+products_2nd_lev)))
    elif args.l[0] == 'raw' and args.mpb[0] not in background_file:
        raise NameError(msg_body.format(args.mpb[0],strlist2str(background_file)))

def meas_arg_parser(args):
    '''Checks dependent options for measurements'''        
    msg_body = "'{}' requires argument {} {} to be given also."
    if args.mpb[0] == 'stare' and args.pol==None:
        raise NameError(msg_body.format(args.mpb[0],'-pol',strlist2str(polarization)))
    elif args.mpb[0] == 'vad' and args.e==None:
        raise NameError(msg_body.format(args.mpb[0],'-e','[0-90]'))
    elif args.mpb[0] == 'rhi' and args.a==None:
        raise NameError(msg_body.format(args.mpb[0],'-a','[0-180]'))

def prod_arg_parser(args):
    '''Checks dependent options for products'''
    msg_body = "'{}' requires argument {} {} to be given also."
    if args.mpb[0] == 'wind_vad' and args.e==None:
        raise NameError(msg_body.format(args.mpb[0],'-e','[0-90]'))
    elif args.mpb[0] == 'wind_dbs' and args.b==None:
        raise NameError(msg_body.format(args.mpb[0],'-b','[3-5]'))
    elif args.mpb[0] == 'wind_shear' and xor(args.e==None,args.b==None):
        raise NameError(msg_body.format(args.mpb[0],'-b | -e','[0-90] | [3-5] (respectively)'))
    elif args.mpb[0] == 'epsilon' and xor(args.e==None,args.b==None):
        raise NameError(msg_body.format(args.mpb[0],'-b | -e','[0-90] | [3-5] (respectively)'))
    elif args.mpb[0] == 'sigma2_vad' and args.e==None:
        raise NameError(msg_body.format(args.mpb[0],'-e','[0-90]'))

def optional_arg_parser(args):
    '''Check optional arguments'''
    msg_body = "Optional argument {} is accepted only with '{}', not with '{}'..."
    if args.mpb[0]!='stare' and args.pol!=None:
        raise NameError(msg_body.format('-pol','stare',args.mpb[0]))
    elif args.mpb[0]!='rhi' and args.a!=None:
        raise NameError(msg_body.format('-a','rhi',args.mpb[0]))
    elif (args.mpb[0]!='vad' or args.mpb[0]!='wind_vad' or args.mpb[0]!='epsilon') and args.e!=None:
        raise NameError(msg_body.format('-e','vad',args.mpb[0]))
    elif (args.mpb[0]!='epsilon' or args.mpb[0]!='wind_dbs') and args.b!=None:
        raise NameError(msg_body.format('-b',"'epsilon' or 'wind_dbs'",args.mpb[0]))
    elif (args.mpb[0]!='epsilon' or args.mpb[0]!='wind_vad' or \
        args.mpb[0]!='sigma2_vad') and args.e!=None:
        raise NameError(msg_body.format('-e',"'epsilon', 'wind_vad', or 'sigma2_vad'",args.mpb[0]))
    elif (args.mpb[0]=='raw' or args.mpb[0]=='wstats' or args.mpb[0]=='cloud_precip' or \
        args.mpb[0]=='attbeta_velo_covar') and (args.a!=None or args.e!=None or \
        args.b!=None or args.pol!=None):
        raise NameError("Optional arguments are not accepted with '{}'".format(args.mpb[0]))
    
def my_arg_parser():
    '''Checks given arguments'''
    # Parse inputs
    parser = argparse.ArgumentParser()

    # site
    s_msg = "e.g. 'kuopio'"
    parser.add_argument('site', type=str, help=s_msg, metavar="site")
    
    # date(s)
    d_format = "'YYYY-mm-dd', 'YYYY-mm-dd_HH', 'YYYY-mm-dd_HH:MM', 'YYYY-mm-dd_HH:MM:SS'"
    parser.add_argument('start_date', nargs=1,type=str, help=d_format)
    parser.add_argument('end_date', nargs=1,type=str, help=d_format)
    
    # processing levels
    l_msg = strlist2str(levels)
    parser.add_argument('l',nargs=1, type=str, help=l_msg, choices=levels, 
                        metavar="level_of_processing")

    # measurement mode or product name
    mpb_msg = strlist2str(measurements) + strlist2str(products) + strlist2str(background_file)
    mpb_choices = measurements + products
    parser.add_argument('mpb', nargs=1, type=str, help=mpb_msg, choices=mpb_choices, 
                        metavar='measurement | product | background_file')
    # polarization
    pol_msg = strlist2str(polarization)
    parser.add_argument('-pol', nargs=1, type=str, help=pol_msg, 
                        choices=polarization, metavar='polarization')
    # elevation
    ele_msg = "degrees from the horizon, e.g. -e 75"
    parser.add_argument('-e',nargs=1, type=int, help=ele_msg,
                        choices=range(0,90), metavar='elevation')
    # azimuth
    azi_msg = "degrees from North, e.g. -a 180"
    parser.add_argument('-a', nargs=1, type=int, help=azi_msg,
                        choices=range(0,360), metavar='azimuth')
       
    args = parser.parse_args()
    
    # Check processing level and respective dependent arguments
    try:
        meas_or_prod(args,measurements,products,background_file)
    except NameError as err:
        raise(err)
    
    # Check dependent options of measurements
    if args.mpb[0] in measurements:
        try:
            meas_arg_parser(args)
        except NameError as err:
            raise(err)
    # Check dependent options of products
    elif args.mpb[0] in products:
        try:
            prod_arg_parser(args)
        except NameError as err:
            raise(err)
                    
    # Check date formats
    try:
        validate_date(args.start_date)
    except ValueError as err:
        raise(err)
    try:
        validate_date(args.end_date)
    except ValueError as err:
        raise(err)
        
if __name__ == '__main__':
    my_arg_parser()
    

    