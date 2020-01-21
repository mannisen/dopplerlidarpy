#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:10:04 2019

@author: manninan
"""
import time
import calendar
import numpy as np
from datetime import datetime, timedelta


def check_date_format(date_txt):
    """Check date format. Not a bullet proof..

    Args:
        date_txt (str): Date in 'YYYY-mm-dd_HH:MM:SS', 'YYYY-mm-dd_HH:MM', or 'YYYY-mm-dd_HH' format.

    Returns (str, boolean): date format

    """

    len_dtxt = len(date_txt)

    if not(len_dtxt == 19 or len_dtxt == 16 or len_dtxt == 13 or len_dtxt == 10):
        date_format = False
        return date_format
    else:
        if len_dtxt == 19:
            date_format = '%Y-%m-%d_%H:%M:%S'
            return date_format
        elif len_dtxt == 16:
            date_format = '%Y-%m-%d_%H:%M'
            return date_format
        elif len_dtxt == 13:
            date_format = '%Y-%m-%d_%H'
            return date_format
        elif len_dtxt == 10:
            date_format = '%Y-%m-%d'
            return date_format


def date_txt2epoch(date_txt, date_format):
    """Converts date in readable format to unix time.

    Args:
        date_txt: (str)  Date in readable format
        date_format: (str) 'YYYY-mm-dd_HH:MM:SS', 'YYYY-mm-dd_HH:MM', or 'YYYY-mm-dd_HH'

    Returns:
        date_epoch: (int) Date in UNIX Epoch time, seconds since 1970-01-01 00:00:00

    """
    try:
        isinstance(date_txt, str)
    except False:
        raise ValueError('First input must be a string.')
    try:
        isinstance(date_format, str)
    except False:
        raise ValueError('Second input must be a string.')
        
    date_epoch = calendar.timegm(time.strptime(date_txt, date_format))
    
    return date_epoch


def epoch2date_txt(posix, date_format):
    """Converts UNIX Epoch time to date in readable format.

    Args:
        posix: (int) Date in UNIX Epoch time, seconds since 1970-01-01 00:00:00
        date_format: (str) 'YYYY-mm-dd_HH:MM:SS', 'YYYY-mm-dd_HH:MM', or 'YYYY-mm-dd_HH'

    Returns:
        date_txt: (str) Date in readable format

    """
    try:
        isinstance(posix, int)
    except False:
        raise ValueError('First input must be an integer.')
    try:
        isinstance(date_format, str)
    except False:
        raise ValueError('Second input must be a string.')

    utc_time = datetime.utcfromtimestamp(posix)
    date_txt = utc_time.strftime(date_format)

    return date_txt


def validate_date(date_txt):
    """Validate given date in user friendly format and returns the format as string.

    Args:
        date_txt: (str) Date

    Returns:
        date_format: (str) Date format 'YYYY-mm-dd_HH:MM:SS', 'YYYY-mm-dd_HH:MM', or 'YYYY-mm-dd_HH'.

    """

    # General error message for date errors
    error_msg = "Date should be given in 'YYYY-mm-dd_HH:MM:SS', 'YYYY-mm-dd_HH:MM', or 'YYYY-mm-dd_HH' formats."

    # Check date format from length of string (not robust at all!)
    date_format = check_date_format(date_txt)
    if not date_format:
        raise ValueError(error_msg)

    # Validate date input
    try:
        datetime.strptime(date_txt, date_format)
    except ValueError:
        raise ValueError(error_msg)

    return date_format


def date_txt2verbose(date_txt_in):
    """

    Args:
        date_txt_in: (str)

    Returns:
        date_txt_out: (str)

    """

    # Check date_txt_in
    try:
        date_format = validate_date(date_txt_in)
    except ValueError as err:
        raise err

    len_d_txt = len(date_txt_in)
    if not(len_d_txt == 19 or len_d_txt == 16 or len_d_txt == 13 or len_d_txt == 10):
        date_txt_out = False
    else:
        if date_format == "%Y-%m-%d_%H:%M:%S":
            date_txt_out = date_txt_in
        elif date_format == "%Y-%m-%d_%H:%M":
            date_txt_out = date_txt_in + ":00"
        elif date_format == "%Y-%m-%d_%H":
            date_txt_out = date_txt_in + ":00:00"
        elif date_format == "%Y-%m-%d":
            date_txt_out = date_txt_in + "_00:00:00"
        else:
            date_txt_out = False
    return date_txt_out


def daterange(start_date, end_date):
    """Generates a range of dates.

    Args:
        start_date: (date) Start date, e.g. date(2001,1,1)
        end_date: (date) End date, e.g. date(2001,12,31)

    Returns:
        : (range) Date range from start_date to end_date

    """

    for n in range(int((end_date - start_date).days)):
        yield start_date + timedelta(n)


def datetime_range(start_datetime, end_datetime):
    """Generates a range of dates in epoch.

    Args:
        start_datetime: (datetime) Start date, e.g. datetime(2001,1,1,00,00,00)
        end_datetime: (datetime) End date, e.g. datetime(2001,12,31,23,59,59)

    Returns:
        : (range) Datetime range from start_datetime to end_datetime

    """
    # date format for this function
    d_format = "%Y%m%d%H%M%S"

    # Convert to Epoch time
    start_date_epoch = date_txt2epoch(start_datetime.strftime(d_format), d_format)
    end_date_epoch = date_txt2epoch(end_datetime.strftime(d_format), d_format)

    # Get range of date-times from start_date to end_date with timedelta 1 second
    for n in range(start_date_epoch, end_date_epoch):
        the_date_txt = epoch2date_txt(n, d_format)
        yield datetime.strptime(the_date_txt, d_format)


def time_hrs_utc2epoch(year_, month_, day_, time_hrs):
    """

    Args:
        year_:
        month_:
        day_:
        time_hrs:

    Returns:

    """
    hrs_ = np.floor(time_hrs[:])
    mins_ = np.floor((time_hrs[:] - hrs_) * 60)
    secs_ = np.round((((time_hrs[:] - hrs_) * 60) - mins_) * 60)

    unix_time_ = np.empty([len(time_hrs)])
    unix_time_[:] = np.nan
    for i in range(len(time_hrs)):
        thedate = "{}-{:02d}-{:02d}_{:02d}:{:02d}:{:02d}".format(year_, int(month_), int(day_), int(hrs_[i]), int(mins_[i]), int(secs_[i]))
        thedate.replace(" ", "0")
        date_format = check_date_format(thedate)
        unix_time_[i] = date_txt2epoch(thedate, date_format)

    return unix_time_
