#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 12:42:15 2019

@author: manninan
"""

def date_format_ret(date_len):
    return {
        '18': '%Y-%m-%d %H:%M:%S',
        '15': '%Y-%m-%d %H:%M',
        '12': '%Y-%m-%d %H',
        '10': '%Y-%m-%d',
    }.get(date_len, 'ABC')