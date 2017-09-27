__author__ = 'roehrig'
"""
Created on 14.10.2013

@author: Roehrig
"""


def string_to_float(s, nan=None, warn=False):
    try:
        return float('.'.join(s.strip().split(',')) if isinstance(s, basestring) else s)
    except ValueError, e:
        if warn:
            raise e
        return nan

