from __future__ import print_function

def check_packages():
    # TODO: check for gdal not only import, but also whether the binding works (libgdal)
    error = False
    try:
        import numpy
    except ImportError:
        error = True
        print('ERROR: module {} could no be imported'.format('numpy'))

    try:
        import scipy
    except ImportError:
        error = True
        print('ERROR: module {} could no be imported'.format('scipy'))

    try:
        import pandas
    except ImportError:
        error = True
        print('ERROR: module {} could no be imported'.format('pandas'))

    try:
        from osgeo import gdal
    except ImportError:
        error = True
        print('ERROR: module {} could no be imported'.format('gdal'))

    try:
        from osgeo import ogr
    except ImportError:
        error = True
        print('ERROR: module {} could no be imported'.format('ogr'))

    try:
        from osgeo import osr
    except ImportError:
        error = True
        print('ERROR: module {} could no be imported'.format('osr'))

    import os
    if not 'GDAL_DATA' in os.environ:
        print('GDAL_DATA is not set in the environment')

    if not error:
        print('All packages could be loaded')
