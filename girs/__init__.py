from __future__ import print_function
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict


def check_packages():

    error = list()
    try:
        pkg_resources.require(['future', 'numpy', 'scipy', 'pandas', 'gdal', 'matplotlib'])
    except Exception as err:
        error.append('{}'.format(err))


    try:
        from osgeo import gdal
    except ImportError:
        error.append('ERROR: module {} could no be imported'.format('gdal'))

    try:
        from osgeo import ogr
    except ImportError:
        error.append('ERROR: module {} could no be imported'.format('ogr'))

    try:
        from osgeo import osr
    except ImportError:
        error.append('ERROR: module {} could no be imported'.format('osr'))

    # import os
    # if not 'GDAL_DATA' in os.environ:
    #     print('GDAL_DATA is not set in the environment')

    if error:
        return '{}'.format('\n'.join(error))
    else:
        return 'All necessary packages are installed.'


if __name__ == '__main__':
    print(check_packages())
