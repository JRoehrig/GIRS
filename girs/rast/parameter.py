from __future__ import division
from builtins import str
from builtins import range
from builtins import object
from past.utils import old_div
import os
import math
import numpy as np
from osgeo import gdal


class RasterParameters(object):
    def __init__(self, raster_x_size, raster_y_size, geo_trans, srs, number_of_bands, nodata, data_types,
                 driver_short_name=None):
        self.RasterXSize = raster_x_size
        self.RasterYSize = raster_y_size
        self.geo_trans = geo_trans
        self.srs = srs
        self.number_of_bands = number_of_bands
        self.nodata = self.check_value_length(nodata)
        self.data_types = self.check_value_length(data_types)
        self.driverShortName = driver_short_name

    def __repr__(self):
        from girs import srs
        s = srs.srs_from_wkt(self.srs)
        geocs = s.GetAttrValue('GEOGCS')
        projcs = s.GetAttrValue('PROJCS')
        srs = (geocs if geocs else '') + (projcs if projcs else '')
        data_types = ','.join([str(gdal.GetDataTypeName(dt)) for dt in self.data_types])
        return 'DIM[{}, {}, {}] ND{} DT[{}] DRV[{}] SRS[{}] TRANS{}'.format(
            self.number_of_bands, self.RasterXSize, self.RasterYSize, self.nodata, data_types,
            self.driverShortName, srs, self.geo_trans)

    def get_coordinate_system(self):
        return self.srs

    def set_coordinate_system(self, srs):
        self.srs = srs

    def check_value_length(self, v):
        n = self.number_of_bands
        try:
            if n < len(v):
                v = v[:n]
            elif n > len(v):
                v = v[:-1] + [v[-1]] * (n - len(v) + 1)
        except TypeError as te:
            v = [v] * n
        except:
            raise
        return v

    def set_nodata(self, nodata):
        self.nodata = self.check_value_length(nodata)

    def pixel_size(self):
        x_min, y_max = self.pixel_to_world(0, 0)
        x_max, y_min = self.pixel_to_world(self.RasterXSize, self.RasterYSize)
        return old_div((x_max - x_min),self.RasterXSize) , old_div((y_max-y_min),self.RasterYSize)

    def pixel_to_world(self, x, y):
        return self.geo_trans[0] + (x * self.geo_trans[1]), self.geo_trans[3] + (y * self.geo_trans[5])

    def get_extent_world(self):
        x_min0, y_max0 = self.pixel_to_world(0, 0)
        x_max0, y_min0 = self.pixel_to_world(self.RasterXSize, self.RasterYSize)
        return x_min0, x_max0, y_min0, y_max0

    def world_to_pixel(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        return int(math.floor(float(x - self.geo_trans[0]) / self.geo_trans[1])),\
               int(math.floor(float(y - self.geo_trans[3]) / self.geo_trans[5]))

    def extent_world_to_pixel(self, min_x, max_x, min_y, max_y):
        u_min, v_min = self.world_to_pixel(min_x, max_y)
        u_max, v_max = self.world_to_pixel(max_x, min_y)
        geo_trans = list(self.geo_trans)
        geo_trans[0], geo_trans[3] = self.pixel_to_world(u_min, v_min)
        return (u_min, u_max, v_min, v_max),  geo_trans

    def set_driver_short_name(self, filename):
        filename = filename.lower()
        if filename == 'mem':
            self.driverShortName = 'MEM'
        else:
            try:
                self.driverShortName = gdal.IdentifyDriver(filename).ShortName
            except AttributeError:
                self.driverShortName = filename

    def get_default_values(self, values):
        try:
            if self.number_of_bands < len(values):
                values = values[:self.number_of_bands]
            elif self.number_of_bands > len(values):
                values = values[-1] * (self.number_of_bands - len(values))
        except TypeError:
            values = [values] * self.number_of_bands
        except:
            raise
        return values

    def create_array(self, value=np.nan):
        array = np.empty((self.number_of_bands, self.RasterYSize, self.RasterXSize))
        for i in range(self.number_of_bands):
            array[i] = np.empty((self.RasterYSize, self.RasterXSize)) * value
        return array

    def clip(self, x_min, x_max, y_min, y_max):
        """

        :param x_min:
        :param x_max:
        :param y_min:
        :param y_max:
        :return:
        """
        (u_min, u_max, v_min, v_max), geo_trans = self.extent_world_to_pixel(x_min, x_max, y_min, y_max)
        return RasterParameters(u_max - u_min , v_max - v_min, geo_trans, self.srs, self.number_of_bands,
                                self.nodata, self.data_types, self.driverShortName)


def get_parameters(ds):
    """Return the raster parameters defined in the dataset

    :param ds: dataset or filename
    :type ds: gdal.Dataset
    :return: raster parameters
    :rtype: RasterParameters
    """
    try:
        if ds.endswith('.zip'):
            ds = gdal.Open('/vsizip/' + ds + '/' + os.path.basename(ds[:-4]))
        elif ds.endswith('.gz'):
            ds = gdal.Open('/vsigzip/' + ds + '/' + os.path.basename(ds[:-3]))
        else:
            ds = gdal.Open(ds)
    except Exception:
        if isinstance(ds, RasterParameters):
            return ds
    xs = ds.RasterXSize
    ys = ds.RasterYSize
    gt = ds.GetGeoTransform()
    rs = ds.GetProjection()
    nb = ds.RasterCount
    nd = [ds.GetRasterBand(i+1).GetNoDataValue() for i in range(ds.RasterCount)]
    dt = [ds.GetRasterBand(i+1).DataType for i in range(ds.RasterCount)]
    ds = ds.GetDriver().ShortName
    return RasterParameters(xs, ys, gt, rs, nb, nd, dt, ds)


