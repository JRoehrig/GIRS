import os

import numpy as np
from osgeo import gdal


class RasterParameters:
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

    def check_value_length(self, v):
        n = self.number_of_bands
        try:
            if n < len(v):
                v = v[:n]
            elif n > len(v):
                v = v[-1] * (n - len(v))
        except TypeError, te:
            v = [v] * n
        except:
            raise
        return v

    def set_nodata(self, nodata):
        self.nodata = self.check_value_length(nodata)

    def pixel_size(self):
        x_min, y_max = self.pixel_to_world(0, 0)
        x_max, y_min = self.pixel_to_world(self.RasterXSize, self.RasterYSize)
        return (x_max - x_min)/self.RasterXSize , (y_max-y_min)/self.RasterYSize

    def pixel_to_world(self, x, y):
        return self.geo_trans[0] + (x * self.geo_trans[1]), self.geo_trans[3] + (y * self.geo_trans[5])

    def get_extent_world(self):
        x_min0, y_max0 = self.pixel_to_world(0, 0)
        x_max0, y_min0 = self.pixel_to_world(self.RasterXSize, self.RasterYSize)
        return x_min0, x_max0, y_min0, y_max0

    def world_to_pixel(self, x, y, np_func=np.round):
        return np_func(np.divide(x - self.geo_trans[0], self.geo_trans[1])).astype(np.int), \
               np_func(np.divide(y - self.geo_trans[3], self.geo_trans[5])).astype(np.int)

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


def get_parameters(ds):
    """
    :param ds: dataset or filename
    :return: RasterParameters object
    """
    try:
        if ds.endswith('.zip'):
            ds = gdal.Open('/vsizip/' + ds + '/' + os.path.basename(ds[:-4]))
        elif ds.endswith('.gz'):
            ds = gdal.Open('/vsigzip/' + ds + '/' + os.path.basename(ds[:-3]))
        else:
            ds = gdal.Open(ds)
    except Exception:
        pass
    xs = ds.RasterXSize
    ys = ds.RasterYSize
    gt = ds.GetGeoTransform()
    rs = ds.GetProjection()
    nb = ds.RasterCount
    nd = [ds.GetRasterBand(i+1).GetNoDataValue() for i in range(ds.RasterCount)]
    dt = [ds.GetRasterBand(i+1).DataType for i in range(ds.RasterCount)]
    ds = ds.GetDriver().ShortName
    return RasterParameters(xs, ys, gt, rs, nb, nd, dt, ds)


