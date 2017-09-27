import operator
import os
import tarfile

import numpy as np
from PIL import Image, ImageDraw
from osgeo import gdal
from osgeo import ogr, gdalnumeric
from osgeo.gdalconst import GA_ReadOnly, GA_Update
from girs.util.raster import get_driver
import parameter
from gdals import gdal_merge

'''
See:
http://www.gdal.org/gdal_utilities.html
http://pcjericks.github.io_ex/py-gdalogr-cookbook/index.html
https://pypi.python.org/pypi/GDAL/
http://effbot.org/imagingbook/imagedraw.htm
'''


# ===================== use Python Exceptions =================================
gdal.UseExceptions()


# =============================================================================
# Raster
# =============================================================================
class Raster(object):

    def __init__(self, filename):
        """
        filename (str): 'MEM' or file name
        """
        self.dataset = None
        if filename.endswith('.zip'):
            self.filename = filename[:-4]
            self.compressed = '.zip'
        elif filename.endswith('.gz'):
            self.filename = filename[:-3]
            self.compressed = '.gz'
        else:
            self.filename = filename
            self.compressed = None

    def __repr__(self):
        return self.filename + ' ' + self.get_parameters().__repr__()

    def info(self, **kwargs):
        return info(self.dataset, **kwargs)

    def get_filename(self):
        return self.filename

    def get_compressed_filename(self):
        return self.filename + self.compressed if self.compressed else self.filename

    def is_compressed(self):
        return self.compressed is not None

    def get_parameters(self):
        return parameter.get_parameters(self.dataset)

    def copy(self, name, driver_short_name='GTiff'):
        return copy(self, name, driver_short_name)

    def get_nodata(self, band_number=None):
        if band_number:
            try:
                n = len(band_number)
                return [self.dataset.GetRasterBand(i).GetNoDataValue() for i in range(n)]
            except:
                return self.dataset.GetRasterBand(band_number).GetNoDataValue()
        else:
            return [self.dataset.GetRasterBand(i).GetNoDataValue() for i in range(1, self.dataset.RasterCount+1)]

    def get_arrays(self):
        n_cols, n_rows = self.get_raster_size()
        n_bands = self.get_number_of_bands()
        arrays = np.empty((n_bands, n_rows, n_cols))
        for i in range(n_bands):
            arrays[i] = self.get_array(i+1)
        return arrays

    def get_array(self, band_number=1, *args):
        """
            u_min, v_min, u_size, v_size = args[0], args[1], args[2], args[3]
        """
        if not args:
            u_min, v_min = 0, 0
            u_size, v_size = self.dataset.RasterXSize, self.dataset.RasterYSize
        else:
            u_min, v_min, u_size, v_size = args[0], args[1], args[2], args[3]

        return self.dataset.GetRasterBand(band_number).ReadAsArray(u_min, v_min, u_size, v_size)

    def get_masked_array(self, band_number=1, *args):
        arr = self.get_array(band_number, *args)
        arr[arr==self.get_nodata()] = np.nan
        return np.ma.array(arr, mask=np.isnan(arr))

    def get_geotransform(self):
        return self.dataset.GetGeoTransform()

    def transform(self, filename, srs, **kwargs):
        """

        :param filename:
        :param srs:
        :param kwargs:
            format: driver short name
        :return:
        """
        return RasterModifier(filename, gdal.Warp(filename, self.dataset, dstSRS=srs, **kwargs))


    # USE gdal.Warp
    # def gdalwarp(self, dst_filename, **kwargs):
    #     """
    #     """
    #     cmd = 'gdalwarp'
    #
    #     for k, v in kwargs.items():
    #         cmd += ' -' + k + ' ' + v
    #
    #     cmd += ' ' + '"' + self.filename + '" "' + dst_filename + '"'
    #
    #     process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #     stdout = ''.join(process.stdout.readlines())
    #     stderr = ''.join(process.stderr.readlines())
    #
    #     if len(stderr) > 0:
    #         print str(stdout)
    #         raise IOError(stderr)

    def get_coordinate_system(self):
        return self.dataset.GetProjection()

    def get_number_of_bands(self):
        return self.dataset.RasterCount

    def get_band(self, band_number):
        return self.dataset.GetRasterBand(band_number)

    def get_raster_size(self):
        return self.dataset.RasterXSize, self.dataset.RasterYSize

    def get_pixel_size(self):
        return pixel_size(self.get_geotransform())

    def get_extent(self):
        return extent_pixel_to_world(self.get_geotransform(), self.dataset.RasterXSize, self.dataset.RasterYSize)

    def world_to_pixel(self, x, y):
        return world_to_pixel(self.get_geotransform(), x, y)

    def extent_world_to_pixel(self, min_x, max_x, min_y, max_y):
        return extent_world_to_pixel(self.get_geotransform(), min_x, max_x, min_y, max_y)

    def pixel_to_world(self, x, y):
        """
        Return the top-left world coordinate of the pixel
        :param x:
        :param y:
        :return:
        """
        return pixel_to_world(self.get_geotransform(), x, y)

    def get_centroid_world_coordinates(self):
        x_size, y_size = self.get_pixel_size()
        return get_centroid_world_coordinates(self.get_geotransform(),
                                              self.dataset.RasterXSize, self.dataset.RasterYSize,
                                              x_size, y_size)

    # def get_utm_zone(self):
    #     # TODO: remove it
    #     prj = self.dataset.GetProjection()
    #     srs = osr.SpatialReference()
    #     srs.ImportFromWkt(prj)
    #     if srs.IsGeographic() and srs.GetAttrValue('GEOGCS') == 'WGS 84':
    #         lon, lat = self.get_centroid_world_coordinates()
    #         return int(1+(lon+180.0)/6.0), lat >= 0.0
    #     else:
    #         return 0, None

    def get_band_data_types(self):
        ds = self.dataset
        return [ds.GetRasterBand(i).DataType for i in range(1, ds.RasterCount+1)]

    def get_band_data_type(self, band_number=1):
        return self.dataset.GetRasterBand(band_number).DataType

    def clip_by_extent(self, **kwargs):
        """
        Args:
            raster_in (string):

            kwargs:
                'extent' (list): [x_min, x_max, y_min, y_max]
                'layers' (string / Layers): layers file name or Layers object used to set the extent
                'rasters' (string / Layers): rasters file name or Raster object used to set the extent
                'layer_number' (int): layer number. Default is the first layer (0)
                'scale' (float): used in order to scale the extent (scale > 1 creates a buffer)
                'driver' (string): driver name
                'burn_value' (int): burn_value. Default is 1
                'output_raster' (string): filename of the rasters.
        return:
            If 'output_raster' is not set in kwargs, return the new rasters dataset as 'MEM' driver, otherwise None
        """
        from girs.rastfeat import clip
        return clip.clip_by_extent(self, **kwargs)

    def clip_by_vector(self, **kwargs):
        """
        Args:
            raster_in (string):
            layers_in (string / Layers): layers file name or Layers object. The geometry must be line or polygon
            layer_number (int): layer number. Default is the first layer (0)
            scale (float): to scale the bounding box of the features
            driver (string): driver name
            burn_value (int): burn_value. Default is 1
            output_raster (string): filename of the rasters.
            nodata (int/float): nodata

        return:
            If output is not set in kwargs, return the new rasters dataset as MEM driver, otherwise None
        """
        from girs.rastfeat import clip
        return clip.clip_by_vector(self, **kwargs)

    def clip_arrays(self, u0, v0, nu, nv):
        """Clip the all rasters and return a ndarray
        """
        # gdalnumeric.LoadFile does not work with 'MEM'
        # arr0 = gdalnumeric.LoadFile(self.name, u0, v0, nu, nv)
        nb = self.get_number_of_bands()
        if nb == 1:
            return self.clip_array(1, u0, v0, nu, nv)
        else:
            arr = np.empty((nb, nv, nu))
            for ib in range(0, nb):
                arr[ib] = self.clip_array(ib + 1, u0, v0, nu, nv)
            return arr  # ndarray

    def clip_array(self, band_number=1, u_min=0, v_min=0, u_width=None, v_height=None):
        return self.dataset.GetRasterBand(band_number).ReadAsArray(u_min, v_min, u_width, v_height)

    def resample(self, pixel_sizes, resample_alg=gdal.GRA_NearestNeighbour, **kwargs):
        import resample
        return resample.resample(self, pixel_sizes, resample_alg, **kwargs)


# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
class RasterReader(Raster):

    def __init__(self, filename):
        super(RasterReader, self).__init__(filename)
        """
        name (str): 'MEM' or file name
        """
        if self.compressed == '.zip':
            self.dataset = gdal.Open('/vsizip/' + self.get_compressed_filename(), GA_ReadOnly)
        elif self.compressed == '.gz':
            self.dataset = gdal.Open('/vsigzip/' + self.get_compressed_filename(), GA_ReadOnly)
        else:
            self.dataset = gdal.Open(self.filename, GA_ReadOnly)


class RasterEditor(Raster):

    def __init__(self, filename):
        super(RasterEditor, self).__init__(filename)

    def set_nodata(self, nodata=None, band_numbers=None):
        if isinstance(nodata, basestring):
            nodata = [nodata]
        else:
            try:
                len(nodata)
            except TypeError:
                nodata = [nodata]
        if not band_numbers:
            band_numbers = range(1, self.dataset.RasterCount + 1)

        nodata = nodata + nodata[-1:] * (len(band_numbers) - len(nodata))

        for i, nd in enumerate(nodata):
            self.dataset.GetRasterBand(i+1).SetNoDataValue(nodata[i])
        self.dataset.FlushCache()

    def set_array(self, array, band_number=1):
        """
        """
        result = self.dataset.GetRasterBand(band_number).WriteArray(array)
        self.dataset.FlushCache()
        return result

    def set_projection(self, srs):
        result = self.dataset.SetProjection(srs)
        self.dataset.FlushCache()
        return result


class RasterModifier(RasterEditor):

    def __init__(self, filename, dataset=None):
        super(RasterModifier, self).__init__(filename)

        """
        name (str): 'MEM' or file name
        """
        if not dataset:
            if self.compressed == '.zip':
                self.dataset = gdal.Open('/vsizip/' + self.get_compressed_filename(), GA_Update)
            elif self.compressed == '.gz':
                self.dataset = gdal.Open('/vsigzip/' + self.get_compressed_filename(), GA_Update)
            else:
                self.dataset = gdal.Open(self.filename, GA_Update)
        else:
            self.dataset = dataset


class RasterWriter(RasterEditor):

    def __init__(self, filename, raster_parameters):
        super(RasterWriter, self).__init__(filename)
        """
        name (str): 'MEM' or file name
        """
        driver_short_name = raster_parameters.driverShortName
        if driver_short_name:
            drv = gdal.GetDriverByName(driver_short_name)
        else:
            if self.filename.upper() == 'MEM':
                drv = gdal.GetDriverByName('MEM')
            else:
                drv = gdal.IdentifyDriver(filename)

        n_bands = raster_parameters.number_of_bands
        try:
            self.dataset = drv.Create(self.filename, raster_parameters.RasterXSize, raster_parameters.RasterYSize,
                                      n_bands, raster_parameters.data_types[0])
        except RuntimeError, e:
            raise RuntimeError('raster {} is being eventually used (locked)'.format(self.filename))
        self.dataset.SetGeoTransform(raster_parameters.geo_trans)
        self.dataset.SetProjection(raster_parameters.srs)
        for i in range(n_bands):
            if raster_parameters.nodata[i]:
                rb_out = self.dataset.GetRasterBand(i+1)
                rb_out.SetNoDataValue(raster_parameters.nodata[i])
                rb_out.FlushCache()


# =============================================================================
# Functions
# =============================================================================


def info(raster, **kwargs):
    try:
        raster = RasterReader(raster)
        dataset = raster.dataset
    except AttributeError:
        try:
            dataset = raster.dataset
        except AttributeError:
            dataset = raster
    return gdal.Info(dataset, **kwargs)


def get_parameters(raster):
    return parameter.get_parameters(raster)


def copy(raster, dst_filename, driver_short_name=None):
    try:
        raster = RasterReader(raster)
    except AttributeError:
        raster = raster
    dst_driver = get_driver(dst_filename, driver_short_name)
    return RasterModifier(dst_filename, dst_driver.CreateCopy(dst_filename, raster.dataset))


def pixel_size(geo_trans):
    return geo_trans[1], -geo_trans[5]


def world_to_pixel(geo_trans, x, y, np_func=np.trunc):
    # print geo_trans, x, y
    # xOffset = int((x - geo_trans[0]) / geo_trans[1])
    # yOffset = int((y - geo_trans[3]) / geo_trans[5])
    # print xOffset, yOffset,
    xOffset = np_func(np.divide(x - geo_trans[0], geo_trans[1])).astype(np.int)
    yOffset = np_func(np.divide(y - geo_trans[3], geo_trans[5])).astype(np.int)
    # print xOffset, yOffset
    return xOffset, yOffset


def pixel_to_world(geo_trans, x, y):
    """
    Return the top-left world coordinate of the pixel
    :param geo_trans:
    :param x:
    :param y:
    :return:
    """
    return geo_trans[0] + (x * geo_trans[1]), geo_trans[3] + (y * geo_trans[5])


def extent_pixel_to_world(geo_trans, raster_x_size, raster_y_size):
    x_min0, y_max0 = pixel_to_world(geo_trans, 0, 0)
    x_max0, y_min0 = pixel_to_world(geo_trans, raster_x_size, raster_y_size)
    return x_min0, x_max0, y_min0, y_max0


def extent_world_to_pixel(geo_trans, min_x, max_x, min_y, max_y):
    u_min, v_min = world_to_pixel(geo_trans, min_x, max_y)
    u_max, v_max = world_to_pixel(geo_trans, max_x, min_y)
    # u_min, v_min = world_to_pixel(geo_trans, min_x, max_y, np.floor)
    # u_max, v_max = world_to_pixel(geo_trans, max_x, min_y, np.ceil)
    geo_trans[0], geo_trans[3] = pixel_to_world(geo_trans, u_min, v_min)
    return (u_min, u_max, v_min, v_max),  geo_trans


def get_centroid_world_coordinates(geo_trans, raster_x_size, raster_y_size, x_pixel_size, y_pixel_size):
    x0, y0 = pixel_to_world(geo_trans, 0, 0)
    x1, y1 = pixel_to_world(geo_trans, raster_x_size-1, raster_y_size-1)
    x1 += x_pixel_size
    y1 -= y_pixel_size
    return (x0 + x1) * 0.5, (y0 + y1) * 0.5


def get_default_values(number_of_bands, values):
    try:
        if number_of_bands < len(values):
            values = values[:number_of_bands]
        elif number_of_bands > len(values):
            values = values[-1] * (number_of_bands - len(values))
    except TypeError, te:
        values = [values] * number_of_bands
    except:
        raise
    return values


    return result


def get_extent(rasters, extent_type='intersection'):
    """
    Args:
        rasters:
        extent_type (str): intersection or union
    """
    # Get get common extent
    # Get the rasters
    rasters = [RasterReader(ir) if isinstance(ir, basestring) else ir for ir in rasters]
    # Get the extent
    xmin, xmax, ymin, ymax = rasters[0].get_extent()
    et = extent_type.lower()
    if et == 'intersection':
        for r in rasters:
            xmin0, xmax0, ymin0, ymax0 = r.get_extent()
            xmin = max(xmin, xmin0)
            xmax = min(xmax, xmax0)
            ymin = max(ymin, ymin0)
            ymax = min(ymax, ymax0)
    elif et == 'union':
        for r in rasters:
            xmin0, xmax0, ymin0, ymax0 = r.get_extent()
            xmin = min(xmin, xmin0)
            xmax = max(xmax, xmax0)
            ymin = min(ymin, ymin0)
            ymax = max(ymax, ymax0)
    return xmin, xmax, ymin, ymax


def get_pixel_sizes(rasters, minmax='max'):
    rasters = [RasterReader(ir) if isinstance(ir, basestring) else ir for ir in rasters]
    xs, ys = rasters[0].get_pixel_size()
    if minmax == 'max':
        for r in rasters:
            xs0, ys0 = r.get_pixel_size()
            xs = max(xs, xs0)
            ys = max(ys, ys0)
    elif minmax == 'min':
        for r in rasters:
            xs0, ys0 = r.get_pixel_size()
            xs = min(xs, xs0)
            ys = min(ys, ys0)
    return xs, ys


def pim2gim(pim):
    """Converts a Python Imaging Library array to a gdalnumeric image.
    """
    gim = gdalnumeric.fromstring(pim.tostring(), 'b')
    gim.shape = pim.im.size[1], pim.im.size[0]
    return gim


def gim2pim(gim):
    """Converts a gdalnumeric image array to a Python Imaging Library.
    """
    return Image.fromstring('L', (gim.shape[1], gim.shape[0]), (gim.astype('b')).tostring())


def histogram(a, bins=range(0, 256)):
    """Histogram function for multi-dimensional array.
    a = array
    bins = range of numbers to match

    See:
        http://geospatialpython.com/2011/02/clip-rasters-using-shapefile.html
    """
    fa = a.flat
    n = gdalnumeric.searchsorted(gdalnumeric.sort(fa), bins)
    n = gdalnumeric.concatenate([n, [len(fa)]])
    hist = n[1:]-n[:-1]
    return hist


def stretch(gim):
    """Performs a histogram stretch on a gdalnumeric array image.

    See:
        http://geospatialpython.com/2011/02/clip-rasters-using-shapefile.html
    """
    hist = histogram(gim)
    im = gim2pim(gim)
    lut = []
    for b in range(0, len(hist), 256):
        # step size
        step = reduce(operator.add, hist[b:b+256]) / 255
        # create equalization lookup table
        n = 0
        for i in range(256):
            lut.append(n / step)
            n += hist[i+b]
    return pim2gim(im.point(lut))


def merge(rasters,  target_filename, no_data=None):
    merge_par = ['', '-o', target_filename]
    if no_data:
        merge_par += ['-a_nodata', str(no_data)]

    for raster in rasters:
        merge_par.append(raster)
    gdal_merge.main(merge_par)


def geometry_to_gdal_image(geom, geo_trans, u_width, v_height):
    geometry_type = geom.GetGeometryType()
    # Transform features points coordinates into pixels coordinates
    geom2 = geom.GetGeometryRef(0) if geometry_type == ogr.wkbPolygon else geom
    points = [(geom2.GetX(p), geom2.GetY(p)) for p in range(geom2.GetPointCount())]
    pixels = [world_to_pixel(geo_trans, p[0], p[1]) for p in points]

    r_pol = Image.new("L", (u_width, v_height), 1)  # L: 8-bit pixels, black and white,
    i_pol = ImageDraw.Draw(r_pol)
    if geometry_type == ogr.wkbPolygon:
        i_pol.polygon(pixels, 0)
    else:
        i_pol.line(pixels, 0)
    return pim2gim(r_pol)


def no_verbose(_):
    pass


def default_verbose(text, same_line=False):
    if same_line:
        print text,
    else:
        print text


# =============================================================================
# Members and names
# =============================================================================
def get_image_members(filename, substrings=None, sorted_by_name=True):
    """
    :param filename: full path or tar object
    :param substrings:
    :param sorted_by_name:
    :return:
    """
    is_filename = True
    try:
        mode = 'r:gz{' if filename.endswith('.gz') else 'r'
        tar = tarfile.open(filename, mode=mode)
    except AttributeError, e:
        tar = filename
        is_filename = False
    members = [m for m in tar.getmembers()]
    if substrings:
        members = [[m for m in members if s in m.name] for s in substrings]
        members = [m for m0 in members for m in m0]
    if is_filename:
        tar.close()
    return members if not sorted_by_name else sorted(members, key=lambda member: member.name)


def get_image_names(filename, substrings=None, sorted_by_name=True):
    """
    :param filename:
    :param substrings:
    :param sorted_by_name:
    :return:
    """
    return [m.name for m in get_image_members(filename, substrings, sorted_by_name)]


# =============================================================================
# Reading
# =============================================================================
def read_image(filename, band_number=1):
    r = RasterReader(filename)
    array = r.get_array(band_number=band_number)
    nodata = r.get_nodata(band_number)
    array[array == nodata] = np.nan
    return array


def read_images_from_directory(input_dir, substrings=('.tif',), x_off=0, y_off=0, x_size=None, y_size=None):
    filenames = [[f for f in os.listdir(input_dir) if s in f] for s in substrings]
    filenames = [os.path.join(input_dir, f) for f0 in filenames for f in f0]
    return {os.path.basename(filename): read_image(filename, x_off, y_off, x_size, y_size) for filename in filenames}


def check_and_extract_images(f_in, product_dir, verbose=default_verbose):
    f_base = os.path.basename(f_in)
    print f_in, product_dir, f_base

    download_dict = {}
    for f in os.listdir(product_dir):
        download_dict[f] = os.path.getsize(os.path.join(product_dir, f))

    mode = 'r:gz' if f_in.endswith('.gz') else 'r'
    verbose('\tExtracting ' + f_in)
    with tarfile.open(f_in, mode=mode) as tar:
        for member in get_image_members(tar):
            if member.name not in download_dict:
                verbose('\tExtracting ' + member.name)
                tar.extract(member, path=product_dir)
            elif member.size != download_dict[member.name]:
                verbose('\tRe-extracting ' + member.name)
                tar.extract(member, path=product_dir)


# =============================================================================
# Statistics
# =============================================================================
def variance(filenames, filename_out):
    arrays = [read_image(f).astype(float) for f in filenames]
    parameters = RasterReader(filenames[0]).get_parameters()
    parameters.data_types = [gdal.GDT_Float32]
    r_out = RasterWriter(filename_out, parameters)
    r_out.set_array(np.var(np.dstack(arrays), axis=2))


