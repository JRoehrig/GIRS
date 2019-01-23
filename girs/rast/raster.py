from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from past.builtins import basestring
from past.utils import old_div
from builtins import object
import zipfile
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal, gdal_array
from osgeo.gdalconst import GA_ReadOnly, GA_Update
from girs.geom.envelope import merge_envelopes
from . import parameter

# ===================== use Python Exceptions =================================
gdal.UseExceptions()


# =============================================================================
def driver_dictionary():
    """Return the driver dictionary

    The driver dictionary:
        - key: source extension
        - value: driver short name

    :return: driver dictionary
    :rtype: dict
    """
    drivers_dict = {}
    for i in range(gdal.GetDriverCount()):
        drv = gdal.GetDriver(i)
        if drv.GetMetadataItem(gdal.DCAP_RASTER):
            extensions = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
            extensions = extensions.split() if extensions else [None]
            for ext in extensions:
                if ext:
                    if ext.startswith('.'):
                        ext = ext[1:]
                    ext = ext.lower()
                    for ext1 in [e for e in ext.split('/')]:
                        if ext1 not in drivers_dict:
                            drivers_dict[ext1] = []
                        drivers_dict[ext1].append(drv.ShortName)
                else:
                    if None not in drivers_dict:
                        drivers_dict[None] = []
                    drivers_dict[None].append(drv.ShortName)
    return drivers_dict


def get_driver(filename=None, driver_short_name=None):
    """Return a driver.

    If driver_short_name is given, return the corresponding driver.
    filename can be 'MEM' or a file name. If a file name is given, guess
    the driver, returning it only when the correspondence is one to one,
    else return None.

    Filename suffixes, which should work:
    ace2: ACE2, asc: AAIGrid, bin: NGSGEOID, blx: BLX, bmp: BMP, bt: BT, cal: CALS, ct1: CALS, dat: ZMap, ddf: SDTS,
    dem: USGSDEM, dt0: DTED, dt1: DTED, dt2: DTED, e00: E00GRID, gen: ADRG, gff: GFF, grb: GRIB, grc: NWT_GRC,
    gsb: NTv2, gtx: GTX, gxf: GXF, hdf: HDF4, hdf5: HDF5, hf2: HF2, hgt: SRTMHGT, jpeg: JPEG, jpg: JPEG, kea: KEA,
    kro: KRO, lcp: LCP, map: PCRaster, mem: JDEM, mpl: ILWIS, n1: ESAT, nat: MSGN, ntf: NITF, pix: PCIDSK, png:
    PNG, pnm: PNM, ppi: IRIS, rda: R, rgb: SGI, rik: RIK, rst: RST, rsw: RMF, sdat: SAGA, tif: GTiff, tiff: GTiff,
    toc: RPFTOC, vrt: VRT, xml: ECRGTOC, xpm: XPM, xyz: XYZ,

    Filenames with ambiguous suffixes:
    gif: GIF, BIGGIF
    grd: GSAG, GSBG, GS7BG, NWT_GRD
    hdr: COASP, MFF, SNODAS
    img: HFA, SRP
    nc: GMT, netCDF
    ter: Leveller, Terragen

    Without suffix: SAR_CEOS, CEOS, JAXAPALSAR, ELAS, AIG, GRASSASCIIGrid, MEM, BSB, DIMAP, AirSAR, RS2, SAFE,
    HDF4Image, ISIS3, ISIS2, PDS, VICAR, TIL, ERS, L1B, FIT, INGR, COSAR, TSX, MAP, KMLSUPEROVERLAY, SENTINEL2, MRF,
    DOQ1, DOQ2, GenBin, PAux, MFF2, FujiBAS, GSC, FAST, LAN, CPG, IDA, NDF, EIR, DIPEx, LOSLAS, CTable2, ROI_PAC, ENVI,
    EHdr, ISCE, ARG, BAG, HDF5Image, OZI, CTG, DB2ODBC, NUMPY

    :param filename:
    :param driver_short_name:
    :return:
    """
    if driver_short_name:
        driver = gdal.GetDriverByName(driver_short_name)
        if not driver:
            raise ValueError('Could not find driver for short name {}'.format(driver_short_name))
    elif not filename:
        driver = gdal.GetDriverByName('MEM')
    else:
        try:
            driver = gdal.IdentifyDriver(filename)
        except RuntimeError:
            driver = None
        if not driver:
            drivers_dict = driver_dictionary()
            try:
                driver_short_name = drivers_dict[filename.split('.')[-1]]
                if len(driver_short_name) == 1:
                    driver = gdal.GetDriverByName(driver_short_name[0])
                else:
                    raise ValueError('Ambiguous file name {} with possible drivers: {}'.format(
                        filename, ', '.join(driver_short_name)))
            except KeyError:
                raise ValueError('Could not find driver for file {}'.format(filename))
    return driver


# =============================================================================
class RasterFilename(object):
    def __init__(self, filename):
        self.filename = filename

    def get_filename(self):
        return self.filename

    def get_member(self):
        return self.filename

    def is_compressed(self):
        return False


class RasterGzipFilename(RasterFilename):
    def __init__(self, filename):
        super(RasterGzipFilename, self).__init__(filename)

    def get_member(self):
        return self.filename[:-3]

    def is_compressed(self):
        return True


class RasterZipFilename(RasterFilename):
    def __init__(self, filename, member):
        super(RasterZipFilename, self).__init__(filename)
        if not member:
            with zipfile.ZipFile(filename) as zf:
                members = zf.namelist()
                assert len(members) == 1
                member = members[0]
        self.member = member

    def get_member(self):
        return self.member

    def is_compressed(self):
        return True


# =============================================================================
# Raster
# =============================================================================
class Raster(object):

    def __init__(self, filename, member=None):
        """

        :param filename:
        :param member:
        """
        self.dataset = None
        if filename is None:
            filename = ''

        fnl = filename.lower()
        if fnl.endswith('.zip'):
            self.filename = RasterZipFilename(filename, member)
        elif fnl.endswith('.gz'):
            self.filename = RasterGzipFilename(filename)
        else:
            self.filename = RasterFilename(filename)

    def __repr__(self):
        return self.get_filename() + ' ' + self.get_parameters().__repr__()

    def info(self, **kwargs):
        """Return information on a dataset
        ::

            return gdal.Info(self.dataset, **kwargs)

        :param kwargs: see gdal.Info
        :return: information on a dataset
        :rtype: str
        """
        return gdal.Info(self.dataset, **kwargs)

    def show(self, band_number=1, mask=True, scale=False, plot=True):
        """

        :param band_number:
        :param mask:
        :param scale:
        :return:
        """
        array = self.get_array(band_number, mask=mask, scale=scale)
        if np.ma.all(array) is np.ma.masked:
            print('Empty array: nothing to show')
        else:
            plt.imshow(array)
            if plot:
                plt.show()

    def plot(self, axes_dict=None, mask=True, scale=False, plot=True):
        """

        :param axes_dict: dictionary with ax as key and list of bands as plot
        :param mask:
        :param scale:
        :param plot:
        :return:
        """
        for ax, band_number in list(axes_dict.items()):
            if isinstance(band_number, int):
                array = self.get_array(band_number, mask=mask, scale=scale)
            else:
                array = self.get_arrays(band_number, mask=mask, scale=scale)
            ax.imshow(array)
        if plot:
            plt.show()

    def is_compressed(self):
        """Return True is the file is compressed

        Valid compressions are gzip (.gz) and zip (.zip)

        :return: True is the file is compressed
        """
        return self.filename.is_compressed()

    def get_filename(self):
        """Return the file name

        :return: file name
        :rtype: str
        """
        return self.filename.get_filename()

    def get_rastername(self):
        """Return the raster name

        Raster name may be different of file name in case of zipped files

        :return: file name
        :rtype: str
        """
        return self.filename.get_member()

    def get_parameters(self):
        """Return the raster parameters defined in the dataset

        :return: raster parameters
        :rtype: RasterParameters
        """
        return parameter.get_parameters(self.dataset)

    def get_raster_size(self):
        """Return (number of columns, number of rows)

        Return the x- and y-sizes as tuple (number of columns, number of rows)

        :return: number of columns and rows
        :rtype: (int, int)
        """
        return self.dataset.RasterXSize, self.dataset.RasterYSize

    def get_nodata(self, band_number=1):
        """Return nodata or list of nodata

        If band_number is 'all' or a list of band  numbers, return a list with one nodata for each band.
        If band_number is a number, return the nodata value

        :param band_number: 'all', band number or list of band numbers
        :type band_number: int or list of int
        :return: nodata or list of nodata
        :rtype: raster type or list of raster types
        """
        if band_number == 'all':
            band_number = list(range(1, self.get_band_count() + 1))
        try:
            return [self.dataset.GetRasterBand(i).GetNoDataValue() for i in band_number]
        except TypeError:
            return self.dataset.GetRasterBand(band_number).GetNoDataValue()

    def get_extent(self, scale=0.0):
        """Return the raster extent in world coordinates.

        :return: (x_min, x_max, y_min, y_max)
        :rtype:
        """
        xmin, xmax, ymin, ymax = extent_pixel_to_world(self.get_geotransform(),
                                                       self.dataset.RasterXSize, self.dataset.RasterYSize)
        if scale and scale != 0.0:
            dx, dy = xmax - xmin, ymax - ymin
            xmin -= dx * scale
            xmax += dx * scale
            ymin -= dy * scale
            ymax += dy * scale
        return xmin, xmax, ymin, ymax

    def copy(self, name='', driver_short_name=None):
        """Return a copy of this raster

        Creates a new RasterUpdate instance using the parameters of this raster

        :param name:
        :param driver_short_name:
        :return:
        """
        return copy(self, name, driver_short_name)

    def get_band_count(self):
        """Return the number of bands

        :return: number of bands
        :rtype: int
        """
        return self.dataset.RasterCount

    def get_band_data_type(self, band_number=1):
        """Return data type or list of data types

        If band_number is 'all' or a list of band  numbers, return a list with one data type for each band.
        If band_number is a number, return the data type 

        :param band_number: 'all', band number or list of band numbers
        :type band_number: int or list of int
        :return: data type or list of data types
        :rtype: int or list of int
        """
        if band_number == 'all':
            band_number = list(range(1, len(band_number) + 1))
        try:
            return [self.dataset.GetRasterBand(i).DataType for i in band_number]
        except TypeError:
            return self.dataset.GetRasterBand(band_number).DataType

    def get_band(self, band_number=1):
        """Return a raster band

        :param band_number: band number. Default band_number=1
        :return: raster band
        :rtype: gdal.Band
        """
        return self.dataset.GetRasterBand(band_number)

    def array_block(self, band_number=1):
        """Loop through an array and yields i_row, i_col, n_rows, n_cols, array block

        It is 5x slower than get_array

        :return: i_row, i_col, n_rows, n_cols, block
        """
        band = self.dataset.GetRasterBand(band_number)
        row_size, col_size = band.YSize, band.XSize
        block_sizes = band.GetBlockSize()
        row_block_size, col_block_size = block_sizes[1], block_sizes[0]
        col_range = list(range(0, col_size, col_block_size))
        for i_row in range(0, row_size, row_block_size):
            n_rows = row_block_size if i_row + row_block_size < row_size else row_size - i_row
            for i_col in col_range:
                n_cols = col_block_size if i_col + col_block_size < col_size else col_size - i_col
                yield i_row, i_col, n_rows, n_cols, band.ReadAsArray(i_col, i_row, n_cols, n_rows)

    def get_array(self, band_number=1, col0=0, row0=0, n_cols=None, n_rows=None, mask=False, scale=False):
        """Return the raster band array

        If band number is a number, return a 2D-array, else if band number is a list/tuple of numbers, return a 3D-array
        If mask is True, returned a numpy masked array
        If scale is True, scale the raster: (array - min)/(max - min)

        :param band_number: band number. Default band_number=1
        :param col0: starting x pixel
        :type col0: int
        :param row0: starting y pixel
        :type row0: int
        :param n_cols: number of x-pixels
        :type n_cols: int
        :param n_rows: number of y-pixels
        :type n_rows: int
        :return: 2D- or 3D-array
        :rtype: numpy.ndarray
        """
        if not n_cols:
            n_cols = self.dataset.RasterXSize
        if not n_rows:
            n_rows = self.dataset.RasterYSize
        if band_number == 'all':
            band_number = list(range(1, self.get_band_count() + 1))
        try:
            arrays = np.empty((len(band_number), n_rows, n_cols))
            for i, bn in enumerate(band_number):
                arrays[i] = self.get_array(bn, col0=col0, row0=row0, n_cols=n_cols, n_rows=n_rows,
                                           mask=mask, scale=scale)
            return arrays
        except TypeError:
            array = self.dataset.GetRasterBand(band_number).ReadAsArray(col0, row0, n_cols, n_rows)
            if mask:
                nodata = self.get_nodata(band_number=band_number)
                array = np.ma.array(array, mask=(array == nodata))
            if scale:
                array = scale_array(array)
            return array

    def get_array_full(self, value=0, **kwargs):
        """

        :param value:
        :type value: int or list
        :param kwargs:
            :key dtype: numpy dtype, default raster type converted into dtype
        :return: array
        """
        n_cols, n_rows = self.get_raster_size()
        dtype = kwargs.pop('dtype', gdal_array.GDALTypeCodeToNumericTypeCode(self.get_band_data_type()))
        if isinstance(value, (list, tuple)):
            n_bands = len(value)
            if value.count(0) == len(value):
                return np.zeros((n_bands, n_rows, n_cols), dtype)
            elif value.count(None) == len(value):
                return np.empty((n_bands, n_rows, n_cols), dtype)
            else:
                array = np.empty((n_bands, n_rows, n_cols), dtype)
                for i in range(n_bands):
                    array[i] = np.full((n_rows, n_cols), value[i], dtype)
                return array
        else:
            if value == 0:
                return np.zeros((n_rows, n_cols), dtype)
            elif value is None:
                return np.empty((n_rows, n_cols), dtype)
            else:
                return np.full((n_rows, n_cols), value, dtype)

    def get_geotransform(self):
        """Return geo transform.

        :return:
        """
        return self.dataset.GetGeoTransform()

    def transform(self, **kwargs):
        """Return a raster in the given coordinate system

        Create an instance of RasterUpdate in 'MEM':
            - If filename and driver are unset
            - If driver is 'MEM'

        :param kwargs:
            :key epsg: (str)
            :key proj4: (str)
            :key wkt: (str)
            :key output_raster: full path file name, any string if drivername='mem', or None
            :key drivername: short driver name
        :return:
        """
        output_raster = kwargs.pop('output_raster', 'MEM')
        drivername = kwargs.pop('drivername', 'Memory')
        if output_raster:
            target = output_raster
        else:
            target = drivername
        srs = kwargs.pop('wkt', kwargs.pop('proj4', None))
        if not srs:
            srs = 'epsg:{}'.format(kwargs.pop('epsg', None))
        return RasterUpdate(gdal.Warp(destNameOrDestDS=target, srcDSOrSrcDSTab=self.dataset, dstSRS=srs, **kwargs))

    def get_coordinate_system(self):
        """Return the coordinate system

        :return:
        """
        return self.dataset.GetProjection()

    def get_pixel_size(self):
        """Return pixel sizes (x, y): (column width, row height)

        :return: pixel sizes (x, y): (column width, row height)
        :rtype: tuple
        """
        return pixel_size(self.get_geotransform())

    def world_to_pixel(self, x, y):
        """Transform world to pixel coordinates

        :param x:
        :param y:
        :return: pixel coordinates of (x, y)
        :rtype: list of int
        """
        return world_to_pixel(self.get_geotransform(), x, y)

    def extent_world_to_pixel(self, min_x, max_x, min_y, max_y):
        """Return extent in pixel coordinates

        :param min_x: minimum x (minimum longitude)
        :type min_x: float
        :param max_x: maximum x (maximum longitude)
        :type max_x: float
        :param min_y: minimum x (minimum latitude)
        :type min_y: float
        :param max_y: maximum x (maximum latitude)
        :type max_y: float
        :return: (u_min, u_max, v_min, v_max)
        :rtype: tuple
        """
        return extent_world_to_pixel(self.get_geotransform(), min_x, max_x, min_y, max_y)

    def pixel_to_world(self, x, y):
        """Return the top-left world coordinate of the pixel

        :param x:
        :param y:
        :return:
        """
        return pixel_to_world(self.get_geotransform(), x, y)

    def get_pixel_centroid_coordinates(self):
        """

        :return:
        """
        dx, dy = self.get_pixel_size()
        dy = -dy
        nc, nr = self.get_raster_size()
        tr = self.get_geotransform()
        arr = np.concatenate([x.reshape(nr, nc, 1) for x in np.indices((nr, nc))][::-1], 2).astype(np.float)
        arr[:][:] *= np.array([dx, dy])
        arr[:][:] += np.array([tr[0], tr[3]])
        arr[:][:] += np.array([dx / 2.0, dy / 2.0])
        return arr

    def get_centroid_world_coordinates(self):
        """Return the raster centroid in world coordinates

        :return:
        """
        x_size, y_size = self.get_pixel_size()
        return get_centroid_world_coordinates(self.get_geotransform(),
                                              self.dataset.RasterXSize, self.dataset.RasterYSize,
                                              x_size, y_size)

    def resample(self, pixel_sizes, resample_alg=gdal.GRA_NearestNeighbour, **kwargs):
        """Resample the raster

        :param pixel_sizes:
        :param resample_alg:
        :param kwargs:
        :return:
        """
        from girs.rast.proc import resample
        return resample(self, pixel_sizes, resample_alg, **kwargs)

    def strip(self):
        from girs.rast.proc import strip
        return strip(self)


# =============================================================================
# RasterReader
# =============================================================================
class RasterReader(Raster):

    def __init__(self, filename, member=''):
        """Filename, also als .zip or .gz. In case of zip-files, a member name
        can be also be given in case there are more then one raster files in the
        zip file
        :param filename:
        :param member:
        """
        super(RasterReader, self).__init__(filename, member=member)
        fnl = filename.lower()
        if fnl.endswith('.zip'):  # /vsizip/path/to/the/file.zip/path/inside/the/zip/file
            filename = '/vsizip/' + filename + '/' + member
        elif fnl.endswith('.gz'):  # /vsigzip/path/to/the/file.gz
            filename = '/vsigzip/' + filename
        self.dataset = gdal.Open(filename, GA_ReadOnly)


class RasterEditor(Raster):

    def __init__(self, filename):
        super(RasterEditor, self).__init__(filename)

    def set_nodata(self, nodata, band_number=1):
        """Set nodata

        :param nodata:
        :param band_number:
        :return:
        """
        if isinstance(nodata, basestring):
            nodata = [nodata]
        else:
            try:
                len(nodata)
            except TypeError:
                nodata = [nodata]

        try:
            band_number_nodata = {bn: nodata[i] for i, bn in enumerate(band_number)}
        except TypeError:
            band_number = [band_number]
            band_number_nodata = {bn: nodata[i] for i, bn in enumerate(band_number)}
        for bn in band_number:
            self.dataset.GetRasterBand(bn).SetNoDataValue(band_number_nodata[bn])
        self.dataset.FlushCache()

    def set_array(self, array, band_number=1):
        """Set array

        :param array:
        :param band_number:
        :return:
        """
        result = self.dataset.GetRasterBand(band_number).WriteArray(array)
        self.dataset.FlushCache()
        return result

    def set_projection(self, srs):
        """Set projection

        :param srs:
        :return:
        """
        result = self.dataset.SetProjection(srs)
        self.dataset.FlushCache()
        return result

    def set_geotransform(self, x_min=0.0, x_pixel_size=1.0, x_rot=0.0, y_max=0.0, y_rot=0.0, y_pixel_size=1.0):
        """
        :param x_min: x location of East corner of the raster
        :param x_pixel_size: pixel width
        :param x_rot: x pixel rotation, usually zero
        :param y_max: y location of North corner of the raster
        :param y_rot: x pixel rotation, usually zero
        :param y_pixel_size: negative value of pixel height
        :return: True if setting was successful else gdal.CE_Failure
        """
        result = self.dataset.SetGeoTransform(x_min, x_pixel_size, x_rot, y_max, y_rot, y_pixel_size)
        self.dataset.FlushCache()
        return True if result == gdal.CE_None else gdal.CE_Failure


class RasterUpdate(RasterEditor):

    def __init__(self, source, drivername=None):
        """
        If source is another Raster or a raster dataset, create a copy of the dataset in 'MEM'
        If source is a filename, open the file in update modus

        :param source: raster filename, Raster, or raster dataset
        """
        try:
            super(RasterUpdate, self).__init__(source)
            # No support for zip-files
            if source.lower().endswith('.gz'):  # /vsigzip/path/to/the/file.gz
                source = '/vsigzip/' + source
            if not drivername or drivername == 'MEM':
                drv = gdal.GetDriverByName('MEM')
                drv
            else:
                raise ValueError('No filename defined')

            self.dataset = gdal.Open(source, GA_Update)
        except AttributeError:
            super(RasterUpdate, self).__init__('')
            try:
                self.dataset = gdal.GetDriverByName('MEM').CreateCopy('', source.dataset)
            except AttributeError:
                self.dataset = gdal.GetDriverByName('MEM').CreateCopy('', source)
            self.filename = ''


class RasterWriter(RasterEditor):

    def __init__(self, raster_parameters, source=None, drivername=None):
        """Create an instance of RasterWriter in 'MEM':
            - If source and driver are not given
            - If drivername is 'MEM'

        :param raster_parameters:
        :param source:
        :param drivername: gdal driver short name or an instance of gdal.Driver
        """
        super(RasterWriter, self).__init__(source)
        drv = None
        if not source:
            if not drivername or drivername == 'MEM':
                drv = gdal.GetDriverByName('MEM')
            else:
                raise ValueError('No filename defined')
        elif source:
            if not drivername:
                drv = get_driver(source)
            else:
                try:
                    drv = gdal.GetDriverByName(drivername)
                except TypeError as e:
                    if not isinstance(drivername, gdal.Driver):
                        raise e
                    drv = drivername

        n_bands = raster_parameters.number_of_bands
        try:
            dt = raster_parameters.data_types[0]
        except TypeError:
            dt = raster_parameters.data_types
        try:
            filename = self.get_filename()
            self.dataset = drv.Create(filename, raster_parameters.RasterXSize, raster_parameters.RasterYSize,
                                      n_bands, dt)
        except RuntimeError as e:
            msg = '{} or raster {} is being eventually used (locked)'.format(e.message, self.filename)
            raise RuntimeError(msg)
        self.dataset.SetGeoTransform(raster_parameters.geo_trans)
        self.dataset.SetProjection(raster_parameters.srs)
        raster_parameters.set_nodata(raster_parameters.nodata)
        for i in range(n_bands):
            if raster_parameters.nodata[i] is not None:
                rb_out = self.dataset.GetRasterBand(i+1)
                rb_out.SetNoDataValue(raster_parameters.nodata[i])
                rb_out.FlushCache()


# =============================================================================
# Functions
# =============================================================================


def info(raster, **kwargs):
    """Return raster information



    :param raster:
    :param kwargs:
    :return:
    """
    try:
        raster = RasterReader(raster)
        dataset = raster.dataset
    except AttributeError:
        try:
            dataset = raster.dataset
        except AttributeError:
            dataset = raster
    return gdal.Info(dataset, **kwargs)


def create_gifs(output_filename, *args, **kwargs):
    """

    :param self:
    :param output_filename:
    :param args:
    :param kwargs:
        :key mask: default True
        :key scale: default False
        :key band_number:
        :key nodata:
        :key cmap_name: default Blues
        :key cmap: default Blues
        :key resize: default 1
    :return:
    """
    from PIL import Image
    import matplotlib as mpl

    resize = kwargs.pop('resize', 1)
    cm = kwargs.pop('cmap', mpl.cm.get_cmap(kwargs.pop('cmap_name', 'plasma_r')))
    images = list()
    a_max, a_min = None, None
    for i, arg in enumerate(args):
        try:
            r = RasterReader(arg)
        except AttributeError:
            r = arg
        array = r.get_array(mask=True, scale=False)
        a_min = array.min() if a_min is None else min(a_min, array.min())
        a_max = array.max() if a_max is None else max(a_max, array.max())
    for i, arg in enumerate(args):
        try:
            r = RasterReader(arg)
        except AttributeError:
            r = arg
        array = r.get_array(mask=True, scale=False)
        array = old_div((array - a_min), (a_max - a_min))
        array = cm(array)
        img = Image.fromarray((array * 255).astype('uint8'))
        img = img.resize((img.size[0] * resize, img.size[1] * resize))
        images.append(img)
    images[0].save(output_filename, 'GIF', duration=1000, save_all=True, optimize=False, append_images=images[1:])


# def create_gifs(output_filename, *args, **kwargs):
#     """
#
#     :param self:
#     :param output_filename:
#     :param args:
#     :param kwargs:
#         :key mask: default True
#         :key scale: default False
#         :key band_number:
#     :return:
#     """
#     from PIL import Image
#     import matplotlib as mpl
#     cm_hot = mpl.cm.get_cmap('hot')
#     images = list()
#     for i, arg in enumerate(args):
#         try:
#             r = RasterReader(arg)
#         except AttributeError:
#             r = arg
#
#         p = r.get_parameters()
#         nodata = p.nodata
#         array = r.get_array(mask=False, scale=False)
#         array[array == nodata] = 0
#         array *= 255.0 / array.max()
#         array = array.astype(np.uint8)
#         array = cm_hot(array)
#         array *= 255
#         array = array.astype('uint8')
#         print array
#         # data = img.getdata()
#         # max_d = max(data) * 1.2
#         # img.putdata([item if item != nodata else 0 for item in data])
#         img = Image.fromarray(array)
#         img = img.resize((img.size[0] * 50, img.size[1] * 50))
#         images.append(img)
#     images[0].save(output_filename, 'GIF', duration=2000, save_all=True, optimize=False, append_images=images[1:])


# def create_gifs(output_filename, *args, **kwargs):
#     """
#
#     :param self:
#     :param output_filename:
#     :param args:
#     :param kwargs:
#         :key mask: default True
#         :key scale: default False
#         :key band_number:
#     :return:
#     """
#     from PIL import Image
#     import imageio as io
#     images = list()
#
#     for i, arg in enumerate(args):
#         try:
#             r = RasterReader(arg)
#         except AttributeError:
#             r = arg
#         array = r.get_array(mask=False, scale=False)
#         images = [io.imread(os.path.join(input_dir, f1)) for f1 in filenames]
#     io.mimsave(output_gif, jpeg_images, duration=0.5)


def get_parameters(raster):
    """Return the raster parameters defined in this raster

    :param raster: dataset or filename
    :type raster: gdal.Dataset
    :return: raster parameters
    :rtype: RasterParameters
    """

    return parameter.get_parameters(raster)


def copy(raster, dst_filename='', driver_short_name=None):
    """Return a copy of given raster

    Creates a new RasterUpdate instance using the parameters of this raster

    :param raster:
    :param dst_filename:
    :param driver_short_name:
    :return: copy of this raster
    :rtype: RasterUpdate
    """
    try:
        raster = RasterReader(raster)
    except AttributeError:
        raster = raster
    drv = get_driver(dst_filename, driver_short_name)
    dataset = drv.CreateCopy(dst_filename, raster.dataset)
    return RasterUpdate(dataset)


def scale_array(array):

    def scale(a):
        array_min = np.amin(a)
        array_max = np.amax(a)
        return old_div((a - array_min), (array_max - array_min))

    if len(array.shape) > 2:  # ndim does not work for masked arrays
        for i in range(len(array)):
            array[i, :, :] = scale(array[i, :, :])
    else:
        array = scale(array)
    return array


def pixel_size(geo_trans):
    """Return pixel sizes

    :param geo_trans: geo transformation
    :type geo_trans: tuple with six values
    :return: pixel sizes (x, y): (column width, row height)
    :rtype: tuple
    """
    return geo_trans[1], -geo_trans[5]


def world_to_pixel(geo_trans, x, y, np_func=np.trunc):
    """Transform world into pixel coordinates

    :param geo_trans: geo transformation
    :type geo_trans: tuple with six values
    :param x:
    :param y:
    :param np_func:
    :return: pixel coordinates of (x, y)
    :rtype: list of int
    """
    # print geo_trans, x, y
    # xOffset = int((x - geo_trans[0]) / geo_trans[1])
    # yOffset = int((y - geo_trans[3]) / geo_trans[5])
    # print xOffset, yOffset,
    xOffset = np_func(np.divide(x - geo_trans[0], geo_trans[1])).astype(np.int)
    yOffset = np_func(np.divide(y - geo_trans[3], geo_trans[5])).astype(np.int)
    # print xOffset, yOffset
    return xOffset, yOffset


def pixel_to_world(geo_trans, x, y):
    """Return the top-left world coordinate of the pixel

    :param geo_trans: geo transformation
    :type geo_trans: tuple with six values
    :param x:
    :param y:
    :return:
    """
    return geo_trans[0] + (x * geo_trans[1]), geo_trans[3] + (y * geo_trans[5])


def extent_pixel_to_world(geo_trans, raster_x_size, raster_y_size):
    """Return extent in world coordinates.

    Transform the given pixel coordinates `raster_x_size` (number of columns) and `raster_y_size` (number of rows) into
    world coordinates.

    :param geo_trans: geo transformation
    :type geo_trans: tuple with six values
    :param raster_x_size: number of columns
    :type raster_x_size: int
    :param raster_y_size: number of rows
    :return: (x_min, x_max, y_min, y_max)
    :rtype: tuple
    """
    x_min0, y_max0 = pixel_to_world(geo_trans, 0, 0)
    x_max0, y_min0 = pixel_to_world(geo_trans, raster_x_size, raster_y_size)
    return x_min0, x_max0, y_min0, y_max0


def extent_world_to_pixel(geo_trans, min_x, max_x, min_y, max_y):
    """Return extent in pixel coordinates

    :param geo_trans: geo transformation
    :type geo_trans: tuple with six values
    :param min_x: minimum x (minimum longitude)
    :type min_x: float
    :param max_x: maximum x (maximum longitude)
    :type max_x: float
    :param min_y: minimum x (minimum latitude)
    :type min_y: float
    :param max_y: maximum x (maximum latitude)
    :type max_y: float
    :return: (u_min, u_max, v_min, v_max)
    :rtype: tuple
    """
    geo_trans = list(geo_trans)
    u_min, v_min = world_to_pixel(geo_trans, min_x, max_y)
    u_max, v_max = world_to_pixel(geo_trans, max_x, min_y)
    geo_trans[0], geo_trans[3] = pixel_to_world(geo_trans, u_min, v_min)
    return (u_min, u_max, v_min, v_max),  geo_trans


def get_centroid_world_coordinates(geo_trans, raster_x_size, raster_y_size, x_pixel_size, y_pixel_size):
    """Return the raster centroid in world coordinates

    :param geo_trans: geo transformation
    :type geo_trans: tuple with six values
    :param raster_x_size: number of columns
    :type raster_x_size: int
    :param raster_y_size: number of rows
    :param x_pixel_size: pixel size in x direction
    :type: x_pixel_size: float
    :param y_pixel_size: pixel size in y direction
    :type y_pixel_size: float
    :return:
    """
    x0, y0 = pixel_to_world(geo_trans, 0, 0)
    x1, y1 = pixel_to_world(geo_trans, raster_x_size-1, raster_y_size-1)
    x1 += x_pixel_size
    y1 -= y_pixel_size
    return (x0 + x1) * 0.5, (y0 + y1) * 0.5


def get_default_values(number_of_bands, values):
    """Return values for bands

    Utility function to get values (e.g., nodata) for bands.

    For n = number_of_bands:
    - If values is a single value, transform it into a list with n elements
    - If values is a list with size lower than n, extend the list to size n by repeating the last value (n=4, values=[1, 2], result=[1, 2, 2, 2]
    - If values is a list with size greater than n, slice values to values[:n]

    :param number_of_bands: number of bands
    :type number_of_bands: int
    :param values: value or list of values
    :type values: same as raster type
    :return: values
    :rtype: same as input values
    """
    try:
        if number_of_bands < len(values):
            values = values[:number_of_bands]
        elif number_of_bands > len(values):
            values = values[-1] * (number_of_bands - len(values))
    except TypeError:
        values = [values] * number_of_bands
    except:
        raise
    return values


def rasters_get_extent(rasters, extent_type='intersection'):
    """Return the extent of a list of rasters.

    Return the extent of the union or intersection of a list of rasters

    :param rasters: list of rasters or raster filenames (also mixed)
    :param extent_type: intersection or union
    :return: (xmin, xmax, ymin, ymax) in world coordinates
    """
    # Get get common extent
    # Get the rasters
    rasters = [RasterReader(ir) if isinstance(ir, basestring) else ir for ir in rasters]
    return merge_envelopes([r.get_extent() for r in rasters])


def rasters_get_pixel_size(rasters, minmax='max'):
    """Return union or intersection of pixel sizes
        - If minmax='min', return the intersection of the pixel sizes defined in the list of rasters. This corresponds to the smallest pixel size among all rasters.
        - If minmax='max', return the union of the pixel sizes defined in the list of rasters. This corresponds to the largest pixel size among all rasters.

    :param rasters: list of rasters
    :type rasters: list of raster file names or list of Raster instances, also both types in the same list
    :param minmax: 'min' for intersection and 'max' for union
    :type minmax: str
    :return: pixel sizes (x, y): (number of columns, number of rows)
    :rtype: tuple
    """
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

