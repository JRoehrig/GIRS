from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
import numpy as np
from osgeo import gdal, gdal_array
from girs.rast.parameter import RasterParameters
from girs.rast.raster import get_driver, RasterReader, RasterWriter
from girs.geom.envelope import merge_envelopes, buffer_envelope


def get_parameters_from_raster(r):
    try:
        r = RasterReader(r)
    except AttributeError:
        pass
    try:
        return r.get_parameters()
    except AttributeError:
        return None


def mosaic(*args, **kwargs):
    """

    :param args:
    :param kwargs:
    :return:
    """
    # Handle the case of *args being a tuple of list
    rp0 = get_parameters_from_raster(args[0])
    if not rp0:
        args = args[0]
    rp0 = get_parameters_from_raster(args[0])
    gt0 = rp0.geo_trans

    raster_parameters = [rp0]
    for i in range(1, len(args)):
        raster_parameters.append(get_parameters_from_raster(args[i]))
    envs = [rp0.get_extent_world() for rp0 in raster_parameters]
    env = merge_envelopes(envs, extent_type='union')
    env = buffer_envelope(env, (-gt0[1]*0.5, gt0[5]*0.5))
    (u_min, u_max, v_min, v_max), geo_trans = rp0.extent_world_to_pixel(*env)
    u_min, u_max = 0, u_max - u_min
    v_min, v_max = 0, v_max - v_min
    rp_out = RasterParameters(u_max+1, v_max+1, geo_trans, rp0.srs, rp0.number_of_bands, rp0.nodata, rp0.data_types,
                              driver_short_name=rp0.driverShortName)
    r_out = RasterWriter(rp_out, kwargs.pop('output_raster', None))
    for i in range(rp_out.number_of_bands):
        bn = i + 1
        dtype = gdal_array.GDALTypeCodeToNumericTypeCode(rp_out.data_types[i])
        array_out = np.full((rp_out.RasterYSize, rp_out.RasterXSize), rp_out.nodata[i], dtype)
        for arg in args:
            try:
                r = RasterReader(arg)
            except AttributeError:
                r = arg
            p = r.get_parameters()
            x_min, y_max = p.pixel_to_world(0, 0)
            x_min, y_max = x_min + (p.geo_trans[1] * 0.5), (y_max + p.geo_trans[5] * 0.5)
            u_min, v_min = rp_out.world_to_pixel(x_min, y_max)
            array = r.get_array(band_number=bn)
            array_out[v_min:v_min+p.RasterYSize, u_min:u_min+p.RasterXSize] = array
        r_out.set_array(array=array_out, band_number=bn)
    return r_out


def composite(*args, **kwargs):
    """

    :param args: Raster or filenames
    :param kwargs:
        :key output_raster: filename of the output raster
    :return:
    """
    p = get_parameters_from_raster(args[0])
    if not p:
        args = args[0]
    p = get_parameters_from_raster(args[0])
    p.number_of_bands = len(args)
    r_out = RasterWriter(p, kwargs.pop('output_raster', None))
    for i, arg in enumerate(args):
        try:
            r = RasterReader(arg)
        except AttributeError:
            r = arg
        r_out.set_array(r.get_array(), i+1)
    r_out.dataset.FlushCache()
    return r_out


def calculate(calc, **kwargs):
    """Examples::

        # Returns in 'MEM', applying the algebra to raster R, bands 1 and 2
        calculate("(R1-R2)/(R1+R2)", R='C:/tmp/raster1.tif')
        # Save to file
        calculate("(R1-R2)/(R1+R2)", R='C:/tmp/raster1.tif', output_raster='C:/tmp/ndvi.tif')
        # Using lists of bands. Note that [1:2] means band 1 and band 2. band numbers start at
        # 1 and include the last index
        calculate("(R[1, 2]+S[1, 2])/(R[1, 2]-S[1, 2])")
        # ... which is equivalent to
        calculate("(R[:2]+S[:2])/(R[:2]-S[:2])")
        # Applying calculate to all bands
        calculate("(R[:]+S[:])/(R[:]-S[:])")
        # This is also possible:
        calculate("(R[1, 2]+S[3, 4])/(R[:2]-S[1, 4])")

    See also:

        - https://svn.osgeo.org/gdal/trunk/gdal/swig/python/scripts/gdal_calc.py
        - https://github.com/rveciana/geoexamples/blob/master/python/gdal-performance/classification_blocks.py

    :param calc: algebraic operation supported by numpy arrays "(R1-R2)/(R1+R2)", where R1 and R2 are the keys used in the argument rasters:

        - R1: raster R band 1
        - R2: raster R band 2

    :param kwargs:

        - `<letter>` (a letter used in calc, string):  filename or Raster instance
        - `output_raster` (str): filename of the new rasters
        - `driver` (str): driver name

    :return:
    """
    output_raster = kwargs.pop('output_raster', '')
    driver = kwargs.pop('driver', None)
    driver = get_driver(output_raster, driver)
    driver = driver.ShortName

    r = RasterReader(list(kwargs.values())[0])

    # Separate list of bands in single calculations
    calc = _parse_range_to_list(calc, r.get_band_count())
    calculations = _split_calculation(calc)

    raster_parameters = r.get_parameters()
    raster_parameters.driverShortName = driver
    raster_parameters.number_of_bands = len(calculations)

    r_out = RasterWriter(raster_parameters, output_raster)

    for i, calc in enumerate(calculations):
        calc_dict = _get_symbol_and_masked_arrays(calc, kwargs)
        arr = eval(calc, calc_dict)
        r_out.set_array(arr.data, i+1)
    r_out.dataset.FlushCache()
    return r_out


def resample(input_raster, pixel_sizes, resample_alg=gdal.GRA_NearestNeighbour, **kwargs):
    """
    Args:
        input_raster: file name of a rasters or a Raster object

        pixel_sizes (list, str or Raster): list of pixel_width, pixel_height, file name of a rasters, or Raster object

        kwargs:
            'output_raster' (string): filename of the new rasters.
            'driver' (string): driver name (e.g., 'MEM')

    """
    try:
        pixel_width, pixel_height = float(pixel_sizes), float(pixel_sizes)
    except:
        try:
            pixel_width, pixel_height = pixel_sizes
            pixel_width, pixel_height = float(pixel_width), float(pixel_height)
        except:
            try:
                raster0 = RasterReader(pixel_sizes)
            except:
                raster0 = pixel_sizes
            pixel_width, pixel_height = raster0.get_pixel_size()
            del raster0

    try:
        input_raster = RasterReader(input_raster)
    except:
        pass

    x_min, x_max, y_min, y_max = input_raster.get_extent()

    driver = get_driver(kwargs['output_raster'] if 'output_raster' in kwargs else None,
                        kwargs['driver'] if 'driver' in kwargs else None)
    driver = driver.ShortName

    len_x = x_max-x_min
    len_y = y_max-y_min
    u_max, v_max = int(old_div(len_x,pixel_width)), int(old_div(len_y,pixel_height))
    if (old_div((len_x - u_max * pixel_width), pixel_width)) > 0.5:
        u_max += 1
    if (old_div((len_y - v_max * pixel_height), pixel_height)) > 0.5:
        v_max += 1

    gt_out = list(input_raster.get_geotransform())  # make it mutable
    gt_out[1], gt_out[5] = pixel_width, -pixel_height

    name = kwargs.get('output_raster', None)
    driver = kwargs.get('driver', None)
    raster_parameters = input_raster.get_parameters()
    raster_parameters.RasterXSize = u_max
    raster_parameters.RasterYSize = v_max
    raster_parameters.geo_trans = gt_out
    raster_parameters.driverShortName = driver
    raster_out = RasterWriter(raster_parameters, name)

    gdal.ReprojectImage(input_raster.dataset, raster_out.dataset, input_raster.get_coordinate_system(), raster_out.get_coordinate_system(), resample_alg)

    return raster_out


def strip(input_raster):
    """Remove top and bottom rows, and left and right columns containing only nodata

    :param input_raster:
    :return:
    """
    import numpy as np
    raster_parameters = input_raster.get_parameters()
    u_min = v_min = 0
    u_max, v_max = input_raster.get_raster_size()
    u_min_min, v_min_min = u_max, v_max
    u_max_max, v_max_max = u_min, v_min
    for bn in range(1, input_raster.get_band_count() + 1):
        array = input_raster.get_array()
        nodata = input_raster.get_nodata(bn)
        if nodata is np.nan or nodata is None:
            for u_min in range(u_min, u_max):
                if not np.all(np.isnan(array[:, u_min])):
                    break
            for u_max in range(u_max-1, u_min, -1):
                if not np.all(np.isnan(array[:, u_max])):
                    break
            for v_min in range(v_min, v_max):
                if not np.all(np.isnan(array[v_min, :])):
                    break
            for v_max in range(v_max-1, v_min, -1):
                if not np.all(np.isnan(array[v_max:, :])):
                    break
        else:
            for u_min in range(u_min, u_max):
                if not np.all((array[:, u_min] == nodata)):
                    break
            for u_max in range(u_max-1, u_min, -1):
                if not np.all((array[:, u_max] == nodata)):
                    break
            for v_min in range(v_min, v_max):
                if not np.all((array[v_min, :] == nodata)):
                    break
            for v_max in range(v_max-1, v_min, -1):
                if not np.all((array[v_max:, :] == nodata)):
                    break
        u_min_min = min(u_min_min, u_min)
        u_max_max = max(u_max_max, u_max)
        v_min_min = min(v_min_min, v_min)
        v_max_max = max(v_max_max, v_max)
    x_min, y_max = raster_parameters.pixel_to_world(u_min_min, v_min_min)
    raster_parameters.geo_trans = list(raster_parameters.geo_trans)
    raster_parameters.geo_trans[0], raster_parameters.geo_trans[3] = x_min, y_max
    raster_parameters.RasterXSize = u_max_max - u_min_min + 1
    raster_parameters.RasterYSize = v_max_max - v_min_min + 1
    output_raster = RasterWriter(raster_parameters, input_raster.get_rastername())
    for bn in range(1, input_raster.get_band_count() + 1):
        array = input_raster.get_array(band_number=bn, col0=u_min_min, row0=v_min_min,
                                       n_cols=raster_parameters.RasterXSize,
                                       n_rows=raster_parameters.RasterYSize)
        r_band = output_raster.get_band(bn)
        r_band.WriteArray(array)
        output_raster.dataset.FlushCache()
    return output_raster


# =============================================================================
# Utilities
# =============================================================================
def _get_symbol_and_masked_arrays(calc, symbol_dict):
    symbol_keys = sorted(symbol_dict.keys())

    def _starts_with(s):
        for k in symbol_keys:
            if s.startswith(k):
                return True
        return False

    def _get_symbol(calc0):
        key0 = ''
        band0 = ''
        if not _starts_with(calc0[0]):
            return key0, band0, calc0[1:] if len(calc0) > 1 else ''
        while calc0 and _starts_with(calc0[0]):
            key0, calc0 = key0 + calc0[0], calc0[1:] if len(calc0) > 1 else ''
        if key0:
            if calc0 and calc0[0].isdigit():
                band0 = ''
                while calc0 and calc0[0].isdigit():
                    band0, calc0 = band0 + calc0[0], calc0[1:] if len(calc0) > 1 else ''
            else:
                band0 = ''
        return key0, band0, calc0

    result_dict = dict()
    while calc:
        key, band, calc = _get_symbol(calc)
        symbol = key + band
        if symbol and symbol not in result_dict:
            r = RasterReader(symbol_dict[key])
            result_dict[symbol] = r.get_array(int(band) if band != '' else 1, mask=True)
    return result_dict


def _parse_range_to_list(calc, number_of_bands):
    idx0 = calc.find('[')
    while idx0 > -1:
        idx1 = calc.find(']', idx0)
        r = calc[idx0:idx1].split(':')
        try:
            i0 = int(r[0])
        except ValueError:
            i0 = 1
        try:
            i1 = int(r[1])
        except ValueError:
            i1 = number_of_bands
        r = ', '.join([str(i) for i in range(i0, i1+1)])
        calc = calc[:idx0+1] + r + calc[idx1:]
        idx1 = idx0 + len(r)
        idx0 = calc.find('[', idx1)
    return calc


def _split_calculation(calc):

    def _list_to_calculations(calc0, ii):
        i0 = calc0.find('[')
        while i0 > -1:
            i1 = calc0.find(']', i0)
            r = calc0[i0 + 1:i1].split(',')[ii].strip()
            calc0 = calc0[:i0] + r + calc0[i1+1:]
            i1 = i0 + len(r) - 1
            i0 = calc0.find('[', i1)
        return calc0

    idx0 = calc.find('[')
    if idx0 > -1:
        idx1 = calc.find(']', idx0)
        n = len(calc[idx0 + 1:idx1].split(','))
        l2t = [_list_to_calculations(calc, i) for i in range(n)]
    else:
        l2t = [calc]
    return l2t


