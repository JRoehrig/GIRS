from __future__ import print_function
from builtins import str
from builtins import range
import os
import sys
import numpy as np
from girs.feat.layers import LayersReader
from girs.rast.raster import RasterReader, RasterWriter, get_driver
from girs.rast.raster import get_parameters
from girs.rastfeat import rasterize


def clip_rasters(rasters, output_dir=None, suffix=''):
    """Clip a list of rasters

    :param rasters:
    :param output_dir:
    :param suffix:
    :return:
    """
    if not output_dir:
        output_dir = os.path.dirname(rasters[0])
    x_min = -sys.float_info.max
    x_max = sys.float_info.max
    y_min = -sys.float_info.max
    y_max = sys.float_info.max
    w_list = []
    h_list = []
    for r in rasters:
        parameters = get_parameters(r)
        w, h = parameters.pixel_size()
        w_list.append(w)
        h_list.append(h)
        x_min0, x_max0, y_min0, y_max0 = parameters.get_extent()
        x_min = max(x_min0, x_min)
        x_max = min(x_max0, x_max)
        y_min = max(y_min0, y_min)
        y_max = min(y_max0, y_max)
    assert w_list[1:] == w_list[:-1]
    assert h_list[1:] == h_list[:-1]
    x_min += w_list[0] / 2.0
    x_max -= w_list[0] / 2.0
    y_min += h_list[0] / 2.0
    y_max -= h_list[0] / 2.0
    output_filenames = []
    for r in rasters:
        f_out = os.path.basename(r).split('.')[0] + suffix + '.tif'
        f_out = os.path.join(output_dir, f_out)
        clip_by_extent(r, extent=[x_min, x_max, y_min, y_max], output_raster=f_out)
        output_filenames.append(f_out)
    return output_filenames


def clip_by_extent(raster_in, **kwargs):
    """Clip the raster by extent

    If `output_raster` is not set in kwargs, return the new rasters dataset as `MEM` driver, otherwise None

    :param raster_in:
    :param kwargs:
        :key extent: (list): [x_min, x_max, y_min, y_max] is the extent in world coordinates
        :key layers: (string / Layers): layers file name or LayersSet object used to get the extent
        :key rasters: (string / Layers): rasters file name or Raster object used to get the extent
        :key layer_number: (int): layer number, only used if `layers` is defined. Default is the first layer (0)
        :key scale: (float): scale the extent: scale > 1 creates a buffer
        :key driver: (string): driver short name
        :key output_raster: (string): filename of the rasters.
    :return:
    """
    try:
        raster_in = RasterReader(raster_in)
    except:
        pass

    try:
        raster_parameters = raster_in.get_parameters()
    except Exception:
        try:
            raster_parameters = get_parameters(raster_in)
        except Exception:
            raster_parameters = raster_in

    x_min0, x_max0, y_min0, y_max0 = raster_parameters.get_extent_world()

    if 'extent' in kwargs:
        x_min, x_max, y_min, y_max = kwargs['extent']
    elif 'layers' in kwargs:
        try:
            layers = LayersReader(kwargs['layers'])
        except (TypeError, RuntimeError):
            layers = kwargs['layers']
        layer_number = kwargs['layer_number'] if 'layer_number' in kwargs else 0
        layers = layers.transform(wkt=raster_in.get_coordinate_system())
        x_min, x_max, y_min, y_max = layers.get_extent(layer_number=layer_number)
    elif 'rasters' in kwargs:
        try:
            raster0 = RasterReader(kwargs['rasters'])
        except Exception as e:
            raster0 = kwargs['rasters']
        x_min, x_max, y_min, y_max = raster0.get_extent()
    else:
        x_min, x_max, y_min, y_max = x_min0, x_max0, y_min0, y_max0

    if 'scale' in kwargs:
        scale = kwargs['scale'] - 1.0
        d = max((x_max - x_min) * scale, (y_max - y_min) * scale)
        x_min -= d
        x_max += d
        y_min -= d
        y_max += d

    x_min = max(x_min, x_min0)
    x_max = min(x_max, x_max0)
    y_min = max(y_min, y_min0)
    y_max = min(y_max, y_max0)
    (u_min, u_max, v_min, v_max), geo_trans = raster_parameters.extent_world_to_pixel(x_min, x_max, y_min, y_max)
    u_min = max(u_min, 0)
    u_max = min(u_max, raster_parameters.RasterXSize - 1)
    v_min = max(v_min, 0)
    v_max = min(v_max, raster_parameters.RasterYSize - 1)
    raster_parameters.RasterXSize = u_max - u_min + 1
    raster_parameters.RasterYSize = v_max - v_min + 1
    raster_parameters.geo_trans = geo_trans

    if raster_parameters.RasterXSize < 0 or raster_parameters.RasterYSize < 0:
        msg = 'clip_by_extent: negative number of cells ({}, {}){}'
        if 'extent' in kwargs:
            msg1 = '\n\tThe parameter extent may be in a different coordinate system from raster:' + \
                   '\n\t\traster: {}\n\t\textent: {}'
            msg1 = msg1.format(str(raster_in.get_extent()), str(kwargs['extent']))
        else:
            msg1 = ''
        msg = msg.format(raster_parameters.RasterXSize,raster_parameters.RasterYSize, msg1)
        raise ValueError(msg)

    try:
        name = kwargs['output_raster']
        r_dir = os.path.dirname(name)
        if not os.path.isdir(r_dir):
            os.makedirs(r_dir)
    except Exception:
        name = ''

    try:
        driver = get_driver(driver_short_name=kwargs['driver'])
    except KeyError:
        driver = get_driver(name)

    raster_parameters.driverShortName = driver.ShortName

    raster_out = RasterWriter(raster_parameters, name)

    array = raster_in.get_array(band_number='all', col0=u_min, row0=v_min,
                                n_cols=raster_parameters.RasterXSize, n_rows=raster_parameters.RasterYSize)
    for i in range(raster_parameters.number_of_bands):
        r_band = raster_out.get_band(i+1)
        r_band.WriteArray(array[i])
    raster_out.dataset.FlushCache()
    return raster_out


def clip_by_vector(raster_in, layers_in, **kwargs):
        # layer_number=0, driver=None, burn_value=1, output_raster=None, nodata=None, all_touched=False):
    """Clip by vector

    :param raster_in:
    :type raster_in:
    :param layers_in: layers file name or Layers object. The geometry must be line or polygon
    :type layers_in:
    :param kwargs:
        :key layer_number: (int) layer number. Default is the first layer (0)
        :key scale: (float) scale the features' bounding box
        :key driver: driver name
        :key output_raster: filename of the rasters.
        :key nodata: nodata
        :key all_touched:
    :return: RasterWriter instance
    :rtype: RasterWriter
    """
    output_raster = kwargs.pop('output_raster', None)
    driver = get_driver(filename=output_raster, driver_short_name=kwargs.pop('driver', None))
    layer_number = kwargs.pop('layer_number', 0)
    nodata = kwargs.pop('nodata', None)

    all_touched = kwargs.pop('all_touched', False)

    try:
        layers_in = LayersReader(layers_in)
        layers_in.get_layer(layer_number)
    except (TypeError, RuntimeError):
        try:
            layers_in.get_layer(layer_number)
        except AttributeError:
            pass

    try:
        raster_in = RasterReader(raster_in)
    except AttributeError:
        pass

    srs_raster = raster_in.get_coordinate_system()
    if not srs_raster:
        print ('WARNING: raster {} has no coordinate system'.format(raster_in.get_rastername()) )
    else:
        layers_in = layers_in.transform(wkt=srs_raster)

    r_out = clip_by_extent(raster_in, extent=layers_in.get_extent(layer_number), driver='MEM')

    raster_parameters = r_out.get_parameters()
    raster_parameters.set_driver_short_name(driver.ShortName)
    array_out = r_out.get_array(band_number='all')
    del r_out

    if not nodata:
        nodata = raster_parameters.nodata

    burn_values = np.int32(1)  # TODO
    r_burn = rasterize.rasterize_layers(layers_in, burn_values=[burn_values], raster_parameters=raster_parameters,
                                        layer_number=layer_number, all_touched=all_touched)

    for i in range(0, raster_parameters.number_of_bands):
        array_burn = r_burn.get_array(i+1)
        array_mask = np.ma.masked_where(array_burn != burn_values, array_burn)
        b_nodata = nodata[i]
        array_out[i] = np.where(array_mask.mask, b_nodata, array_out[i])
    del r_burn

    raster_out = RasterWriter(raster_parameters, output_raster) #nu, nv, nb, data_types[0], geo_trans, srs, driver=driver, nodata=nodata)
    for i in range(raster_parameters.number_of_bands):
        raster_out.set_array(array_out[i], i+1)
    raster_out.dataset.FlushCache()
    return raster_out

