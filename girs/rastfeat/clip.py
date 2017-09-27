import os
import sys

import numpy as np
from osgeo import gdal, ogr, gdalnumeric

from girs.feat.layers import LayersReader
from girs.rast.raster import RasterReader, RasterWriter, get_driver
from girs.rast.raster import geometry_to_gdal_image, get_parameters
from girs.rastfeat import rasterize


def clip_rasters(rasters, output_dir=None, suffix=''):
    if not output_dir:
        output_dir = os.path.dirname(rasters[0])
    x_min = -sys.float_info.max
    x_max = sys.float_info.max
    y_min = -sys.float_info.max
    y_max = sys.float_info.max
    w_list = []
    h_list = []
    for f in rasters:
        parameters = get_parameters(f)
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
    for f in rasters:
        f_out = os.path.basename(f).split('.')[0] + suffix + '.tif'
        f_out = os.path.join(output_dir, f_out)
        clip_by_extent(f, extent=[x_min, x_max, y_min, y_max], output_raster=f_out)
        output_filenames.append(f_out)
    return output_filenames


def clip_by_extent(raster_in, **kwargs):
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
        except TypeError, e:
            layers = kwargs['layers']
        layer_number = kwargs['layer_number'] if 'layer_number' in kwargs else 0
        x_min, x_max, y_min, y_max = layers.get_extent(layer_number=layer_number)
    elif 'rasters' in kwargs:
        try:
            raster0 = RasterReader(kwargs['rasters'])
        except Exception, e:
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
    raster_parameters.RasterXSize = u_max-u_min
    raster_parameters.RasterYSize = v_max-v_min
    raster_parameters.geo_trans = geo_trans

    try:
        name = kwargs['output_raster']
        r_dir = os.path.dirname(name)
        if not os.path.isdir(r_dir):
            os.makedirs(r_dir)
    except Exception:
        name = ''

    try:
        driver = get_driver(kwargs['driver'])
    except KeyError:
        driver = get_driver(name)

    raster_parameters.driverShortName = driver.ShortName

    raster_out = RasterWriter(name, raster_parameters)

    array = raster_in.clip_arrays(u_min, v_min, u_max - u_min, v_max - v_min)
    array_shape_len = len(array.shape)
    assert raster_parameters.number_of_bands == 1 if array_shape_len == 2 else raster_parameters.number_of_bands > 1
    for i in range(raster_parameters.number_of_bands):
        r_band = raster_out.get_band(i+1)
        r_band.WriteArray(array)
    raster_out.dataset.FlushCache()
    return raster_out


def clip_by_vector(raster_in, layers_in, layer_number=0, scale=1.0, driver=None, burn_value=1, output_raster='mem',
                   nodata=None, all_touched=False):
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
    if not driver:
        driver = get_driver(output_raster)

    try:
        layers_in = LayersReader(layers_in)
        _ = layers_in.get_layer(layer_number)
    except (TypeError, RuntimeError):
        try:
            _ = layers_in.get_layer(layer_number)
        except AttributeError:
            pass

    try:
        raster_in = RasterReader(raster_in)
    except:
        pass

    r_out = clip_by_extent(raster_in, extent=layers_in.get_extent(layer_number), driver='MEM')

    raster_parameters = r_out.get_parameters()
    drv = get_driver(output_raster)
    raster_parameters.set_driver_short_name(drv.ShortName)
    array_out = r_out.get_arrays()
    del r_out

    if not nodata:
        nodata = raster_parameters.nodata

    r_burn = rasterize.rasterize_layer(layers_in, 'mem', [burn_value], raster_parameters, layer_number=layer_number,
                                       all_touched=all_touched)

    for i in range(0, raster_parameters.number_of_bands):
        array_burn = r_burn.get_array(i+1)
        array_mask = np.ma.masked_where(array_burn!=burn_value, array_burn)
        b_nodata = nodata[i]
        array_out[i] = np.where(array_mask.mask, b_nodata, array_out[i])
    del r_burn

    raster_out = RasterWriter(output_raster, raster_parameters) #nu, nv, nb, data_types[0], geo_trans, srs, driver=driver, nodata=nodata)
    for i in range(raster_parameters.number_of_bands):
        raster_out.set_array(array_out[i], i+1)
    raster_out.dataset.FlushCache()
    return raster_out


def clip_by_vector_pil(r_in, shp_in, r_out):
    """
    Args:
        r_in (string):

        shp_in (string): filename of the shapefile. The geometry must be line or polygon

        r_out (string):
    """

    layers = LayersReader(shp_in)

    # Get the union of the polygons and its envelope
    geom = ogr.CreateGeometryFromWkb(layers.get_geometry_union_as_wkb(shp_in))
    min_x, max_x, min_y, max_y = geom.GetEnvelope()

    r_in = RasterReader(r_in)
    rb_in_nodata = r_in.get_nodata()

    (u_min, u_max, v_min, v_max), geo_trans = r_in.extent_world_to_pixel(min_x, max_x, min_y, max_y)

    # rasterize the shapefile
    ds_shp, ly_shp = layers.get_layer(shp_in)
    ds0_shp = gdal.GetDriverByName('MEM').Create('', u_max-u_min, v_max-v_min, 1, gdal.GDT_Byte)
    ds0_shp.SetGeoTransform(geo_trans)
    ds0_shp.SetProjection(r_in.get_coordinate_system())
    gdal.RasterizeLayer(ds0_shp, [1], ly_shp, burn_values=[1])

    # Clip the image using the mask
    gim_pol = geometry_to_gdal_image(geom, geo_trans, u_max-u_min, v_max-v_min)
    clip = gdalnumeric.LoadFile(r_in.filename, u_min, v_min, u_max-u_min, v_max-v_min)

    dr_out = gdal.GetDriverByName('GTiff')
    ds_out = dr_out.Create(r_out, u_max - u_min, v_max - v_min, r_in.get_number_of_bands(), r_in.get_band_data_type())
    ds_out.SetGeoTransform(geo_trans)
    ds_out.SetProjection(r_in.get_coordinate_system())

    for i in range(r_in.get_number_of_bands()):
        # Write the array into the band
        rb_out = ds_out.GetRasterBand(i+1)
        rb_out.WriteArray(gdalnumeric.choose(gim_pol, (clip[i], 0)))
        rb_out.SetNoDataValue(rb_in_nodata[i])

    gdal.ErrorReset()


