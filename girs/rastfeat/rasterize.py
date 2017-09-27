import numpy as np
from osgeo import gdal, ogr

from girs.rast.parameter import RasterParameters
from girs.rast.raster import RasterWriter, get_default_values
from girs.feat.layers import LayersReader


def rasterize_layer(input_layer, output_raster, values, raster_parameters=None, layer_number=0,
                    pixel_size=None, all_touched=False):
    if isinstance(values, basestring):
        return _layer_to_raster_by_field(input_layer, output_raster, values, raster_parameters, layer_number, pixel_size)
    else:
        return _layer_to_raster_by_value(input_layer, output_raster, values, raster_parameters, layer_number,
                                         pixel_size, all_touched)


def _layer_to_raster_by_value(lrs, output_raster, values, raster_parameters, layer_number=0, pixel_size=None,
                              all_touched=False):
    """Rasterize the input layer.
        - If input_layer is a file name or Layers object, layer_number will be used.
        - If not None, raster_parameters will be used to create the resulting rasters

    Args:
        input_layer (str or Layers or Layer): layers file name or Layers object or Layer object.

        output_raster (str): an existing rasters or filename of the new rasters

        values (number or list): unique burn value or list with one burn value for each rasters band

        layer_number

        nodata: unique nodata value for all bands or list of nodata values with end equal number of bands

        pixel_size (float): pixel size if output_raster is a file name. If not given, the pixel size will be 1/100
            of the narrowest layer extent (width or height)

    return:
        rasters (Raster): a Raster object if raster_out is a file name, otherwise the same object as in raster_out but
            with the burned layer
    """
    # Check combination of parameters

    try:
        lrs = LayersReader(lrs)
        lyr = lrs.get_layer(layer_number)
    except (TypeError, RuntimeError):
        try:
            lyr = lrs.get_layer(layer_number)
        except AttributeError:
            lyr = lrs

    if raster_parameters:
        values = get_default_values(raster_parameters.number_of_bands, values)
    else:
        rs = lyr.GetSpatialRef().ExportToWkt()
        # Create the rasters
        xmin, xmax, ymin, ymax = lyr.GetExtent()
        dx, dy = xmax - xmin, ymax - ymin
        if not pixel_size:
            pixel_size = max(dx, dy) / 100.0
        nx, ny = int(dx / pixel_size), int(dy / pixel_size)
        gt = [xmin, pixel_size, 0, ymax, 0, -pixel_size]
        try:
            nb = len(values)
        except TypeError, te:
            nb = 1
            values = [values]
        except:
            raise
        nodata = 0
        for i in range(256):
            if i not in values:
                nodata = i
                break
        raster_parameters = RasterParameters(nx, ny, gt, rs, nb, nodata, gdal.GDT_Byte, driver_short_name=None)

    output_raster = RasterWriter(output_raster, raster_parameters)
    options = []
    if all_touched:
        options.append("ALL_TOUCHED=TRUE")
    gdal.RasterizeLayer(output_raster.dataset, range(1, raster_parameters.number_of_bands+1), lyr, None, None,
                        burn_values=values, options=options)
    return output_raster


def _layer_to_raster_by_field(layer_in, raster_out, field_name, raster_parameters, layer_number=0, pixel_size=None,
                              driver=None):
    """Rasterize a Layer object containing lines or polygons.

    Args:
        layer_in (str or Layers or Layer): Layer object, Layers object or layers file name. If a Layers object was
            given, use the layer specified in layer_number

        field_name: the field name of the layer used to rasterize. The burn value will be different for each field value.

        output_raster (str): an existing rasters or filename of the new rasters

        kwargs:
            'layer_number' (int): layer number in case layer_in is a Layers object or layers file name. Default is
                layer number = 0
            'pixel_size' (float): pixel size if output_raster is a file name. If not given, the pixel size will be 1/100
                of the narrowest layer extent (width or height)
            'scale' (float): to scale the bounding box of the layers_in
            'driver' (str): driver name for raster_out. if raster_out is a gdal dataset or Raster object, the driver
                will be negletet
    return:
        rasters (Raster): a Raster object if raster_out is a file name, otherwise the same object as in raster_out but
            with the burned layer
    """

    if not driver:
        driver = get_driver_name_from_file_name(raster_out)

    try:
        input_layer = LayersReader(layer_in)
        lyr = input_layer.get_layer(layer_number)
    except TypeError:
        try:
            lyr = input_layer.get_layer(layer_number)
        except AttributeError:
            lyr = input_layer

    srs = lyr.GetSpatialRef()

    field_values = {}
    ldf = lyr.GetLayerDefn()
    for i in range(ldf.GetFieldCount()):
        fd = ldf.GetFieldDefn(i)
        if field_name == fd.GetName():
            for feat in lyr:
                value = feat.GetField(i)
                if value not in field_values:
                    field_values[value] = [feat]
                else:
                    field_values[value].append(feat)
            break

    if not field_values:
        raise Exception('rasterize_layer: no value found for field "' + field_name + '"')

    array_out = None
    nb = None
    nu, nv = None, None
    geo_trans = None
    for field_value in field_values.keys():
        tmp_ds = ogr.GetDriverByName('Memory').CreateDataSource('tmp')
        tmp_lyr = tmp_ds.CreateLayer('', srs=srs)
        for feat in field_values[field_value]:
            tmp_lyr.CreateFeature(feat)
        r_out = _layer_to_raster_by_value(tmp_lyr, 'mem', [field_value], raster_parameters, 0, pixel_size)
        if array_out is None:
            nb = r_out.get_number_of_bands()
            nu, nv = r_out.get_raster_size()
            geo_trans = r_out.get_geotransform()
            array_out = np.empty((nb, nv, nu))
            for i in range(nb):
                array_out[i] = np.empty((nv, nu)) * np.nan
        for i in range(0, nb):
            array_burn = r_out.get_array(i+1)
            array_mask = np.ma.masked_where(array_burn!=field_value, array_burn)
            array_out[i] = np.where(array_mask.mask, array_out[i], array_burn)

    raster_out = RasterWriter(raster_out, nu, nv, nb, gdal.GDT_Byte, geo_trans, srs.ExportToWkt(), driver=driver, nodata=nodata)
    for i in range(nb):
        raster_out.set_array(array_out[i], i+1)
    raster_out.dataset.FlushCache()
    return raster_out

