from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import range
from past.builtins import basestring
from past.utils import old_div
import numbers
import numpy as np
from osgeo import gdal, ogr, gdal_array, gdalconst
from girs import srs
from girs.geom.envelope import is_intersect_envelope
from girs.rast.parameter import RasterParameters
from girs.rast.raster import RasterWriter, get_default_values
from girs.feat.layers import LayersReader


def rasterize_layers(layers, **kwargs):
    """Rasterize the input layer.

        - If input_layer is a file name or Layers object, layer_number will be used.
        - If not None, raster_parameters will be used to create the resulting rasters

    :param layers: layers file name or Layers object or ogr Layer object
    :type layers: str, girs.feat.layers.LayersSet, ogr.Layer
    :param kwargs:
        :key layer_number: (int, [int]): layer number or list of layer numbers
        :key burn_values: (int, [int]): unique burn value or list of burn values, one for each layer number
        :key raster_parameters: raster or raster parameter. If given, nodata and pixel size are not used
        :key nodata: (raster type, [raster type]): nodata or list of nodata, one for each layer number
        :key pixel_size: (float): pixel size if output_raster is a file name. If not given, the pixel size will be 1/100 of the narrowest layer extent (width or height)
        :key all_touched: True/False, default all_touched=False
        :key output_raster: girs.rast.raster.RasterWriter, girs.rast.raster.RasterUpdate, or filename of the output raster

    :return: girs.rast.raster.RasterWriter
    """
    # :key clip: (bool) clip to layers if True

    layer_number = kwargs.pop('layer_number', None)
    burn_values = kwargs.pop('burn_values', 1)
    nodata = kwargs.pop('nodata', 0)
    pixel_size = kwargs.pop('pixel_size', None)
    all_touched = kwargs.pop('all_touched', False)
    # clip = kwargs.pop('clip', False)
    output_raster = kwargs.pop('output_raster', None)
    raster_parameters = kwargs.pop('raster_parameters', None)

    try:
        lrs = LayersReader(layers)  # keep lrs alive
        layers = lrs
    except RuntimeError:
        pass

    if raster_parameters:
        try:
            rst_srs = raster_parameters.get_coordinate_system()
            lrs_srs = layers.get_coordinate_system()
            if not srs.is_same_srs(rst_srs, lrs_srs):
                lrs = layers.transform(wkt=rst_srs)
                layers = lrs
        except AttributeError:
            pass

    if layer_number:
        try:
            layers = [layers.get_layer(i) for i in layer_number]
        except TypeError:
            layers = [layers.get_layer(i) for i in [layer_number]]
    else:
        try:
            layers = [layers.get_layer(i) for i in range(layers.get_layer_count())]
        except AttributeError:  # lrs is an org-layer
            layers = [layers]

    if all(v is None for v in layers):
        msg = 'Layers not found. Layer numbers: {}'.format(layer_number)
        raise TypeError(msg)
    try:
        burn_values = list(burn_values)
    except TypeError:
        burn_values = [burn_values]

    options = ["ALL_TOUCHED=TRUE"] if all_touched else ["ALL_TOUCHED=FALSE"]

    raster_parameters = get_raster_parameters(raster_parameters, layers, pixel_size, nodata, burn_values)

    output_raster = RasterWriter(raster_parameters, output_raster)
    for bn in range(1, output_raster.get_band_count()+1):
        arr = output_raster.get_array(band_number=bn)
        arr[:, :] = raster_parameters.nodata[bn-1]
        output_raster.set_array(array=arr, band_number=bn)

    for ilyr, lyr in enumerate(layers):
        gdal.RasterizeLayer(output_raster.dataset, list(range(ilyr+1, ilyr+2)), lyr, None, None, burn_values=burn_values,
                            options=options)

    output_raster.dataset.FlushCache()

    return output_raster


def rasterize_layers_by_field(layers, field_names, **kwargs):
    """Rasterize a layer containing lines or polygons.

    Raster values are set according to field values found in field_name. If field_name is a list of field names, its
    length must coincide with the number of layers given in layer_in.

    The spatial reference system will be the same as from layer_in.

    If layer_in contains more than one layer and layer_number is not given, generate a raster with as many bands as
    layers.

    if no raster_out was defined, the raster will be created in the memory.It is possible to define a label for
    raster_out in the memory by explicitly setting the driver to 'MEM', e.g., raster_out='mylabel', driver='MEM'

    The parameters defining the raster properties are (in priority order):
        - raster_parameters: a raster will be created, transformed to the the coordinate system of layer_in and clipped
          according to layer_in. The raster parameters are retrieved from the new raster. Driver from raster parameters
          is not used
        - layer_in: raster parameters are obtained from envelope of layer_in; pixel_size = 1/100 of the narrowest layer extent (width or height) if pixel_size: (nx, ny) or n is not given

    :param layers: Layers filename, Layers object or a layer, If a Layers object was given, use the layer specified in layer_number
    :param field_names: the field name of the layer used to rasterize. The burn value will be unique for each unique value found in the field name
    :type field_names: str
    :param kwargs:
        :key layer_number: only used if layer_in is a Layers object. If layer_in is a Layers object and layer_number is unset, use all layers and create the corresponding number of bands
        :key raster_out: an existing rasters or filename of the new rasters
        :key driver: None
        :key raster_parameters:
        :key clip:
        :key pixel_size: None
    :return: a Raster object if raster_out is a file name, otherwise the same object as in raster_out but with the burned layer
    :rtype: RasterWriter
    """
    layer_number = kwargs.pop('layer_number', None)
    # field_names = kwargs.pop('field_names', 1)
    nodata = kwargs.pop('nodata', 0)
    pixel_size = kwargs.pop('pixel_size', None)
    all_touched = kwargs.pop('all_touched', False)
    output_raster = kwargs.pop('output_raster', None)
    raster_parameters = kwargs.pop('raster_parameters', None)
    driver = kwargs.pop('driver', None)

    try:
        layers = LayersReader(layers)
    except RuntimeError:
        pass

    if layer_number:
        layers = [layers.get_layer(i) for i in list(layer_number)]
    else:
        try:
            layers = [layers.get_layer(i) for i in range(layers.get_layer_count())]
        except AttributeError:  # lrs is an org-layer
            layers = [layers]

    # One field name for each layer
    if isinstance(field_names, basestring):
        field_names = [field_names] * len(layers)

    # Get field values list: a list of dictionaries, where each dictionary has field value as key and list of
    # corresponding features as value.
    field_values_list = list()
    for ilyr, lyr0 in enumerate(layers):
        ldf = lyr0.GetLayerDefn()
        field_values = dict()
        for ild in range(ldf.GetFieldCount()):
            fd = ldf.GetFieldDefn(ild)
            if field_names[ilyr] == fd.GetName():
                lyr0.ResetReading()
                for feat in lyr0:
                    value = feat.GetField(ild)
                    if value not in field_values:
                        field_values[value] = [feat]
                    else:
                        field_values[value].append(feat)
                lyr0.ResetReading()
                break
        if not field_values:
            raise Exception('rasterize_layer: no value found for field "{}"'.format(str(field_names[ilyr])))
        field_values_list.append(field_values)

    burn_values = np.array([fv for field_values in field_values_list for fv in list(field_values.keys())])
    sequential = not np.issubdtype(burn_values.dtype, np.number)
    if sequential:
        burn_values_map = {k+1: v for k, v in enumerate(set(burn_values.tolist()))}
    else:
        burn_values_map = {v: v for v in set(burn_values)}

    raster_parameters = get_raster_parameters(raster_parameters=raster_parameters, layers=layers,
                                              pixel_size=pixel_size, nodata=nodata,
                                              burn_values=list(burn_values_map.values()))
    array_out = np.empty((raster_parameters.number_of_bands,
                          raster_parameters.RasterYSize, raster_parameters.RasterXSize))

    for i in range(raster_parameters.number_of_bands):
        array_out[i] = np.empty((raster_parameters.RasterYSize, raster_parameters.RasterXSize)) * np.nan

    for ilyr, lyr in enumerate(layers):
        field_values = field_values_list[ilyr]  # dictionary with field values as key and list of features as value
        for ifv, field_value in enumerate(field_values):  # burn with a mask for each field value
            tmp_ds = ogr.GetDriverByName('Memory').CreateDataSource('tmp')
            tmp_lyr = tmp_ds.CreateLayer('', srs=lyr.GetSpatialRef())
            for feat in field_values[field_value]:  # get all features sharing this field_value
                tmp_lyr.CreateFeature(feat)
            burn_values = [ifv+1] if sequential else np.array([field_value])
            r_out = rasterize_layers(layers=tmp_lyr, burn_values=burn_values, raster_parameters=raster_parameters,
                                     all_touched=all_touched)
            array_burn = r_out.get_array()
            array_mask = np.ma.masked_where(array_burn != burn_values[0], array_burn)
            array_out[ilyr] = np.where(array_mask.mask, array_out[ilyr], array_burn)
            del tmp_ds
    r_out = RasterWriter(raster_parameters=raster_parameters, source=output_raster, drivername=driver)
    for i in range(raster_parameters.number_of_bands):
        arr = array_out[i]
        arr[np.isnan(arr)] = raster_parameters.nodata[i]
        r_out.set_array(arr, i+1)
        r_out.dataset.FlushCache()
    burn_values_map['nodata'] = raster_parameters.nodata
    return r_out, burn_values_map


def get_raster_parameters(raster_parameters, layers, pixel_size, nodata, burn_values):
    """Return raster parameters from layer

    :param raster_parameters:
    :param layers:
    :param pixel_size:
    :param nodata:
    :param burn_values:
    :param clip:
    :return:
    """
    try:
        if not isinstance(raster_parameters, RasterParameters):
            rst = raster_parameters
            raster_parameters = rst.get_parameters()

        if not srs.is_same_srs(raster_parameters.get_coordinate_system(), layers[0].GetSpatialRef()):
            msg = 'Raster and layer are not in the same coordinate system: \n\t{}, \n\t{}'.format(
                raster_parameters.get_coordinate_system(), srs.export(layers[0].GetSpatialRef(), fmt='wkt'))
            raise TypeError(msg)
        lrs_env = list(layers[0].GetExtent())
        rst_env = raster_parameters.get_extent_world()
        if not is_intersect_envelope(lrs_env, rst_env):
            msg = "Raster does not intersect the layer: no rasterization possible."
            raise TypeError(msg)
        return raster_parameters
    except AttributeError as e:
        pass

    number_of_bands = len(layers)
    rs = layers[0].GetSpatialRef().ExportToWkt()
    env = layers[0].GetExtent()

    xmin, xmax, ymin, ymax = env
    dx, dy = xmax - xmin, ymax - ymin
    if not pixel_size:
        pixel_size = max(dx, dy) / 100.0
        pixel_size = (pixel_size, pixel_size)
    nx, ny = int(old_div(dx, pixel_size[0])), int(old_div(dy, pixel_size[1]))
    geo_trans = [xmin, pixel_size[0], 0, ymax, 0, -pixel_size[1]]

    burn_values = np.array(burn_values)
    if np.issubdtype(burn_values.dtype, np.integer):
        dtype_info = np.iinfo(burn_values.dtype)
        min_type, max_type = dtype_info.min, dtype_info.max
        if nodata is None or nodata in burn_values:
            if nodata == 0:
                nodata = max_type
                while nodata in burn_values and nodata > min_type:
                    nodata -= 1
            else:
                nodata = min_type
                while nodata in burn_values and nodata < max_type:
                    nodata += 1
    else:
        dtype_info = np.finfo(burn_values.dtype)
        min_type, max_type, resolution = dtype_info.min, dtype_info.max, dtype_info.resolution
        if nodata is None or nodata in burn_values:
            found = False
            for nd in np.arange(min_type, max_type, resolution):
                if nd not in burn_values:
                    found = True
                    break
            assert found
    assert min_type <= nodata <= max_type
    if issubclass(burn_values.dtype.type, np.int64):
        data_types = gdalconst.GDT_Int32
    elif issubclass(burn_values.dtype.type, np.uint64):
        data_types = gdalconst.GDT_UInt32
    elif issubclass(burn_values.dtype.type, np.int0):
        data_types = gdalconst.GDT_Int16
    elif issubclass(burn_values.dtype.type, np.uint0):
        data_types = gdalconst.GDT_Byte
    else:
        data_types = gdal_array.NumericTypeCodeToGDALTypeCode(burn_values.dtype.type)
    nodata = get_default_values(number_of_bands=number_of_bands, values=nodata)
    return RasterParameters(nx, ny, geo_trans, rs, number_of_bands, nodata, data_types, driver_short_name=None)

