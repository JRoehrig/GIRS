from builtins import range
from collections import Counter
import numpy as np
from girs.rast.raster import RasterReader, RasterWriter
from girs.feat.layers import LayersReader
from girs.rastfeat.rasterize import rasterize_layers
from osgeo import gdal, gdalconst, ogr, osr
gdal.UseExceptions()


def zonal_stats(layers, input_raster, field_name, stats, **kwargs):
    """Return a dictionary with zonal statistics

    The method:

        1. collects layer geometries sharing the same value of field_name
        2. gor each geometry, calculate raster statistics for all pixels coinciding with the geometry
        3. returns a dictionary with layer's field values as keys and a dictionary as value
            The dictionary has the raster band as key and a list of statistics as values

    Example of a result (stats=['min', 'max'] for layer values 10 and 20 and a two bands raster:
        result = {10: {1:[1.92, 67.34], 2:[3.47, 12.87]}, 20: {1:[2.33, 79.41], 2:[4.17, 13.55]}}



    :param layers: layer
    :type layers: LayersSet, filename of a LayersSet, or an ogr-layer
    :param input_raster: raster
    :type input_raster: Raster, filename of a Raster
    :param field_name: the layer's field name. The burn value will be different for each field value
    :type field_name: str
    :param stats: 'min', 'max', 'mean', 'sum', 'count', or 'std'
    :type stats: str, list
    :param kwargs:
        :key layer_number: (int) layer number, default is zero
        :key all_touched: True/False, default all_touched=False
    :return: dictionary with the record values from field field_name as keys and a list of statistics in the same order as given in stats
    :rtype: dict
    """

    all_touched = kwargs.pop('all_touched', False)
    layer_number = kwargs.pop('layer_number', 0)

    result = {}

    if not stats:
        return result

    def catch_index_error(z, idx):
        try:
            z.most_common()[idx][0]
        except IndexError:
            return None

    cstats = ['min', 'max', 'mean', 'sum', 'count', 'std']

    if stats[0] == 'all':
        zs_functions = [eval("lambda ir, ic, z: float(ir." + s + '())') for s in cstats]
    else:
        zs_functions = [eval("lambda ir, ic, z: float(ir." + s + '())') for s in stats if s in cstats]

    has_compressed_mask = False
    has_compressed_mask_count = False
    for s in stats:
        if s == 'median' or s == 'all':
            zs_functions.append(lambda x, y, z: float(np.median(y)))
            has_compressed_mask = True
        if s == 'range' or s == 'all':
            zs_functions.append(lambda x, y, z: float(x.max()) - float(x.min()))
        if s == 'unique' or s == 'all':
            zs_functions.append(lambda x, y, z: len(list(z.keys())))
            has_compressed_mask_count = True
        elif s == 'minority' or s == 'all':
            zs_functions.append(lambda x, y, z: catch_index_error(z, -1))
            # zs_functions.append(zs_minority)
            has_compressed_mask_count = True
        elif s == 'majority' or s == 'all':
            zs_functions.append(lambda x, y, z: catch_index_error(z, 1))
            # zs_functions.append(zs_majority)
            has_compressed_mask_count = True
        elif s.startswith('percentile_'):
            zs_functions.append(lambda x, y, z: np.percentile(y, float(s[11:])) if y.size != 0 else None)

    if not zs_functions:
        return result

    try:
        input_raster = RasterReader(input_raster)
    except AttributeError as e:
        pass

    nb = input_raster.get_band_count()
    try:
        input_layer = LayersReader(layers)
        lyr = input_layer.get_layer(layer_number)
    except (TypeError, RuntimeError):
        try:
            lyr = layers.get_layer(layer_number)
        except AttributeError:
            lyr = layers

    srs = lyr.GetSpatialRef()

    field_values = {}
    ldf = lyr.GetLayerDefn()
    for i in range(ldf.GetFieldCount()):
        fd = ldf.GetFieldDefn(i)
        fn = fd.GetName()
        if field_name == fn:
            lyr.ResetReading()
            for feat in lyr:
                value = feat.GetField(i)
                if value not in field_values:
                    field_values[value] = [feat]
                else:
                    field_values[value].append(feat)
            break

    array_in = input_raster.get_array()

    raster_parameters = input_raster.get_parameters()
    raster_parameters.driverShortName = 'MEM'
    raster_parameters.number_of_bands = 1
    raster_parameters.set_nodata(0)
    for field_value in list(field_values.keys()):
        tmp_ds = ogr.GetDriverByName('Memory').CreateDataSource('tmp')
        tmp_lyr = tmp_ds.CreateLayer('', srs=srs)
        for feat in field_values[field_value]:
            tmp_lyr.CreateFeature(feat)
        r_burned = rasterize_layers(tmp_lyr, burn_values=[1], raster_parameters=raster_parameters,
                                    all_touched=all_touched)
        a_burned = r_burned.get_array(1)
        del r_burned
        if np.any(a_burned):
            result[field_value] = {}
            for bn in range(1, nb + 1):
                array_mask = np.ma.MaskedArray(array_in, np.ma.logical_not(a_burned))
                compressed_mask = array_mask.compressed() if has_compressed_mask else None
                if has_compressed_mask_count:
                    compressed_mask_count = Counter(compressed_mask if has_compressed_mask else array_mask.compressed())
                else:
                    compressed_mask_count = None
                result[field_value][bn] = [f(array_mask, compressed_mask, compressed_mask_count) for f in zs_functions]
        else:
            result[field_value] = {bn: [None] * len(zs_functions) for bn in range(1, nb + 1)}
    return result


def zonal_stats_raster(layers, input_raster, field_name, stats, **kwargs):
    """Return a raster with zonal statistics

    The method:

        1. collects layer geometries sharing the same value of field_name
        2. gor each geometry, calculate raster statistics for all pixels coinciding with the geometry
        3. returns a dictionary with layer's field values as keys and a dictionary as value
            The dictionary has the raster band as key and a list of statistics as values


    :param layers: layer
    :type layers: LayersSet, filename of a LayersSet, or an ogr-layer
    :param input_raster: raster
    :type input_raster: Raster, filename of a Raster
    :param field_name: the layer's field name. The burn value will be different for each field value
    :type field_name: str
    :param stats: One of 'min', 'max', 'mean', 'sum', 'count', or 'std'
    :type stats: str
    :param kwargs:
        :key layer_number: (int) layer number, default is zero
        :key all_touched: True/False, default all_touched=False
        :key output_raster: girs.rast.raster.RasterWriter, girs.rast.raster.RasterUpdate, or filename of the output raster
    :return: RasterWriter with the result of the given stats
    :rtype: RasterWriter
    """

    layer_number = kwargs.pop('layer_number', 0)
    all_touched = kwargs.pop('all_touched', False)
    output_raster = kwargs.pop('output_raster', None)

    result = {}

    if not stats:
        return result

    def catch_index_error(z, idx):
        try:
            z.most_common()[idx][0]
        except IndexError:
            return None

    cstats = ['min', 'max', 'mean', 'sum', 'count', 'std']

    if stats[0] == 'all':
        zs_functions = [eval("lambda ir, ic, z: float(ir." + s + '())') for s in cstats]
    else:
        zs_functions = [eval("lambda ir, ic, z: float(ir." + s + '())') for s in cstats if s in stats]

    has_compressed_mask = False
    has_compressed_mask_count = False
    for s in stats:
        if s == 'median':
            zs_functions.append(lambda x, y, z: float(np.median(y)))
            has_compressed_mask = True
        if s == 'range':
            zs_functions.append(lambda x, y, z: float(x.max()) - float(x.min()))
        if s == 'unique':
            zs_functions.append(lambda x, y, z: len(list(z.keys())))
            has_compressed_mask_count = True
        elif s == 'minority':
            zs_functions.append(lambda x, y, z: catch_index_error(z, -1))
            # zs_functions.append(zs_minority)
            has_compressed_mask_count = True
        elif s == 'majority':
            zs_functions.append(lambda x, y, z: catch_index_error(z, 1))
            # zs_functions.append(zs_majority)
            has_compressed_mask_count = True
        elif s.startswith('percentile_'):
            zs_functions.append(lambda x, y, z: np.percentile(y, int(s[11:])) if y.size != 0 else None)

    if len(zs_functions) > 1:
        raise TypeError('More than one statistics given: {}'.format(stats))
    if not zs_functions:
        raise TypeError('No valied statistics given: {}'.format(stats))
    zs_function = zs_functions[0]

    try:
        input_raster = RasterReader(input_raster)
    except AttributeError as e:
        pass

    nb = input_raster.get_band_count()
    try:
        input_layer = LayersReader(layers)
        lyr = input_layer.get_layer(layer_number)
    except (TypeError, RuntimeError):
        try:
            lyr = layers.get_layer(layer_number)
        except AttributeError:
            lyr = layers

    srs = lyr.GetSpatialRef()

    field_values = {}
    ldf = lyr.GetLayerDefn()
    for i in range(ldf.GetFieldCount()):
        fd = ldf.GetFieldDefn(i)
        fn = fd.GetName()
        if field_name == fn:
            lyr.ResetReading()
            for feat in lyr:
                value = feat.GetField(i)
                if value not in field_values:
                    field_values[value] = [feat]
                else:
                    field_values[value].append(feat)
            break

    array_in = input_raster.get_array()
    raster_parameters = input_raster.get_parameters()
    array_out = np.empty((raster_parameters.number_of_bands,
                          raster_parameters.RasterYSize, raster_parameters.RasterXSize))
    for i in range(raster_parameters.number_of_bands):
        array_out[i] = np.empty((raster_parameters.RasterYSize, raster_parameters.RasterXSize)) * np.nan

    rp = input_raster.get_parameters()
    rp.driverShortName = 'MEM'
    rp.number_of_bands = 1
    rp.set_nodata(0)
    for field_value in list(field_values.keys()):
        tmp_ds = ogr.GetDriverByName('Memory').CreateDataSource('tmp')
        tmp_lyr = tmp_ds.CreateLayer('', srs=srs)
        for feat in field_values[field_value]:
            tmp_lyr.CreateFeature(feat)
        r_burned = rasterize_layers(tmp_lyr, burn_values=[1], raster_parameters=rp, all_touched=all_touched)
        a_burned = r_burned.get_array(1)
        del r_burned
        if np.any(a_burned):
            result[field_value] = {}
            for bn in range(1, nb + 1):
                array_mask = np.ma.MaskedArray(array_in, np.ma.logical_not(a_burned))
                if has_compressed_mask:
                    compressed_mask = array_mask.compressed()
                else:
                    compressed_mask = None
                if has_compressed_mask_count:
                    compressed_mask_count = Counter(compressed_mask if has_compressed_mask else array_mask.compressed())
                else:
                    compressed_mask_count = None
                value = zs_function(array_mask, compressed_mask, compressed_mask_count)
                array_out[bn-1] = np.where(array_mask.mask, array_out[bn-1], value)

    r_out = RasterWriter(raster_parameters=raster_parameters, source=output_raster)
    for i in range(raster_parameters.number_of_bands):
        arr = array_out[i]
        arr[np.isnan(arr)] = raster_parameters.nodata[i]
        r_out.set_array(arr, i+1)
        r_out.dataset.FlushCache()
    return r_out
