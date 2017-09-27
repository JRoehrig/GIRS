from collections import Counter
import numpy as np
from girs.rast.raster import RasterReader
from girs.feat.layers import LayersReader
from girs.rastfeat.rasterize import rasterize_layer
try:
    from osgeo import gdal, gdalconst, ogr, osr
    gdal.UseExceptions()
except:
    import girs
    import gdalconst
    import ogr
    import osr

# See https://pcjericks.github.io/py-gdalogr-cookbook/raster_layers.html


def zonal_stats(layers, input_raster, field_name, stats, layer_number=0):
    """Rasterize a Layer object containing lines or polygons.

    Args:
        layer_in (str): Layers file name

        input_raster (str): an existing rasters or filename of the new rasters

        field_name: the field name of the layer used to rasterize. The burn value will be different for each field value.

    return:
        dictionary with the record values from field field_name as keys and a list of statistics in the same order as
            given in stats
    """

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
    except:
        pass

    nb = input_raster.get_number_of_bands()
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

    for field_value in field_values.keys():
        tmp_ds = ogr.GetDriverByName('Memory').CreateDataSource('tmp')
        tmp_lyr = tmp_ds.CreateLayer('', srs=srs)
        for feat in field_values[field_value]:
            tmp_lyr.CreateFeature(feat)
        raster_parameters = input_raster.get_parameters()
        raster_parameters.driverShortName = 'MEM'
        r_out = rasterize_layer(tmp_lyr, 'mem', [1], raster_parameters, 0, None)
        for i in range(0, nb):
            if not np.any(r_out.get_array(i+1)):
                r_out = rasterize_layer(tmp_lyr, 'mem', [1], raster_parameters, 0, None, all_touched=True)
                break

        result[field_value] = []
        for i in range(0, nb):
            array_burn = r_out.get_array(i+1)
            del r_out
            array_mask = np.ma.MaskedArray(array_in, np.ma.logical_not(array_burn))
            compressed_mask = array_mask.compressed() if has_compressed_mask else None
            compressed_mask_count = Counter(compressed_mask if has_compressed_mask else array_mask.compressed()) if has_compressed_mask_count else None
            result[field_value].append([f(array_mask, compressed_mask, compressed_mask_count) for f in zs_functions])
    return result

