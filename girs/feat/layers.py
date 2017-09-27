import operator
import os
import subprocess
import time
from abc import ABCMeta

import numpy as np
import pandas as pd
from osgeo import gdal, ogr, osr
from scipy.spatial import Voronoi

from girs import srs
from girs.feat import ogrpandas
from girs.util.osu import assure_path_exists, create_tmp_directory, remove_directory

ogr.UseExceptions()
# ogr.DontUseExceptions()


class DataFrameGeometry(object):

    def __init__(self, layer, feature_id):
        self.layer = layer
        self.feature_id = feature_id

    def __repr__(self):
        return ogr.GeometryTypeToName(self.layer.GetGeomType())

    def get_geometry_type(self):
        return self.layer.GetGeomType()

    def get_id(self):
        return self.feature_id

    def apply(self, method, *args, **kwargs):
        feat = self.layer.GetFeature(self.feature_id)
        g = feat.GetGeometryRef()
        result = getattr(g, method)(*args, **kwargs)
        del feat
        return result


def _get_extension_shortname_dict():

    def _insert_extensions(drv_dict, short_name, extensions=None):
        extensions = extensions.split() if extensions else [None]
        for ext in extensions:
            if ext:
                if ext.startswith('.'):
                    ext = ext[1:]
                ext = ext.lower()
                for ext1 in [e for e in ext.split('/')]:
                    if ext1 not in drv_dict:
                        drv_dict[ext1] = []
                    drv_dict[ext].append(short_name)
            else:
                if None not in drv_dict:
                    drv_dict[None] = []
                drv_dict[None].append(short_name)

    gdal.AllRegister()
    drivers_dict = {}
    for i in range(ogr.GetDriverCount()):
        drv = ogr.GetDriver(i)
        metadata = drv.GetMetadata()
        driver_name = drv.GetName()
        if gdal.DMD_EXTENSION in metadata:
            _insert_extensions(drivers_dict, driver_name, metadata[gdal.DMD_EXTENSION])
        elif gdal.DMD_EXTENSIONS in metadata:
            _insert_extensions(drivers_dict, driver_name, metadata[gdal.DMD_EXTENSIONS])
        else:
            _insert_extensions(drivers_dict, driver_name)
    return drivers_dict


class FeatDrivers(object):

    extension_driver_name_dict = _get_extension_shortname_dict()

    @staticmethod
    def get_driver(source='', drivername=None):
        """Returns an object ``osgeo.ogr.Driver`` or ``None``.
        Returns a Memory driver:
        1- If drivername='Memory',
        2- If drivername=``None`` and source=``None``
        3- If drivername=``None`` and source=''
        :param source:
        :param drivername:
        :return: ``osgeo.ogr.Driver`` or ``None``
        """

        if drivername:
            return ogr.GetDriverByName(drivername)
        elif source:
            extension = source.split('.')[-1].strip()
            driver_names = FeatDrivers.extension_driver_name_dict[extension]
            if len(driver_names) == 1:
                return ogr.GetDriverByName(driver_names[0])
            else:
                raise ValueError('Filename extension ambiguous: {}'.format(', '.join(driver_names)))
        else:
            return ogr.GetDriverByName('Memory')

    @staticmethod
    def get_driver_names():
        """Return all available ogr driver names
        :return:
        """
        return [ogr.GetDriver(i).GetName() for i in range(ogr.GetDriverCount())]


class LayersSet(object):

    __metaclass__ = ABCMeta

    def __init__(self):
        self.dataset = None

    def get_source(self):
        return self.dataset.GetName()

    def get_description(self, layer_number=0):
        return self.get_layer(layer_number).GetDescription()

    def show(self, width=200, max_rows=60):
        pd.set_option("display.width",width)
        pd.set_option("display.max_rows",max_rows)
        print self.data_frame()
        pd.reset_option("display.width")
        pd.reset_option("display.max_rows")

    def data_frame(self, layer_number=0):

        def ogr_to_dtype(t):
            if t == ogr.OFTInteger:
                return np.int64
            elif t == ogr.OFTReal:
                return np.float64
            else:
                return object

        ld = self.get_layer_definition(layer_number)
        nf = ld.GetFieldCount()
        fds = [ld.GetFieldDefn(i) for i in range(ld.GetFieldCount())]
        names, types = zip(*[[fd.GetName(), fd.GetType()] for fd in fds])
        names_types = dict(zip(names, types))
        geometry_column_name = 'geom'
        while geometry_column_name in names:
            geometry_column_name += '0'
        columns = [geometry_column_name] + list(names)
        ogr_layer = self.get_layer(layer_number)
        ogr_layer.ResetReading()
        values = [[DataFrameGeometry(ogr_layer, f.GetFID())] + [f.GetField(i) for i in range(nf)] for f in ogr_layer]
        df = pd.DataFrame(values, columns=columns)
        s_dict = df.to_dict('series')
        names_types[geometry_column_name] = None
        for col, series in s_dict.items():
            t = ogr_to_dtype(names_types[col])
            s_dict[col] = s_dict[col].astype(t)
        return pd.DataFrame(s_dict, columns=columns)

    def copy(self, target='', *args, **kwargs):
        """
        Args:
            args[layer_index] (list): [field_name, field_values, filter_function] or None
                layer_index is the index of the layer for which the filter applies.
                If args[layer_index] = None, all fields will be copied
            kwargs['field_names']: fields to keep
        """
        target = '' if target is None else target.strip()
        if target:
            assure_path_exists(target)
        lrs = LayersWriter(target)

        field_names = kwargs['field_names'] if 'field_names' in kwargs else None

        def create_feature(lyr_out, fd_out, feat_in, indices_in):
            feat_new = ogr.Feature(fd_out)
            for idx_out, idx_in in enumerate(indices_in):
                feat_new.SetField(idx_out, feat_in.GetField(idx_in))
            g = feat_in.GetGeometryRef()
            feat_new.SetGeometry(g)
            lyr_out.CreateFeature(feat_new)

        for ilc in range(self.dataset.GetLayerCount()):
            ly_in = self.dataset.GetLayer(ilc)
            ld_in = ly_in.GetLayerDefn()
            if field_names and ilc < len(field_names) and field_names[ilc]:
                fields_indices_keep = sorted([ld_in.GetFieldIndex(fn) for fn in set(field_names[ilc])])
            else:
                fields_indices_keep = range(ld_in.GetFieldCount())
            fd_out = [ld_in.GetFieldDefn(ifd) for ifd in fields_indices_keep]
            ly_out = lrs.create_layer(str(ilc), ly_in.GetSpatialRef(), ly_in.GetGeomType(), fd_out)
            ld_out = ly_out.GetLayerDefn()
            # copy features
            ly_in.ResetReading()
            if args and ilc < len(args) and args[ilc]:
                parameters = args[ilc]
                field_name = parameters[0]
                field_values = parameters[1]
                filter_function = parameters[2]
                if field_name == 'FID':
                    for feat in ly_in:
                        if filter_function(feat.GetFID(), field_values):
                            create_feature(ly_out, ld_out, feat, fields_indices_keep)
                        feat = None
                else:
                    field_index = ld_in.GetFieldIndex(field_name)
                    if field_index == -1:
                        raise Exception('Field {} could not be found in {}'.format(field_name, self.get_source()))
                    for feat in ly_in:
                        if filter_function(feat.GetField(field_index), field_values):
                            create_feature(ly_out, ld_out, feat, fields_indices_keep)
                        feat = None
            else:
                for feat in ly_in:
                    create_feature(ly_out, ld_out, feat, fields_indices_keep)
                    feat = None

            if field_names and ilc < len(field_names) and field_names[ilc]:
                ld_out = ly_out.GetLayerDefn()
                fields_all = set([ld_in.GetFieldDefn(i).GetName() for i in range(ld_in.GetFieldCount())])
                fields_keep = set(field_names[ilc])
                fields_delete = fields_all - fields_keep
                indices = sorted([ld_out.GetFieldIndex(fn) for fn in fields_delete], reverse=True)
                for idx in indices:
                    if idx >= 0:
                        ly_out.DeleteField(idx)
        lrs.dataset.FlushCache()
        return lrs

    def transform(self, target='', **kwargs):
        """
        kwargs:
            keys: 'epsg', 'epsga', 'erm', 'esri', 'mi', 'ozi', 'pci', 'proj4', 'url', 'usgs', 'wkt', 'xml'
            values: the coordinate system
        """

        if kwargs is None or len(kwargs) == 0:
            return
        elif 'srs' in kwargs:
            sr_out = kwargs['srs']
        else:
            sr_out = srs.srs(**kwargs)

        if target:
            assure_path_exists(target)
        lrs_out = LayersWriter(target)

        for ilc in range(self.dataset.GetLayerCount()):
            ly_in = self.dataset.GetLayer(ilc)
            ld_in = ly_in.GetLayerDefn()
            sr_in = ly_in.GetSpatialRef()
            field_definitions = []
            for ifd in range(ld_in.GetFieldCount()):
                field_definitions.append(ld_in.GetFieldDefn(ifd))
            ly_out = lrs_out.create_layer(str(ilc), sr_out, # ly_in.GetSpatialRef(),
                                          ly_in.GetGeomType(), field_definitions)
            coord_trans = osr.CoordinateTransformation(sr_in, sr_out)
            # copy features
            ly_in.ResetReading()
            for feat_in in ly_in:
                geom_in = feat_in.GetGeometryRef()
                geom_out = geom_in.Clone()
                geom_out.Transform(coord_trans)
                feat_out = feat_in.Clone()
                feat_out.SetGeometry(geom_out)
                ly_out.CreateFeature(feat_out)
            ly_in.ResetReading()
        return lrs_out


        # lrs_out = Layers(target, 2)
        #
        # for ilc in range(self.dataset.GetLayerCount()):
        #     ly_in = self.dataset.GetLayer(ilc)
        #     sr_in = ly_in.GetSpatialRef()
        #     ld_in = ly_in.GetLayerDefn()
        #
        #     coord_trans = osr.CoordinateTransformation(sr_in, sr_in)
        #
        #     values = []
        #     for ifd in range(ld_in.GetFieldCount()):
        #         values.append(ld_in.GetFieldDefn(ifd))
        #
        #     ly_out = lrs_out.create_layer(get_layer_name(target), sr_out, ly_in.GetGeomType(), values)
        #
        #     # copy features
        #     ly_in.ResetReading()
        #     for i in range(ly_in.GetFeatureCount()):
        #         feat_in = ly_in.GetFeature(i)
        #         feat_out = feat_in.Clone()
        #         # ly_out.CreateFeature(feat_out)
        #         geom_in = feat_in.GetGeometryRef()
        #         geom_out = geom_in.Clone()
        #         # geom_out.Transform(coord_trans)
        #         feat_out.SetGeometry(geom_out)
        #         # feat_out.Destroy()
        #         # print geom_in.ExportToWkt()
        #         # print geom_out.ExportToWkt()
        #         del feat_out
        #
        #     del ly_in
        #     del sr_in
        #     del ld_in
        #
        # lrs_out.destroy()

    def get_layer_count(self):
        return self.dataset.GetLayerCount()

    def get_layer(self, layer_number=0):
        return self.dataset.GetLayer(layer_number)

    def layers(self):
        for i in range(self.dataset.GetLayerCount()):
            yield self.dataset.GetLayer(i)

    def get_layer_name(self, layer_number=0):
        return self.get_layer(layer_number).GetName()

    def get_layer_definition(self, layer_number=0):
        return self.get_layer(layer_number).GetLayerDefn()

    def layers_definitions(self):
        for i in range(self.dataset.GetLayerCount()):
            ly = self.dataset.GetLayer(i)
            yield ly.GetLayerDefn()

    def generate_field_name(self, field_name, layer_number=0):
        """
        Returns field_name if it does not exist, else append a digit from 0 to n to field_name and return it
        :param field_name:
        :param layer_number: default = 0
        :return:
        """

        ld = self.get_layer_definition(layer_number)
        if ld.GetFieldIndex(field_name) == -1:
            return field_name

        fn = str(field_name)
        digits = []
        while fn[-1].isdigit():
            digits.append(fn[-1])
            fn = fn[:-1]
        if not digits:
            return field_name + '0'
        else:
            i = int(''.join(digits))
            field_name = fn
            fn = field_name + str(i+1)
            while ld.GetFieldIndex(fn) > -1:
                i += 1
                fn = field_name + str(i)

        return fn

    def has_fields(self, field_names, layer_number=0):
        assert not isinstance(field_names, basestring)
        ld = self.get_layer_definition(layer_number)
        return [ld.GetFieldIndex(field_name) > -1 for field_name in field_names]

    def get_fields(self, layer_number=0):
        """Returns a list of lists containing field name, type code, type name, width, and precision
        """
        ld = self.get_layer_definition(layer_number)
        result = []
        for i in range(ld.GetFieldCount()):
            fd = ld.GetFieldDefn(i)
            field_name = fd.GetName()
            field_type_code = fd.GetType()
            field_type_name = fd.GetFieldTypeName(field_type_code)
            field_width = fd.GetWidth()
            field_precision = fd.GetPrecision()
            result.append([field_name, field_type_code, field_type_name, field_width, field_precision])
        return result

    def get_field_numbers(self, field_names, layer_number=0):
        """Return the field numbers for the given field names. Set field number = -1 if field name does not exist
        """
        layer_field_names = self.get_field_names(layer_number)
        layer_field_names_dict = dict([[fn, i] for i, fn in enumerate(layer_field_names)])
        return [layer_field_names_dict[fn] if fn in layer_field_names else -1 for fn in field_names]

    def get_field_names(self, layer_number=0):
        ld = self.get_layer_definition(layer_number)
        result = []
        for i in range(ld.GetFieldCount()):
            fd = ld.GetFieldDefn(i)
            result.append(fd.GetName())
        return result

    def get_field(self, field_name, layer_number=0):
        """Return [field name, type code, type, width, precision]

        """
        if field_name:
            ld = self.get_layer_definition(layer_number)
            for i in range(ld.GetFieldCount()):
                fd = ld.GetFieldDefn(i)
                if field_name == fd.GetName():
                    t = fd.GetType()
                    return [fd.GetName(), t, fd.GetFieldTypeName(t), fd.GetWidth(), fd.GetPrecision()]
        return None

    def get_field_values(self, field_names=None, layer_number=0):
        """Return field values

        """
        if isinstance(field_names, basestring):
            field_names = [field_names]
        result = []
        ly = self.get_layer(layer_number)
        if not field_names:
            field_names = self.get_field_names(layer_number)
        field_numbers = self.get_field_numbers(field_names, layer_number)
        ly.ResetReading()
        for feat in ly:
            result.append([feat.GetField(fn) for fn in field_numbers])
        ly.ResetReading()
        return result

    def rename_fields(self, shp_out, fields, layer_number=0):
        """
        Args:
            fields (list): list of ['old_name', 'new_name']

        see: http://darrencope.com/2011/04/26/renaming-fields-in-a-shapefile/
        """
        sql = 'SELECT '
        for i, field in enumerate(fields):
            sql += field[0] + ' AS ' + field[1]
            if i > 0:
                sql += ', '
        sql += ' FROM ' + os.path.splitext(os.path.basename(self.get_source()))[0]  # + '"'
        subprocess.check_output(['ogr2ogr', shp_out, self.get_source(), '-sql', sql])
        # ogr2ogr.main(['', shp_out, shp_in, '-sql', sql])

    def get_geometry_and_field_values(self, field_names=None, layer_number=0, geometry_format='wkb'):
        """
        Returns a list of lists. The first element of each sublist is the geometry in well known binary
        format, the remaining field values are as given by field_names. If field_names is None, all field
        values are returned
        :param field_names:
        :param layer_number:
        :param geometry_format:
        :return:
        """
        result = []
        ly = self.get_layer(layer_number)
        if not field_names:
            field_names = self.get_field_names(layer_number)
        ly.ResetReading()
        if geometry_format.lower() == 'wkt':
            for feat in ly:
                result.append([feat.GetGeometryRef().ExportToWkt()] + [feat.GetField(fn) for fn in field_names])
            return result
        else:
            for feat in ly:
                result.append([feat.GetGeometryRef().ExportToWkb()] + [feat.GetField(fn) for fn in field_names])
            return result

    def get_geometries(self, layer_number=0, geometry_format='wkb'):
        result = []
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        if geometry_format.lower() == 'wkt':
            for feat in ly:
                result.append(feat.GetGeometryRef().ExportToWkt())
            return result
        else:
            for feat in ly:
                result.append(feat.GetGeometryRef().ExportToWkb())
            return result

    def get_geometry_type(self, layer_number=0):
        return self.get_layer(layer_number).GetGeomType()

    def is_geometry_point(self, layer_number=0):
        return is_geometry_point(self.get_geometry_type(layer_number))

    def is_geometry_polygon(self, layer_number=0):
        return is_geometry_polygon(self.get_geometry_type(layer_number))

    def get_extent(self, layer_number=0, scale=0.0):
        """Calculate the bounding box of all shapes
        Args:
            the filename of the shapefile
        Return:
            (xmin, xmax, ymin, ymax) as floats

        """
        xmin, xmax, ymin, ymax = self.get_layer(layer_number).GetExtent()
        if scale != 0.0:
            dx, dy = xmax - xmin, ymax - ymin
            xmin -= dx * scale
            xmax += dx * scale
            ymin -= dy * scale
            ymax += dy * scale
        return xmin, xmax, ymin, ymax

    def get_geometry_union_as_wkb(self, layer_number=0):
        ly = self.get_layer(layer_number)
        union = None
        ly.ResetReading()
        for feat in ly:
            geom = feat.GetGeometryRef().ExportToWkb()
            if geom:
                union = ogr.CreateGeometryFromWkb(geom) if not union else union.Union(ogr.CreateGeometryFromWkb(geom))
        return union.ExportToWkb()

    def get_feature_count(self, layer_number=0):
        return self.get_layer(layer_number).GetFeatureCount()

    def create_feature(self, **kwargs):
        lyr = self.get_layer(layer_number=kwargs.pop('layer_number', 0))
        feature = ogr.Feature(lyr.GetLayerDefn())
        geom = kwargs.pop('geom', None)
        try:
            feature.SetGeometry(geom)
        except TypeError:
            try:
                feature.SetGeometry(ogr.CreateGeometryFromWkb(geom))
            except RuntimeError:
                feature.SetGeometry(ogr.CreateGeometryFromWkt(geom))
        for k, v in kwargs.items():
            feature.SetField(k, v)
        lyr.CreateFeature(feature)
        del feature

    def set_spatial_filter(self, lrs, layer_number=0):
        try:
            lrs = LayersReader(lrs)
            geometries = lrs.get_geometries(layer_number)
        except (TypeError, RuntimeError):
            try:
                geometries = lrs.get_geometries(layer_number)
            except (TypeError, RuntimeError):
                geometries = lrs
        geom = merge_geometries(geometries)
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        g = ogr.CreateGeometryFromWkb(geom)
        ly.SetSpatialFilter(g)

    def spatial_filter(self, lrs, layer_number=0, method='Intersects'):
        ly = self.get_layer(layer_number)
        if method == 'Disjoint':
            instc_set = set([feat.GetFID() for feat in self.spatial_filter(lrs, layer_number, method='Intersects')])
            ly.ResetReading()
            for feat in ly:
                if feat.GetFID() not in instc_set:
                    yield feat
        else:
            geometries = lrs.get_geometries(layer_number)
            geometries = [ogr.CreateGeometryFromWkb(g) for g in geometries]
            envelopes = [g.GetEnvelope() for g in geometries]
            ly.ResetReading()
            for feat in ly:
                geom0 = feat.GetGeometryRef()
                e0 = geom0.GetEnvelope()
                for i, e1 in enumerate(envelopes):
                    if not (e0[0] > e1[1] or e0[1] < e1[0] or e0[2] > e1[3] or e0[3] < e1[2]):
                        if getattr(geom0, method)(geometries[i]):
                            yield feat
                            feat = None
                            break

    def spatial_filter_to_layer(self, lrs, layer_number=0, method='Intersects', output_layers=''):
        fids = set([feat.GetFID() for feat in self.spatial_filter(lrs, layer_number, method)])
        return self.copy(output_layers, ['FID', fids, lambda v, values: v in values])

    def set_attribute_filter(self, filter_string, layer_number=0):
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        ly.SetAttributeFilter(filter_string)

    def centroid(self, **kwargs):
        """Return a LayersWriter with the centroids of the geometries in this LayersSet
        :param kwargs:
        :return:
        """
        lrs_point = LayersWriter(**kwargs)
        for i in range(self.get_layer_count()):
            lyr_self = self.get_layer(i)
            ld_self = self.get_layer_definition()
            field_definitions = []
            for ifd in range(ld_self.GetFieldCount()):
                field_definitions.append(ld_self.GetFieldDefn(ifd))
            lyr_point = lrs_point.create_layer(lyr_self.GetName(), prj=lyr_self.GetSpatialRef(), geom=ogr.wkbPoint,
                                               field_definitions=field_definitions)
            feature_def = lyr_point.GetLayerDefn()
            lyr_self.ResetReading()
            for feat in lyr_self:
                feature = ogr.Feature(feature_def)
                for idx in range(len(field_definitions)):
                    feature.SetField(idx, feat.GetField(idx))
                g = feat.GetGeometryRef().Centroid()
                feature.SetGeometry(g)
                lyr_point.CreateFeature(feature)
        return lrs_point

    def get_convex_hull_as_wkb(self, layer_number=0):
        geom_col = ogr.Geometry(ogr.wkbGeometryCollection)
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        for feat in ly:
            geom_col.AddGeometry(feat.GetGeometryRef())

        # Calculate convex hull
        return geom_col.ConvexHull().ExportToWkb()

    def convex_hull(self, target):
        assure_path_exists(target)
        lrs = LayersWriter(target)
        for ilc in range(self.dataset.GetLayerCount()):
            ly_in = self.get_layer(ilc)
            geom = self.get_convex_hull_as_wkb(ilc)
            id_field = ogr.FieldDefn('id', ogr.OFTInteger)
            ly_out = lrs.create_layer(str(ilc), ly_in.GetSpatialRef(), ly_in.GetGeomType())
            ly_out.CreateField(id_field)
            feature_def = ly_out.GetLayerDefn()
            feature = ogr.Feature(feature_def)
            feature.SetGeometry(ogr.CreateGeometryFromWkb(geom))
            feature.SetField("id", 1)
            ly_out.CreateFeature(feature)
            ly_out.SyncToDisk()
        return lrs

    def join(self, fieldname, target, target_fieldname, output_filename=None, nan=0.0, layer_number=0):
        """
        join the feature via source_fieldname with the pandas data frame using index
        :param source:
        :param source_fieldname:
        :param df:
        :return:
        """
        df_source = ogrpandas.LayerDataFrame(ogr_layer=self.get_layer(layer_number))
        try:
            lrs = LayersReader(target)
            df_target = ogrpandas.LayerDataFrame(ogr_layer=lrs.get_layer(layer_number))
        except RuntimeError:
            df_target = target
        df_source = df_source.set_index(fieldname)
        df_target = df_target.set_index(target_fieldname)
        df_source = pd.concat([df_source, df_target], axis=1).reset_index()
        df_source = df_source.fillna(nan)
        delete_layers(output_filename)
        d = os.path.dirname(output_filename)
        if not os.path.exists(d):
            os.makedirs(d)
        try:
            dataset = LayersWriter(output_filename).dataset
        except TypeError:
            dataset = output_filename
        df_source.to_layer(dataset)
        return df_source

    def create_buffer(self, buffer_filename, buffer_dist, layer_number=0):
        lyr_in = self.get_layer(layer_number)
        lrs_out = LayersWriter(buffer_filename)
        lyr_out = lrs_out.create_layer(str(layer_number), lyr_in.GetSpatialRef(), geom=ogr.wkbPolygon)
        fd_out = lyr_out.GetLayerDefn()
        lyr_in.ResetReading()
        for feat_in in lyr_in:
            geom_in = feat_in.GetGeometryRef()
            geom_out = geom_in.Buffer(buffer_dist)
            feat_out = ogr.Feature(fd_out)
            feat_out.SetGeometry(geom_out)
            lyr_out.CreateFeature(feat_out)
            feat_out = None
        return lrs_out

    def dissolve(self, **kwargs):
        import geoprocess
        return geoprocess.dissolve(self, **kwargs)


class LayersReader(LayersSet):

    def __init__(self, source):
        super(LayersReader, self).__init__()
        self.dataset = ogr.Open(source, 0)


class LayersEditor(LayersSet):

    __metaclass__ = ABCMeta

    def __init__(self):
        super(LayersEditor, self).__init__()

    def create_layer(self, name, prj, geom, field_definitions=None):
        ly = self.dataset.CreateLayer(name, prj, geom)
        if not ly:
            raise Exception('Layer {} with projection {} and geometry {} could not be created'.format(name, prj, geom))
        if field_definitions:
            for fd in field_definitions:
                ly.CreateField(fd)
        return ly

    def field_calculator(self, calc_field_name, field_names, calc_function, layer_number=0):
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        for feat in ly:
            feat.SetField(calc_field_name, calc_function([feat.GetField(fn) for fn in field_names]))
            ly.SetFeature(feat)

    def calculate_field_area(self, field_name='area', layer_number=0, scale=1.0):
        if self.is_geometry_polygon(layer_number):
            ld = self.get_layer_definition(layer_number)
            fidx = ld.GetFieldIndex(field_name)
            if fidx > -1:
                ly = self.get_layer(layer_number)
                ly.ResetReading()
                for feat in ly:
                    area = feat.GetGeometryRef().GetArea() * scale
                    feat.SetField(fidx, area)
                    ly.SetFeature(feat)
                    del feat
                ly.SyncToDisk()

    def add_fields(self, fields, layer_number=0):
        ly = self.get_layer(layer_number)
        for field in fields:
            f_len = len(field)
            fd = ogr.FieldDefn(field[0], field[1])
            if f_len > 2:
                fd.SetWidth(field[2])
            if f_len > 3:
                fd.SetPrecision(field[3])
            ly.CreateField(fd)

    def delete_fields(self, fields, layer_number=0):
        ly = self.get_layer(layer_number)
        ld = self.get_layer_definition(layer_number)
        for field in fields:
            field_index = ld.GetFieldIndex(field)
            if field_index > -1:
                ly.DeleteField(field_index)


class LayersModifier(LayersEditor):

    def __init__(self, source):
        super(LayersModifier, self).__init__()
        try:
            source = source.get_source()
        except AttributeError:
            pass
        self.dataset = ogr.Open(source, 1)


class LayersWriter(LayersEditor):

    def __init__(self, source='', drivername=None, **kwargs):
        r"""To use 'Memory' driver, \**kwargs has to have neither 'drivername' nor 'source'
        :param \**kwargs: See below

        :Keyword Arguments:
            * drivername (``str``): a driver name (see :meth:`FeatDrivers.get_driver_names()`)
            * extension (``str``): file name extension with or without dot (.)
            * filename (``str``): file name with extension (basename or full path)
            * all further arguments are layer names. For example:
                * a_name =  [wkb-geometry0, srs0, fields0]
                * another_name = [wkb-geometry1, srs1, fields0]
        """
        super(LayersWriter, self).__init__()
        if source is None:
            source = ''
        dr = FeatDrivers.get_driver(source, drivername)
        self.dataset = dr.CreateDataSource(source)
        if self.dataset is None:
            if os.path.exists(source):
                raise ValueError("Data source {} already exists".format(str(source)))
            else:
                raise ValueError("Data source {} could not be created".format(str(source)))
        for key in kwargs.keys():
            parameters = kwargs[key]
            n = len(parameters)
            geom = parameters[0]
            prj = parameters[1] if n > 1 else None
            field_definitions = parameters[2] if n > 2 else None
            self.create_layer(key, prj, geom, field_definitions)


def delete_layers(filename):
    if os.path.exists(filename):
        driver = FeatDrivers.get_driver(source=filename)
        driver.DeleteDataSource(filename)


def data_frame_to_layer(df, ld_in, ly_out):

    field_names = df.columns.tolist()
    del field_names[0]
    print field_names

    # def create_feature(lyr_out, fd_out, feat_in, indices_in):
    #     feat_new = ogr.Feature(fd_out)
    #     for idx_out, idx_in in enumerate(indices_in):
    #         feat_new.SetField(idx_out, feat_in.GetField(idx_in))
    #     g = feat_in.GetGeometryRef()
    #     feat_new.SetGeometry(g)
    #     lyr_out.CreateFeature(feat_new)
    #
    # # ld_in = ly_in.GetLayerDefn()
    # # if field_names and ilc < len(field_names) and field_names[ilc]:
    # #     fields_indices_keep = sorted([ld_in.GetFieldIndex(fn) for fn in set(field_names[ilc])])
    # # else:
    # #     fields_indices_keep = range(ld_in.GetFieldCount())
    # fields_indices_keep = range(ld_in.GetFieldCount())
    # fd_out = [ld_in.GetFieldDefn(ifd) for ifd in fields_indices_keep]
    # ld_out = ly_out.GetLayerDefn()
    # # copy features
    # for feat in ly_in:
    #     create_feature(ly_out, ld_out, feat, fields_indices_keep)
    #     feat = None
    #
    # if field_names and ilc < len(field_names) and field_names[ilc]:
    #     ld_out = ly_out.GetLayerDefn()
    #     fields_all = set([ld_in.GetFieldDefn(i).GetName() for i in range(ld_in.GetFieldCount())])
    #     fields_keep = set(field_names[ilc])
    #     fields_delete = fields_all - fields_keep
    #     indices = sorted([ld_out.GetFieldIndex(fn) for fn in fields_delete], reverse=True)
    #     for idx in indices:
    #         if idx >= 0:
    #             ly_out.DeleteField(idx)


def merge_geometries(wkb_list):
    geom_col = ogr.Geometry(ogr.wkbGeometryCollection)
    for g_wkb in wkb_list:
        geom = ogr.CreateGeometryFromWkb(g_wkb)
        geotype = geom.GetGeometryType()
        if geotype == ogr.wkbMultiPoint or geotype == ogr.wkbMultiLineString or geotype == ogr.wkbMultiPolygon:
            for g_part in geom:
                found = True
                geom_col.AddGeometry(g_part)
        else:
            geom_col.AddGeometry(geom)
    return geom_col.ExportToWkb()


def get_field_definitions(layer_definition):
    for i in range(layer_definition.GetFieldCount()):
        yield layer_definition.GetFieldDefn(i)


def is_geometry_point(geotype):
    return geotype == ogr.wkbPoint or geotype == ogr.wkbPoint25D or \
           geotype == ogr.wkbMultiPoint or geotype == ogr.wkbMultiPoint25D


def is_geometry_polygon(geotype):
    return geotype == ogr.wkbPolygon or geotype == ogr.wkbPolygon25D or \
           geotype == ogr.wkbMultiPolygon or geotype == ogr.wkbMultiPolygon25D


def intersection(source0, source1, target):
    lys0 = LayersReader(source0)
    lsy1 = LayersReader(source1)
    lys2 = LayersWriter(target)
    ly0 = lys0.get_layer()
    ly1 = lsy1.get_layer()
    lyt = lys2.create_layer(str(0), prj=ly0.GetSpatialRef(), geom=ly0.GetGeomType())
    lyt = ly0.Intersection(ly1, lyt)
    return lyt


def merge(layers, output_filename=None, layers_numbers=None, field_names=None):
    """

    :param layers: a list of ogr layers, ogr datasets, or filenames
    :param output_filename:
    :param layers_numbers: list of layers numbers used for layers ogr-dataset or filename.
        If layers_numbers = None, the first layer will be used (layers_number=0). If the length of layers_numbers is
        less then the length of layers, the list layers_numbers will be completed with zeros.
    :return:
    """
    n_layers = len(layers)
    if not layers_numbers:
        layers_numbers = [0] * n_layers
    elif len(layers_numbers) < n_layers:
        layers_numbers = layers_numbers + [0] * (n_layers - len(layers_numbers))
    odf_list = []
    for i in range(n_layers):
        lrs = layers[i]
        layer_number = layers_numbers[i]
        try:
            lrs = LayersReader(lrs)
            layer = lrs.get_layer(layer_number)
        except TypeError:
            try:
                layer = lrs.GetLayer(layer_number)
            except TypeError:
                layer = lrs
        ldf = ogrpandas.LayerDataFrame(ogr_layer=layer)
        odf_list.append(ldf)
    odf = pd.concat(odf_list, ignore_index=True)
    try:
        lrs_out = LayersWriter(output_filename)
    except TypeError:
        lrs_out = output_filename
    dataset = lrs_out.dataset
    odf.to_layer(dataset, field_names=field_names)
    return lrs_out


# def dissolve2(source, target, fields=None):
#     lys_in = LayersReader(source)
#     lys_out = LayersWriter(target)
#
#     for ly_in in lys_in.layers():
#         ld_in = ly_in.GetLayerDefn()
#         ly_out = lys_out.create_layer('a', prj=ly_in.GetSpatialRef(), geom=ly_in.GetGeomType())
#         # get field indices and definitions in the same order as input fields
#         # fd_list1 = [fd for fd in fd_list0 if fd[1].GetName() not in fields]
#         # if not fields:
#         #     id_field = ogr.FieldDefn('OID', ogr.OFTInteger64)
#         #     ly_out.CreateField(id_field)
#         #     geom_wkb_list = []
#         #     for feat in ly_in:
#         #         geom = feat.GetGeometryRef()
#         #         if geom:
#         #             geom_wkb_list.append(geom.ExportToWkb())
#         #     geom = ogr.CreateGeometryFromWkb(unary_union([loads(g) for g in geom_wkb_list]).to_wkb())
#         #     ld_out = ly_out.GetLayerDefn()
#         #     feature = ogr.Feature(ld_out)
#         #     feature.SetGeometry(geom)
#         #     feature.SetField(0, 0)
#         #     ly_out.CreateFeature(feature)
#         #     del feature
#         # else:
#
#         if fields:
#             fd_list0 = [(i, fd) for i, fd in enumerate(get_field_definitions(ld_in))]
#             fd_list0 = [fd for fd in fd_list0 if fd[1].GetName() in fields]
#             fd_list0 = [fd for (f, fd) in sorted(zip(fields, fd_list0), key=lambda x: x[0])]
#         else:
#             fd_list0 = []
#
#         ly_dict = {}
#         for feat in ly_in:
#             geom = feat.GetGeometryRef()
#             if geom is None:
#                 continue
#             geom_wkb = geom.ExportToWkb()
#             fd = tuple([feat.GetField(fid[0]) for fid in fd_list0])
#             if fd not in ly_dict:
#                 ly_dict[fd] = []
#             ly_dict[fd].append(geom_wkb)
#
#         for fd in ly_dict.keys():
#             ly_dict[fd] = ogr.CreateGeometryFromWkb(unary_union([loads(g) for g in ly_dict[fd]]).to_wkb())
#
#         # Add an ID field
#         for fd in fd_list0:
#             ly_out.CreateField(fd[1])
#
#         ld_out = ly_out.GetLayerDefn()
#         for fd, geom, in ly_dict.items():
#             # Create the features and set values
#             feature = ogr.Feature(ld_out)
#             feature.SetGeometry(geom)
#             for idx, value in enumerate(fd):
#                 feature.SetField(idx, value)
#             ly_out.CreateFeature(feature)

# def dissolve_slow(source, target, fields):
#     lys_in = LayersReader(source)
#     lys_out = LayersWriter(target)
#
#     for ly_in in lys_in.get_layers():
#         ld_in = ly_in.GetLayerDefn()
#         # get field definitions in the same order as fields; get also field indices
#         fd_list = []
#         fd_list0 = [fd for fd in get_field_definitions(ld_in)]
#         field_indices = []
#         for field in fields:
#             for fd in fd_list0:
#                 if field == fd.GetName():
#                     field_indices.append(ld_in.GetFieldIndex(field))
#                     fd_list.append(fd)
#                     break
#
#         ly_dict = {}
#         for feat in ly_in:
#             geom = feat.GetGeometryRef()
#             if geom:
#                 geom_wkb = ogr.CreateGeometryFromWkb(geom.ExportToWkb())
#             else:
#                 geom_wkb = None
#
#             feat_values = ()
#             for fid in field_indices:
#                 feat_values += (feat.GetField(fid),)
#
#             if feat_values not in ly_dict:
#                 ly_dict[feat_values] = geom_wkb
#             else:
#                 ly_dict[feat_values] = ly_dict[feat_values].Union(geom_wkb)
#
#         ly_out = lys_out.create_layer(target, prj=ly_in.GetSpatialRef(), geom=ly_in.GetGeomType())
#         # Add an ID field
#         for field_definition in fd_list:
#             ly_out.CreateField(field_definition)
#
#         ld_out = ly_out.GetLayerDefn()
#         for feat_values, geom, in ly_dict.items():
#             if geom:
#                 # Create the features and set values
#                 feature = ogr.Feature(ld_out)
#                 feature.SetGeometry(geom)
#                 for idx, value in enumerate(feat_values):
#                     feature.SetField(idx, value)
#                 ly_out.CreateFeature(feature)


def thiessen(points, thiessen, points_layer_number=0, point_id_field_names=None, envelope=None,
             clip_region=None, clip_region_layer_number=0, expand=0.1):
    """Calculate the Voronoi diagram (or Thiessen polygons) with the given points.

    If both envelope and polygon_shp are given, envelope will be discarded

    Args:
        points (string): shapefile with the points, for which the Thiessen polygons are created

        thiessen (string): shapefile with polygons, and with coordinate system and attributes from point_shp

        point_id_field_names (list): list of field names to be assigned in thiessen_shp

        envelope (list): [minX, maxX, minY, maxY] envelope used to clip the thiessen polygons

        clip_region (string): shapefile with the polygon(s) used to clip the thiessen polygons
    """

    def merge_extent(ext0, ext1=None, point=None):
        if not ext1:
            ext1 = [point[0], point[1], point[0], point[1]] if point else ext0
        return [min(ext0[0], ext1[0]), max(ext0[1], ext1[1]), min(ext0[2], ext1[2]), max(ext0[3], ext1[3])]

    # Check given point field name is not valid
    lrs_thiessen = LayersWriter(thiessen)

    lrs_point = LayersReader(points)
    lyr_point = lrs_point.get_layer(points_layer_number)
    ext_point = lyr_point.GetExtent()

    # Get the clip polygon, The clip object can be 1) the given polygon, 2) the give bounding box (envelope) or
    # 3) an envelope expanding 10% of the points envelope
    if clip_region:
        import geoprocess as gp
        try:
            clip_region = LayersReader(clip_region)
        except RuntimeError:
            pass
        clip_geometry_wkb = gp.union_cascade(clip_region.get_geometries(clip_region_layer_number), 2)
    else:
        if not envelope:
            dx = (ext_point[1]-ext_point[0]) * expand
            dy = (ext_point[3]-ext_point[2]) * expand
            envelope = [ext_point[0] - dx, ext_point[1] + dx, ext_point[2] - dy, ext_point[3] + dy]
        poly = 'POLYGON((' + '{} {}, {} {}, {} {}, {} {}, {} {}))'.format(
            envelope[0], envelope[2], envelope[1], envelope[2],
            envelope[1], envelope[3], envelope[0], envelope[3], envelope[0], envelope[2])
        clip_geometry_wkb = ogr.CreateGeometryFromWkt(poly).ExportToWkb()

    clip_geometry = ogr.CreateGeometryFromWkb(clip_geometry_wkb)

    # Merge the envelope of the given points with the given envelope or polygons envelope
    merge_extent(ext_point, clip_geometry.GetEnvelope())

    if point_id_field_names:
        point_id_field_names = list(set(f[0] for f in lrs_point.get_fields()) - set(point_id_field_names))

    points_wkb = [[point_feat.GetGeometryRef().GetX(), point_feat.GetGeometryRef().GetY()] for point_feat in lyr_point]

    # Append four points distant enough from the envelope, in this case three times the with/height of the envelope
    dx = 3.0 * (ext_point[1]-ext_point[0])
    dy = 3.0 * (ext_point[3]-ext_point[2])
    points_wkb.append([ext_point[0]-dx, ext_point[2]-dy])
    points_wkb.append([ext_point[0]-dx, ext_point[3]+dy])
    points_wkb.append([ext_point[1]+dx, ext_point[3]+dy])
    points_wkb.append([ext_point[1]+dx, ext_point[2]-dy])

    # Calculate the Voronoi diagram
    vor = Voronoi(points_wkb)

    # Remove the four last point regions, since they are related to the appended four points.
    vor_regions = vor.regions
    vor_vertices = vor.vertices
    vor_point_region = vor.point_region.tolist()[-4:]
    # Check if the four point regions have a negative vertex index
    for region_id in vor_point_region:
        if not any(n < 0 for n in vor_regions[region_id]):
            print 'ERROR Thiessen'
    vor_point_region = vor.point_region # .tolist()[:-4]
    vor_polygons = []
    for region_id in vor_point_region:
        poly = ogr.Geometry(ogr.wkbPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        points = [vor_vertices[reg] for reg in vor_regions[region_id]]
        for p in points:
            ring.AddPoint(p[0], p[1])
        ring.AddPoint(points[0][0], points[0][1])  # close the polygon
        poly.AddGeometry(ring)
        vor_polygons.append(poly.ExportToIsoWkb())

    thiessen_polygons = [clip_geometry.Intersection(ogr.CreateGeometryFromWkb(wkb)).ExportToWkb() for wkb in vor_polygons]

    lyr_thiessen = lrs_thiessen.create_layer(lyr_point.GetName(), prj=lyr_point.GetSpatialRef(), geom=ogr.wkbPolygon)
    lyr_def_point = lyr_point.GetLayerDefn()
    for ifd in range(lyr_def_point.GetFieldCount()):
        lyr_thiessen.CreateField(lyr_def_point.GetFieldDefn(ifd))
    lyr_point.ResetReading()
    for ift, feat in enumerate(lyr_point):
        if thiessen_polygons[ift]:
            feat.SetGeometry(ogr.CreateGeometryFromWkb(thiessen_polygons[ift]))
            lyr_thiessen.CreateFeature(feat)
        del feat

    del lrs_thiessen  # close the datasource

    if point_id_field_names:
        lrs_thiessen = LayersModifier(thiessen)
        for layer_number in range(lrs_thiessen.get_layer_count()):
            lrs_thiessen.delete_fields(point_id_field_names, layer_number)
        del lrs_thiessen


def get_thiessen_weights(point_shp, point_shp_id, polygon_shp, polygon_shp_id,
                         valid_points=None, work_directory=None):
    """
    Thiessen polygons will be created with the given points and intersected
    with given polygons. The intersection areas divided by they corresponding
    polygon areas result the weights. The sum of the weights of a polygon is
    one.

    This procedure will return a dictionary, where the key is the
    polygon_feat_code_field_name and the value is a list of dictionaries, which has a
    tuple(date_from, date_to) as key and a list of tuples
    (precip_feat_code_field_name, weight) as values. Depending on the input parameter
    valid_points, the key tuple may be (None, None).

    This procedure can be used to determine for example the contribution of
    individual precipitation points in areas (e.g. watersheds) with different
    patterns of valid points for each time period (e.g. year)

    Example 1: no time series for valid points
    dict[polygon1] = dict1
        dict1[(None,None)] = [(point1, 0.4), (point2, 0.4), (point3, 0.2)]
    dict[polygon2] = dict2
        dict2[(None,None)] = [(point1, 0.4), (point2, 0.4), (point3, 0.2)]

    Example 2:
    dict[polygon1] = dict1
        dict1[(01.10.2000,30.09.2001)] = [(point1, 0.4), (point2, 0.4), (point3, 0.2)]
        dict1[(01.10.2001,30.09.2002)] = [(point1, 0.5), (point3, 0.5)]
        dict1[(01.10.2002,30.09.2003)] = [(point1, 0.4), (point2, 0.6)]
    dict1[polygon2] = dict2
        dict2[(01.10.2000,30.09.2001)] = [(point1, 0.4), (point3, 0.5), (point4, 0.1)]
        dict2[(01.10.2001,30.09.2002)] = [(point1, 0.45),(point3, 0.55)]
        dict2[(01.10.2002,30.09.2003)] = [(point1, 0.8), (point5, 0.2)]

    The procedure includes the following steps:

    1)  determination Thiessen polygons

    2)  intersection Thiessen polygons with areas in polygon_shp

    3)  determination of the weights of the intersections for each area. The
        weights of each area must sum up to one.

    point_shp: full qualified name of the features used to create Thiessen
        polygons, typically precipitation or meteorological points. The
        geometry must be point

    point_shp_id: name of the field containing unique record
        values. Instead of using the features ID (FID), it is recommended a
        user defined field. The field type must be number or string

    polygon_shp: full qualified name of the features to be intersected with
        the Thiessen polygon to calculate the weights by area

    polygon_shp_id: name of the field containing unique record values.
        Instead of using the features ID (FID), it is recommended a user
        defined field. The field type must be number or string

    valid_points: Dictionary with tuple(datetime, datetime) as key and a list of field
        values for point_shp_id representing valid records as values. The
        tuple(datetime, datetime) is intended to be set the validity of the
        results as tuple(date_from, date_to). If valid_points==None, all points
        will be used to create the Thiessen polygon, else only the points from
        the list. tuple(None, None) is also valid.
    """

    lys_point = LayersReader(point_shp)
    lys_polygon = LayersReader(polygon_shp)

    # ==================== create field names =================================
    # Check t_area_field and a_area_field, rename them if necessary
    t_area_field = lys_point.generate_field_name('TAREA')  # polygon-thiessen intersection area
    a_area_field = lys_polygon.generate_field_name('AAREA')  # polygon area

    # ==================== temporary files and host_directory ======================
    # Create a new temporary work host_directory
    # work_directory: full qualified host_directory name used for temporary files.
    #                 Inside work_directory a new host_directory for temporary files
    #                 will be created. work_directory will be deleted at the
    #                 end of the procedure
    if not work_directory:
        work_directory = create_tmp_directory(os.path.dirname(point_shp))
    timestamp_shp = str(int(time.time())) + '.shp'
    # Names of temporary files
    tmp_point0_shp = work_directory + 'points_' + timestamp_shp
    tmp_point1_shp = work_directory + 'valid_points_' + timestamp_shp
    tmp_polygon_shp = work_directory + 'polygons_' + timestamp_shp
    tmp_thiessen0 = work_directory + 'thiessen0_' + timestamp_shp
    tmp_thiessen1 = work_directory + 'thiessen1_' + timestamp_shp
    tmp_thiessen2 = work_directory + 'thiessen2_' + timestamp_shp

    # ==================== check geometries and ids ===========================
    point_id_field = lys_point.get_field(point_shp_id)
    t = point_id_field[1] if point_id_field else None

    # check whether point_shp_id exists in point_shp and get field type
    if t is None or (t != ogr.OFTInteger and t != ogr.OFTReal and t != ogr.OFTString and t != ogr.OFTWideString):
        remove_directory(work_directory)
        raise Exception('Field ' + point_shp_id + ' is not defined')

    # check whether polygon_shp_id exists in polygon_shp
    if not lys_polygon.has_field(polygon_shp_id):
        remove_directory(work_directory)
        raise Exception('Field ' + polygon_shp_id + ' is not defined in ' + polygon_shp)

    # check polygon geometry type
    if not lys_polygon.is_geometry_polygon():
        remove_directory(work_directory)
        raise Exception(polygon_shp + ' must have polygon geometry')

    # check point geometry type
    if not lys_point.is_geometry_point():
        remove_directory(work_directory)
        raise Exception(point_shp + ' must have point geometry')

    # ==================== check valid points =================================
    if valid_points is None:
        valid_points = dict()
        valid_points[(None, None)] = None
    elif isinstance(valid_points, list):
        valid_points[(None, None)] = valid_points
    elif not isinstance(valid_points, dict):
        raise Exception('valid_points wrong type: ' + str(type(valid_points)))
    elif len(valid_points) == 0:
        valid_points[(None, None)] = None

    # ========== copy polygon_shp, create new and delete unused fields ========
    # copy polygon
    lys_polygon.copy(tmp_polygon_shp)
    lys_polygon.destroy()

    polygon_shp = tmp_polygon_shp
    lys_polygon = LayersModifier(polygon_shp)

    # create a_area_field and calculate the area
    lys_polygon.add_fields([[a_area_field, ogr.OFTReal, 18]])
    lys_polygon.calculate_field_area(a_area_field)

    # delete all fields from polygon_shp but polygon_shp_id and a_area_field
    fields = list(set([f[0] for f in lys_polygon.get_fields()])-set([a_area_field, polygon_shp_id]))
    lys_polygon.delete_fields(fields)

    # ========== copy point_shp, create new and delete unused fields ========
    # copy point
    lys_point.copy(tmp_point0_shp)
    lys_point.destroy()

    lys_point = LayersModifier(tmp_point0_shp)

    # create t_area_field
    lys_point.add_fields([[t_area_field, ogr.OFTReal, 18]])

    if point_shp_id == polygon_shp_id:
        # rename the field and create a new shapefile
        point_shp_id += '_t'
        lys_point.rename_fields(tmp_point0_shp, [[]])

    # delete all fields from point_shp but point_shp_id and t_area_field
    fields = list(set([f[0] for f in lys_point.get_fields()])-set([point_shp_id, t_area_field]))
    lys_point.delete_fields(fields)

    # =========================================================================
    polygon_weights = {}  # polygons

    def point_filter(value, values):
        return value in values

    keys = sorted(valid_points.keys(), key=operator.itemgetter(0))
    for k in keys:
        date_key = k[0], k[1]
        print 'calculating', date_key

        # copy point_shp if valid points was defined
        if valid_points[k]:
            lys_point.copy(tmp_point1_shp, [point_shp_id, [str(p) for p in valid_points[k]], point_filter])
        else:
            tmp_point1_shp = tmp_point0_shp

        # create Thiessen polygons, clip to polygon_shp
        thiessen(tmp_point1_shp, tmp_thiessen0, [point_shp_id, t_area_field], clip_region=polygon_shp)

        # intersect
        intersection(tmp_thiessen0, polygon_shp, tmp_thiessen1)

        # dissolve Thiessen polygons
        fields = [point_shp_id, polygon_shp_id, a_area_field, t_area_field]
        geoprocess.dissolve(tmp_thiessen1, tmp_thiessen2, fields)

        # calculate Thiessen polygon areas
        lys_thiessen = LayersModifier(tmp_thiessen2)
        lys_thiessen.calculate_field_area(t_area_field)

        # calculate polygons weights
        for ly in lys_thiessen.layers():
            ly.ResetReading()
            for feat in ly:
                a_area = feat.GetField(a_area_field)  # polygon area
                t_area = feat.GetField(t_area_field)  # intersection area
                a_code = feat.GetField(polygon_shp_id)  # polygon id
                t_code = feat.GetField(point_shp_id)  # point id
                weight = t_area / a_area

                if a_code not in polygon_weights:
                    polygon_weights[a_code] = {}

                date_tuples = polygon_weights[a_code]

                if date_key not in date_tuples:
                    date_tuples[date_key] = []

                date_tuples[date_key].append([t_code, weight])

        # =================  delete temporary shapefiles ======================
        del lys_thiessen  # if not deleted, tmp_thiessen2 cannot be deleted
        delete_layers(tmp_point1_shp)
        delete_layers(tmp_thiessen0)
        delete_layers(tmp_thiessen1)
        delete_layers(tmp_thiessen2)

    # delete temporary shapefiles from outside the loop
    del lys_point
    del lys_polygon

    delete_layers(tmp_point0_shp)
    delete_layers(tmp_polygon_shp)

    # delete temporary work host_directory
    remove_directory(work_directory)

    return polygon_weights
