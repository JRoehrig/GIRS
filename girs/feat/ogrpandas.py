import numpy as np
import pandas as pd
from osgeo import osr, ogr
# pd.set_option("display.width",500)
# pd.set_option("display.max_rows",50)


def ogr_to_dtype(t):
    if t == ogr.OFTInteger:
        return np.int64
    elif t == ogr.OFTReal:
        return np.float64
    else:
        return object


class LayerDataFrame(pd.DataFrame):

    _metadata = ['name', 'srs', 'geom', 'geometry_column_name']

    # enum OGRFieldType
    # List of feature field types.
    # This list is likely to be extended in the future ...
    # avoid coding applications based on the assumption that all field types can be known.
    # Enumerator
    # OFTInteger: Simple 32bit integer.
    # OFTIntegerList:  List of 32bit integers.
    # OFTReal:  Double Precision floating point.
    # OFTRealList:  List of doubles.
    # OFTString:  String of ASCII chars.
    # OFTStringList:  Array of strings.
    # OFTBinary:  Raw Binary data.
    # OFTDate:  Date.
    # OFTTime:  Time.
    # OFTDateTime:  Date and Time.
    # OFTInteger64:  Single 64bit integer.
    # OFTInteger64List:  List of 64bit integers.

    def __init__(self, *args, **kwargs):
        ogr_layer = kwargs.pop('ogr_layer', None)
        if not ogr_layer:
            try:
                ogr_dataset = kwargs.pop('ogr_dataset', None)
                layer_number = kwargs.pop('layer_number', 0)
                ogr_layer = ogr_dataset.GetLayer(layer_number)
            except AttributeError:
                pass
        if ogr_layer:
            ld = ogr_layer.GetLayerDefn()
            nf = ld.GetFieldCount()
            fds = [ld.GetFieldDefn(i) for i in range(ld.GetFieldCount())]
            names, types = zip(*[[fd.GetName(), fd.GetType()] for fd in fds])
            names_types = dict(zip(names, types))
            geometry_column_name = 'geom'
            while geometry_column_name in names:
                geometry_column_name += '0'
            columns = [geometry_column_name] + list(names)
            ogr_layer.ResetReading()
            values = [[f.GetGeometryRef().ExportToWkb()] + [f.GetField(i) for i in range(nf)] for f in ogr_layer]
            s_dict = pd.DataFrame(values, columns=columns).to_dict('series')
            names_types[geometry_column_name] = None
            for col, series in s_dict.items():
                t = ogr_to_dtype(names_types[col])
                s_dict[col] = s_dict[col].astype(t)
            super(LayerDataFrame, self).__init__(s_dict, columns=columns, *args, **kwargs)
            # for t in set(types):
            #     cols = [name for i, name in enumerate(names) if t == types[i]]
            #     tp = LayerDataFrame._ogr_to_dtype[t]
            #     self[cols].astype(tp, copy=False)
            # print self.dtypes
            self.geometry_column_name = geometry_column_name
            self.layer_name = ogr_layer.GetName()
            self.srs = ogr_layer.GetSpatialRef().ExportToWkt()
            self.geom = ogr_layer.GetGeomType()
        else:
            super(LayerDataFrame, self).__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return LayerDataFrame

    def __finalize__(self, other, method=None, **kwargs):
        # From pandas.tools.merge:
        #   typ = self.left._constructor
        #   result = typ(result_data).__finalize__(self, method='merge')
        #   result = typ(result_data).__finalize__(self, method='ordered_merge')
        #   return df.__finalize__(self, method='concat')
        if method == 'merge' or method == 'ordered_merge':
            other = other.left
        elif method == 'concat':
            other = other.objs[0]
        for name in self._metadata:
            object.__setattr__(self, name, getattr(other, name, None))
        return self

    def to_layer(self, ogr_dataset, field_names=None, name=''):
        name = self.name if self.name else name
        srs = osr.SpatialReference()
        srs.ImportFromWkt(self.srs)
        geom = self.geom
        ly = ogr_dataset.CreateLayer(name, srs, geom)
        if not ly:
            raise Exception('Layer (name={}) with projection {} and geometry {} could not be created'.
                            format(name, srs, geom))
        if field_names:
            try:
                field_names = {field_names[i]: field_names[i] for i in range(len(field_names))}
            except KeyError:
                pass
        else:
            field_names = {f: f for f in self.columns.tolist() if f != self.geometry_column_name}

        for c in field_names.keys():
            v = field_names[c]
            c, v, dtype = str(c), str(v), self[c].dtype
            if dtype == np.int64:
                fd = ogr.FieldDefn(v, ogr.OFTInteger64)
            elif dtype == np.float64:
                fd = ogr.FieldDefn(v, ogr.OFTReal)
            else:  # dtype == object:
                fd = ogr.FieldDefn(v, ogr.OFTString)
            ly.CreateField(fd)
        ld = ly.GetLayerDefn()
        for index, row in self.iterrows():
            try:
                feature = ogr.Feature(ld)
                for idx, f in enumerate(field_names):
                    feature.SetField(idx, row[f])
                feature.SetGeometry(ogr.CreateGeometryFromWkb(row[self.geometry_column_name]))
                ly.CreateFeature(feature)
            except TypeError, e:
                pass

