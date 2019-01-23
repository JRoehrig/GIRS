from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
from past.builtins import basestring
from builtins import object
import os
import collections
import numpy as np
import pandas as pd
import datetime
from abc import ABCMeta
from osgeo import gdal, ogr, osr
from girs import srs
from girs.feat.geom import is_topology_2d, is_topology_0d, geometries_to_geometry_collection
from future.utils import with_metaclass
ogr.UseExceptions()
# ogr.DontUseExceptions()


class DataFrameFeature(object):
    """
    An OGRLayer in LayersSet can be transformed into a pandas DataFrame, each row corresponding to a feature.
    DataFrameFeature is used as geometry field in pandas DataFrame and Series. It encapsulates the geometry and
    shows its name in the pandas column. DataFrameFeature has two attributes:

        - the feature ID of the object in this layer.
        - A DataFrameLayer. Each row in the Series has one and the same DataFrameLayer object
    """

    _format_to_export = {'kml': 'ExportToKML', 'isowkt': 'ExportToIsoWkt', 'gml': 'ExportToGML', 'wkb': 'ExportToWkb',
                         'json': 'ExportToJson', 'isowkb': 'ExportToIsoWkb', 'wkt': 'ExportToWkt'}

    geometry_fieldname = '_GEOM_'

    def __init__(self, data_frame_layer, feature_id):
        """

        :param data_frame_layer: a layer
        :type data_frame_layer: OGRLayer
        :param feature_id: feature ID
        :type feature_id: int
        """
        self.data_frame_layer = data_frame_layer
        self.feature_id = feature_id

    def __repr__(self):
        return ogr.GeometryTypeToName(self.get_layer().GetGeomType())

    def get_layer(self):
        return self.data_frame_layer.get_layer()

    def get_geometry(self, fmt='wkb'):
        """Return the geometry as string. Default is wkb format

        :param fmt: format: gml, isowkb, isowkt, json, kml, wkb, or wkt
        :type fmt: str
        :return: OGRGeometry
        :rtype: str
        """
        feat = self.get_layer().GetFeature(self.feature_id)
        g = feat.GetGeometryRef()
        return getattr(g, DataFrameFeature._format_to_export[fmt])()

    def get_geometry_type(self):
        """Return the layer geometry type (`OGRGeometry.GetGeomType()`)

        :return: geometry type
        :rtype: int
        """
        return self.get_layer().GetGeomType()

    def get_id(self):
        """Return the feature ID

        :return: feature ID
        :rtype: int
        """
        return self.feature_id

    def apply(self, method, *args, **kwargs):
        """Apply the given `method` with given `args` and `kwargs` to the geometry.

        In the following example, ``dfg.apply('GetArea')`` returns ``geom.GetArea()``. `dfg` is an instance of
        `DataFrameGeometry`.

            ``df['area_km2'] = df['geom'].apply(lambda dfg: dfg.apply('GetArea'))``

        The OGRGeometry methods are described in

            - http://gdal.org/python/ and
            - http://www.gdal.org/classOGRGeometry.html:

        ``', '.join([m for m in dir(ogr.Geometry) if not m.startswith('_')])`` shows the available methods:

        ::

            AddGeometry, AddGeometryDirectly, AddPoint, AddPointM, AddPointZM, AddPoint_2D, Area,
            AssignSpatialReference, Boundary, Buffer, Centroid, Clone, CloseRings, Contains,
            ConvexHull, CoordinateDimension, Crosses, DelaunayTriangulation, Destroy, Difference,
            Disjoint, Distance, Distance3D, Empty, Equal, Equals, ExportToGML, ExportToIsoWkb,
            ExportToIsoWkt, ExportToJson, ExportToKML, ExportToWkb, ExportToWkt, FlattenTo2D,
            GetArea, GetBoundary, GetCoordinateDimension, GetCurveGeometry, GetDimension,
            GetEnvelope, GetEnvelope3D, GetGeometryCount, GetGeometryName, GetGeometryRef,
            GetGeometryType, GetLinearGeometry, GetM, GetPoint, GetPointCount, GetPointZM,
            GetPoint_2D, GetPoints, GetSpatialReference, GetX, GetY, GetZ, HasCurveGeometry,
            Intersect, Intersection, Intersects, Is3D, IsEmpty, IsMeasured, IsRing, IsSimple,
            IsValid, Length, Overlaps, PointOnSurface, Segmentize, Set3D, SetCoordinateDimension,
            SetMeasured, SetPoint, SetPointM, SetPointZM, SetPoint_2D, Simplify,
            SimplifyPreserveTopology, SymDifference, SymmetricDifference, Touches, Transform,
            TransformTo, Union, UnionCascaded, Value, Within, WkbSize, next, thisown

        :param method: an OGRGeometry method
        :type method: str
        :param args: args for the OGRGeometry method
        :param kwargs: kwargs for the OGRGeometry method
        :return: the return value of the applied method
        """
        feat = self.get_layer().GetFeature(self.feature_id)
        g = feat.GetGeometryRef()
        result = getattr(g, method)(*args, **kwargs)
        del feat
        return result


class DataFrameLayer(object):

    def __init__(self, lrs, layer_number):
        self.lrs = lrs  # keep dataset alive
        self.layer_number = layer_number
        # Allocate this instance here in order to have only one reference along all geom column
        self.layer = self.lrs.get_layer(layer_number=self.layer_number)

    def get_layer(self):
        # Don't use self.lrs.get_layer(layer_number=self.layer_number) here. Doing that, each return from get_layer()
        # will have a different reference
        return self.layer

    def to_dtype(self):
        return FieldDefinition.oft2numpy(self.field_definition.oft_type)
        # t0 = self.field_definition.oft_type
        # return DataFrameOGRColumn._ogr_to_dtype_dict[t0] if t0 in DataFrameOGRColumn._ogr_to_dtype_dict else object


def _get_data_frame_geometry_indices(df):
    """Return a list of geometry columns indices

    :param df: pandas DataFrame or Series
    :return: list of indices (int)
    """
    try:
        df = df.to_frame()
    except AttributeError:
        pass
    return [idx for idx, field_name in enumerate(df.columns)
            if df[field_name].apply(lambda v: isinstance(v, DataFrameFeature)).any()]


def _remove_data_frame_geometry_column(df):
    """Return a DataFrame with no geometry column

    :param df: pandas DataFrame or Series
    :return: pandas DataFrame
    """
    try:
        df = df.to_frame()
    except AttributeError:
        pass
    idx = _get_data_frame_geometry_indices(df)
    if idx:
        columns = df.columns
        df.columns = list(range(len(columns)))
        columns = [c for c in columns if c not in columns[idx]]
        for c in sorted(idx, reverse=True):
            del df[c]
        df.columns = columns
    return df


def _get_extension_shortname_dict():

    def _insert_extensions(drv_dict, short_name, extensions=None):
        """
        Extension is set to lower case
        :param drv_dict:
        :param short_name:
        :param extensions:
        :return:
        """
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
    """

    Example of driver names obtained with

        ``', '.join(FeatDrivers.get_driver_names())``:

    ::

        PCIDSK, netCDF, JP2OpenJPEG, JPEG2000, PDF, DB2ODBC, ESRI Shapefile, MapInfo File, UK .NTF,
        OGR_SDTS, S57, DGN, OGR_VRT, REC, Memory, BNA, CSV, GML, GPX, LIBKML, KML, GeoJSON,
        OGR_GMT, GPKG, SQLite, ODBC, WAsP, PGeo, MSSQLSpatial, PostgreSQL, OpenFileGDB, XPlane,
        DXF, Geoconcept, GeoRSS, GPSTrackMaker, VFK, PGDUMP, OSM, GPSBabel, SUA, OpenAir, OGR_PDS,
        WFS, HTF, AeronavFAA, Geomedia, EDIGEO, GFT, SVG, CouchDB, Cloudant, Idrisi, ARCGEN,
        SEGUKOOA, SEGY, XLS, ODS, XLSX, ElasticSearch, Walk, Carto, AmigoCloud, SXF, Selafin, JML,
        PLSCENES, CSW, VDV, TIGER, AVCBin, AVCE00, HTTP


    Unambiguous relation between file suffix and driver. The driver is retrieved directly from filename:

    ::

              Suffix         Drivers
        0        000             S57
        1        bna             BNA
        2        csv             CSV
        3        dat          XPlane
        4         db          SQLite
        5        dgn             DGN
        6        dxf             DXF
        7        e00          AVCE00
        8        gdb     OpenFileGDB
        9    geojson         GeoJSON
        10       gml             GML
        11       gmt         OGR_GMT
        12      gpkg            GPKG
        13       gpx             GPX
        14       gtm   GPSTrackMaker
        15       gtz   GPSTrackMaker
        16       gxt      Geoconcept
        17       jml             JML
        18      json         GeoJSON
        19       kmz          LIBKML
        20       map            WAsP
        21       mid    MapInfo File
        22       mif    MapInfo File
        23        nc          netCDF
        24       ods             ODS
        25       osm             OSM
        26       pbf             OSM
        27       pdf             PDF
        28       pix          PCIDSK
        29       rec             REC
        30       shp  ESRI Shapefile
        31       sql          PGDUMP
        32    sqlite          SQLite
        33       svg             SVG
        34       sxf             SXF
        35       tab    MapInfo File
        36       thf          EDIGEO
        37  topojson         GeoJSON
        38       vct          Idrisi
        39       vfk             VFK
        40       vrt         OGR_VRT
        41       x10             VDV
        42       xls             XLS
        43      xlsx            XLSX


    Ambiguous relation between file suffix and driver. The driver must be retrieved from filename and suffix:

    ::

          Suffix                Drivers
        0    jp2  JP2OpenJPEG, JPEG2000
        1    kml            LIBKML, KML
        2    mdb         PGeo, Geomedia
        3    txt        Geoconcept, VDV

    No file suffix. The driver must be retrieved from filename and suffix. In the case of Memory the filename can be
    empty:

    ::

        DB2ODBC, UK .NTF, OGR_SDTS, Memory, ODBC, MSSQLSpatial, PostgreSQL, GeoRSS, GPSBabel, SUA,
        OpenAir, OGR_PDS, WFS, HTF, AeronavFAA, GFT, CouchDB, Cloudant, ARCGEN, SEGUKOOA, SEGY,
        ElasticSearch, Walk, Carto, AmigoCloud, Selafin, PLSCENES, CSW, TIGER, AVCBin, HTTP

    """

    _extension_to_driver_name = _get_extension_shortname_dict()

    @staticmethod
    def get_driver(source='', drivername=None):
        """Returns an object `osgeo.ogr.Driver` or `None`.

        Returns a Memory driver:

            1- If drivername='Memory',
            2- If drivername=``None`` and source=``None``
            3- If drivername=``None`` and source=''

        :param source: file name, data base name, or any other OGRDriver source
        :type source: str
        :param drivername:
        :type drivername: str
        :return: ``osgeo.ogr.Driver`` or ``None``
        """

        if drivername:
            drv = ogr.GetDriverByName(drivername)
            if source:
                extension0 = FeatDrivers.get_extension_by_driver_name(drivername)
                if extension0:
                    extension1 = source.split('.')[-1].strip().lower()
                    if extension0 != extension1:
                        msg = 'Extension "{}" does not match required extension "{}"'.format(extension1, extension0)
                        raise ValueError(msg)
            return drv
        elif source:
            extension = source.split('.')[-1].strip().lower()
            driver_names = FeatDrivers._extension_to_driver_name[extension]
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

    @staticmethod
    def get_extension_by_driver_name(drivername):
        """Return the extension of the driver given in `drivername`

        :param drivername:
        :return:
        """
        for extension, drivernames in list(FeatDrivers._extension_to_driver_name.items()):
            if drivername in drivernames:
                return extension
        return None


class FieldDefinition (object):

    oft2numpy_dict = {
        ogr.OFTInteger: np.int64,
        ogr.OFTInteger64: np.int64,
        ogr.OFTIntegerList: str,
        ogr.OFTInteger64List: str,
        ogr.OFTReal: np.float64,
        ogr.OFTRealList: str,
        ogr.OFTString: str,
        ogr.OFTStringList: str,
        ogr.OFTBinary: object,
        ogr.OFTDate: datetime.date,
        ogr.OFTTime: datetime.time,
        ogr.OFTDateTime: datetime.datetime,
        int: int,  # FID has type long
        DataFrameFeature: object
    }

    numpy2oft_dict = {
        np.uint8: ogr.OFSTBoolean,
        np.uint16: ogr.OFTInteger,
        np.uint32: ogr.OFTInteger64,
        np.uint64: ogr.OFTInteger64List,
        np.int8: ogr.OFSTInt16,
        np.int16: ogr.OFSTInt16,
        np.int32: ogr.OFTInteger,
        np.int64: ogr.OFTInteger64,
        np.float64: ogr.OFTReal,
        np.datetime64: ogr.OFTDateTime,
        str: ogr.OFTString,
        DataFrameFeature: ogr.OFTString,
        list: list
        # np.uint128: ogr.OFTInteger64List,
        # np.int128: ogr.OFTInteger64List,
        # np.: ogr.OFTStringList,
        # np.: ogr.OFTBinary,
        # np.datetime64: ogr.OFTDate,
        # np.: ogr.OFTTime,
        # np.: ogr.OFTInteger64List,
    }

    def __init__(self, name, oft_type, width=None, precision=None, oft_subtype=None, nullable=True, default=None):
        """

        :param name: name
        :type name: str
        :param oft_type: ogr.OFTString, ogr.OFTInteger, ogr.OFTReal, ...
        :type oft_type: int
        :param width: maximal number of characters (optional)
        :type width: int
        :param precision: number of digits after decimal point (optional)
        :type precision: int
        :param oft_subtype: ogr.OFSTBoolean, ... (optional)
        :type oft_subtype: str
        :param nullable: a NOT NULL constraint (optional)
        :type nullable: bool
        :param default: a default value (optional)
        """
        self.name = name
        self.oft_type = oft_type
        self.width = width
        self.precision = precision
        self.oft_subtype = oft_subtype
        self.nullable = nullable
        self.default = default

    def __repr__(self):
        if not self.oft_subtype:
            return '{} ({})'.format(self.name, self.get_field_type_name())
        else:
            return '{} ({})'.format(self.name, self.get_field_subtype_name())

    def get_field_type_name(self):
        return ogr.GetFieldTypeName(self.oft_type)

    def get_field_subtype_name(self):
        return ogr.GetFieldSubTypeName(self.oft_subtype)

    def to_ogr(self):
        fd1 = ogr.FieldDefn(self.name, self.oft_type)
        if self.width:
            fd1.SetWidth(self.width)
        if self.precision:
            fd1.SetPrecision(self.precision)
        if self.oft_subtype:
            fd1.SetSubType(self.oft_subtype)
        if self.nullable:
            fd1.SetNullable(self.nullable)
        if self.default is not None:
            fd1.SetDefault(str(self.default))
        return fd1

    @staticmethod
    def from_ogr(fd):
        return FieldDefinition(name=fd.GetName(), oft_type=fd.GetType(), width=fd.GetWidth(),
                               precision=fd.GetPrecision(), oft_subtype=fd.GetSubType(),
                               nullable=fd.IsNullable(), default=fd.GetDefault())

    @staticmethod
    def oft2numpy(oft_type):
        return FieldDefinition.oft2numpy_dict[oft_type]

    @staticmethod
    def numpy2oft(numpy_type):
        return FieldDefinition.numpy2oft_dict[numpy_type]



class LayersSet(with_metaclass(ABCMeta, object)):
    """`LayersSet` is the basis class for all layers subclasses (read, update, and write).

    Many methods in `LayersSet` apply to a specific layer number, which is defined by the parameter
    `layer_number`. The default layer number is always zero, i.e., the first layer is used whenever `layer_number` is
    unset.

    `LayersSet` has an attribute `dataset`, which is an instance of OGRDataSource.

    A data source consists of one more layers (OGRLayer). If a layer instance is retrieved from `LayersSet` and
    the `LayersSet` instance is destroyed, an error will occur. The same applies to the features of a layer. Do not
    destroy or lose the scope of `LayersSet` before the layer instance or its features was used.

    .. seealso::

        - LayersReader
        - LayersEditor
        - LayersUpdate
        - LayersWriter

    """

    def __init__(self):
        """Initialize the abstract class by setting self.dataset = None

        """
        self.dataset = None

    def get_source(self):
        """Return the dataset source

        :return: the dataset source (e.g., file name)
        :rtype: str
        """
        return self.dataset.GetName()

    def get_description(self, layer_number=0):
        """Return the layer description for the given layer number

        Call `OGRLayer.GetDescription()` for the given layer number.

        :param layer_number: layer number. Default is zero
        :type layer_number: int
        :return: `OGRLayer.GetDescription()` for the given layer number.
        :rtype:
        """
        return self.get_layer(layer_number).GetDescription()

    def show(self, layer_number=0, width=200, max_rows=60):
        """Show the layer as pandas DataFrame

        :param layer_number: layer number. Default is zero
        :type layer_number: int
        :param width: display width
        :type width: int
        :param max_rows: maximal number of rows displayed
        :type max_rows: int
        :return:
        """
        pd.set_option("display.width", width)
        pd.set_option("display.max_rows", max_rows)
        print(self.data_frame(layer_number=layer_number))
        pd.reset_option("display.width")
        pd.reset_option("display.max_rows")

    def copy(self, target='', **kwargs):
        """Return a copy of the layers

        The copy can be:

            - a full copy of all fields and field values
            - a copy of all field values of selected attribute fields
            - a copy of selected field values (filter) from selected attribute fields

        If target is given, save the copy before returning the new LayersWriter object.

        As per default, all fields are copied. The kwargs key `ofields` (output fields) defines the field name or
        list of field names to be copied. Per default this applies to layer number zero. Output fields for specific
        layers are defined by the layer number appended to `ofields`. For example, `ofields2=['NAME', 'AREA']` applies
        to layer number two (which is the third layer). `ofields0=['NAME', 'AREA']` is equivalent to
        `ofields=['NAME', 'AREA']`.

        As per default, all field values are copied. The kwargs key `filter` defines a filter to be applied to the
        features. Per default this applies to layer number zero. Filters for specific layers are defined by the layer
        number appended to `filter`. For example, `filter2=lambda a: a < 1000` applies to all features
        in layer number two (which is the third layer). If `filter` is used, then `ffields` (filter fields) must be
        also defined. In the example above this could be `filter2=lambda a: a < 1000, ffields2=['AREA']`.
        Another example: `lrs.copy(filter2=lambda a, n: a < 1000 and n.startswith('A'), ffields2=['AREA', 'NAME'])`. The
        filter parameters follow the same order of the list in `ffields`.

        :param target: file name. Default on memory
        :param **kwargs:
            ofields: name, list of names, list of (name, new name), dictionary {layer number: name},
                dictionary {layer number: list of names}, or dictionary {layer number: list of (name, new name)}
            filter: filter function
            ffields: list of field names to apply the filter function, or a single field name
        :return: LayerWriter
        """
        target = '' if target is None else target.strip()
        if target and not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        lrs_out = LayersWriter(source=target)

        number_of_layers = self.dataset.GetLayerCount()
        layers_dict = {layer_number: [None]*3 for layer_number in range(number_of_layers)}
        for k, parameters_list in list(kwargs.items()):
            if k.startswith('ofields'):
                idx = 0
            elif k.startswith('filter'):
                idx = 1
            elif k.startswith('ffields'):
                idx = 2
            else:
                msg = 'option {} does not exist. Options are: target, ofields, filter, and ffields'.format(k)
                raise ValueError(msg)
            if not k[-1].isdigit():
                layer_number, k = 0, k + '0'
            else:
                f, n = k, ''
                while f and f[-1].isdigit():
                    n, f = f[-1] + n, f[:-1]
                layer_number = int(n)
            if layer_number >= number_of_layers:
                msg = 'field {}: layer {} greater then the maximum number of layers'.format(k, str(layer_number))
                raise ValueError(msg)
            layers_dict[layer_number][idx] = parameters_list

        def create_feature(lyr_out, fd0_out, feat_in, indices_in):
            feat_new = ogr.Feature(fd0_out)
            for idx_out, idx_in in enumerate(indices_in):
                feat_new.SetField(idx_out, feat_in.GetField(idx_in))
            g = feat_in.GetGeometryRef()
            feat_new.SetGeometry(g)
            lyr_out.CreateFeature(feat_new)

        for ilc in range(number_of_layers):
            ly_in = self.dataset.GetLayer(ilc)
            ld_in = ly_in.GetLayerDefn()

            parameters = layers_dict[ilc]
            o_fields = parameters[0]
            f_filter = parameters[1]
            f_fields = parameters[2]
            if o_fields:
                # try:
                #     o_fields = parameters[0][ilc]
                # except KeyError:
                #     pass
                if isinstance(o_fields, basestring):
                    o_fields = [o_fields]
                for i in range(len(o_fields)):
                    if isinstance(o_fields[i], basestring):
                        o_fields[i] = [o_fields[i], o_fields[i]]
                try:
                    o_fields_names = dict()
                    for fn in o_fields:
                        if isinstance(fn, basestring):
                            o_fields_names[fn] = fn
                        elif len(fn) == 1:
                            o_fields_names[fn[0]] = fn[0]
                        else:
                            o_fields_names[fn[0]] = fn[1]
                    ofields_indices = [ld_in.GetFieldIndex(fn[0]) for fn in o_fields]
                except RuntimeError as e:
                    msg = 'Error in ofields ({}): {}'.format(', '.join(str(f) for f in o_fields), str(e.message))
                    raise RuntimeError(msg)
                if -1 in ofields_indices:
                    fn = o_fields[ofields_indices.index(-1)]
                    if fn.upper() == 'FID':
                        msg = 'ERROR: FID used as output field'
                    else:
                        msg = 'Output field {} does not exist'.format(fn)
                    raise ValueError(msg)
            else:
                ofields_indices = list(range(ld_in.GetFieldCount()))
                o_fields_names = [ld_in.GetFieldDefn(i).GetName() for i in ofields_indices]
                o_fields_names = {fn: fn for fn in o_fields_names}
            # Get filter fields indices
            if f_fields:
                if isinstance(f_fields, basestring):
                    f_fields = [f_fields]
                ffields_indices = sorted([ld_in.GetFieldIndex(fn) if fn != 'FID' else -2 for fn in f_fields])
                if -1 in ffields_indices:
                    msg = 'Filter field {} does not exist'.format(f_fields[ffields_indices.index(-1)])
                    raise ValueError(msg)
            else:
                ffields_indices = list()

            fd_out = [FieldDefinition.from_ogr(ld_in.GetFieldDefn(ifd)) for ifd in ofields_indices]
            for fd in fd_out:
                fd.name = o_fields_names[fd.name]
            ly_out = lrs_out.create_layer(str(ilc), ly_in.GetSpatialRef(), ly_in.GetGeomType(), fd_out)
            ld_out = ly_out.GetLayerDefn()
            # copy features
            ly_in.ResetReading()
            for feat in ly_in:
                if not f_filter or f_filter(*[feat.GetFID() if idx == -2 else feat.GetField(idx)
                                              for idx in ffields_indices]):
                    create_feature(ly_out, ld_out, feat, ofields_indices)
            feat = None
            ly_in.ResetReading()
        lrs_out.dataset.FlushCache()
        return lrs_out

    def join(self, this_fieldname, other, other_fieldname, target=None,
             this_layer_number=0, other_layer_number=0):
        """Return a LayerWriter with joined features

        The (other) features to append to this features may be:

            1- a pandas DataFrame or Series
            2- a LayerSet
            3- a source (file name): a LayerReader instance will be created

        In the last two cases:
            - the layer number of the LayerSet/LayerReader can be set with other_layer_number
            - the geometry will be removed from the DataFrame/Series


        :param this_fieldname: field name in this LayersSet to be used to join to the other object
        :type this_fieldname: str
        :param other: data to be joined to this LayersSet
        :type other: pandas.DataFrame, pandas.Series, LayerSet, or LayersReader file name
        :param other_fieldname: field name used to join to this LayersSet
        :type other_fieldname: str
        :param target: target
        :type target: str
        :param this_layer_number: layer number of this LayerSet (default 0)
        :type this_layer_number: int
        :param other_layer_number: layer number of other LayerSet (default 0)
        :type other_layer_number: int
        :return:
        :rtype:
        """
        df_this = self.data_frame(this_layer_number)
        try:
            lrs_other = LayersReader(other)
            df_other = lrs_other.data_frame(other_layer_number)
            del df_other[DataFrameFeature.geometry_fieldname]
        except RuntimeError:
            try:
                df_other = other.data_frame()
                del df_other[DataFrameFeature.geometry_fieldname]
            except AttributeError:
                df_other = other
        df_other = _remove_data_frame_geometry_column(df_other)
        df_this = df_this.set_index(this_fieldname)
        df_other = df_other.set_index(other_fieldname)
        df_this = pd.concat([df_this, df_other], axis=1).reset_index()
        # df_this = df_this.fillna(nan)
        df_this.columns = _get_unique_field_names(df_this.columns)
        lrs_out = data_frame_to_layer(df_this, target=target)
        return lrs_out

    def data_frame(self, layer_number=0):
        """Return a layer as pandas DataFrame

        :param layer_number: layer number. Default is zero
        :type layer_number: int
        :return: OGRLayer as pandas DataFrame
        """

        ld = self.get_layer_definition(layer_number)
        if not ld:
            return None
        nf = ld.GetFieldCount()
        # Get fields names and types from the layer
        column_name_geo = DataFrameFeature.geometry_fieldname
        column_name_fid = 'FID'
        names = [column_name_fid, column_name_geo]
        types = [int, DataFrameFeature]
        if nf > 0:
            fds = [FieldDefinition.from_ogr(ld.GetFieldDefn(i)) for i in range(ld.GetFieldCount())]
            names0, types0 = list(zip(*[[fd.name, fd.oft_type] for fd in fds]))
            names += names0
            types += types0
        names_types = collections.OrderedDict(list(zip(names, types)))
        # Retrieve values from layer
        ogr_layer = self.get_layer(layer_number)
        dfl = DataFrameLayer(self, layer_number)
        ogr_layer.ResetReading()
        values = [[f.GetFID(), DataFrameFeature(dfl, f.GetFID())] +
                  [f.GetField(i) for i in range(nf)] for f in ogr_layer]
        ogr_layer.ResetReading()
        # Create DataFrame
        df = pd.DataFrame(values, columns=names)
        for name in names:
            df[name] = df[name].astype(FieldDefinition.oft2numpy(names_types[name]))
        return df.set_index(column_name_fid)

    def get_layer_count(self):
        """Return the number of layers

        :return: number of layers
        """
        return self.dataset.GetLayerCount()

    def get_layer_name(self, layer_number=0):
        """Return the layer name for the given layer number

        :param layer_number: default is zero
        :return: layer name
        """
        return self.get_layer(layer_number).GetName()

    def get_layer(self, layer_number=0):
        """Return the OGRLayer instance for the given layer number

        Use this method with care. If the layer is used after this instance (`LayersSet`) is destroyed, the system will
        probably crash. The following may also crash the system::

            lyr = LayersReader(source='D:/tmp/test.shp').get_layer()

        In order to avoid it, use::

            lrs = LayersReader(source='D:/tmp/test.shp')
            lyr = lrs.get_layer()
            # lrs should be deleted only after lyr is used
            del lrs

        :param layer_number: default is zero
        :return: OGRLayer
        """
        return self.dataset.GetLayer(layer_number)

    def layers(self):
        """
        Iterator over all layers
        :return:
        """
        for i in range(self.dataset.GetLayerCount()):
            yield self.dataset.GetLayer(i)

    def get_layer_definition(self, layer_number=0):
        """Return the `OGRLayerDefn` instance for the given layer number

        :param layer_number: default is zero
        :return: OGRLayerDefn
        """
        lyr = self.get_layer(layer_number)
        return lyr.GetLayerDefn() if lyr else None

    def layers_definitions(self):
        """Iterator over layer definitions, one definition for each layer

        The iterations yield the `OGRLayerDefn` instance of each layer

        :return: yield OGRLayerDefn
        """
        for i in range(self.dataset.GetLayerCount()):
            ly = self.dataset.GetLayer(i)
            yield ly.GetLayerDefn()

    def get_features_count(self, layer_number=0):
        """Return the number of features for the given layer number

        :param layer_number:
        :return:
        """
        return self.get_layer(layer_number).GetFeatureCount()

    def create_feature(self, **kwargs):
        """Create a feature and insert it into the layer

        :param kwargs:
            :key geom: geometry as OGRGeometry, wkt, or wkb
            :key layer_number: default 0
            "field name": field name (str) as key and field type as values
        :return:
        """
        lyr = self.get_layer(layer_number=kwargs.pop('layer_number', 0))
        feature = ogr.Feature(lyr.GetLayerDefn())
        geom = kwargs.pop('geom')
        try:
            feature.SetGeometry(geom)
        except TypeError:
            try:
                feature.SetGeometry(ogr.CreateGeometryFromWkb(geom))
            except RuntimeError:
                feature.SetGeometry(ogr.CreateGeometryFromWkt(geom))
        fieldnames_in = list(kwargs.keys())
        for k in fieldnames_in:
            try:
                feature.SetField(k, kwargs[k])
            except (NotImplementedError, RuntimeError) as e:
                print(k, kwargs[k], type(kwargs[k]))
                raise e
        ldf = lyr.GetLayerDefn()
        for i in range(ldf.GetFieldCount()):
            fd = ldf.GetFieldDefn(i)
            if fd.GetName() not in fieldnames_in:
                df = fd.GetDefault()
                if df is not None:
                    feature.SetField(i, df)
        lyr.CreateFeature(feature)
        del feature

    def get_field_count(self, layer_number=0):
        """Return number of fields for the given layer number. FID is not taken into consideration.

        :param layer_number: default is zero
        :return:
        """
        return self.get_layer_definition(layer_number).GetFieldCount()

    def get_field_names(self, layer_number=0):
        """Return the field names for the given layer number

        :param layer_number:
        :return:
        """
        ld = self.get_layer_definition(layer_number)
        result = []
        for i in range(ld.GetFieldCount()):
            fd = ld.GetFieldDefn(i)
            result.append(fd.GetName())
        return result

    def get_field_numbers(self, field_names=None, layer_number=0):
        """Return the field numbers for the given field names. Set field number = -1 if field name does not exist

        :param field_names:
        :param layer_number: default is zero
        :return: list of int
        """
        if isinstance(field_names, basestring):
            field_names = [field_names]
        if not field_names:
            field_names = self.get_field_names(layer_number)
        ld = self.get_layer_definition(layer_number=layer_number)
        return [ld.GetFieldIndex(field_name) for field_name in field_names]

    def get_field_definition(self, field_name, layer_number=0):
        """Return the FieldDefinition for this field name

        :param field_name: field name
        :type field_name: str
        :param layer_number: layer number. Default is zero
        :type layer_number: int
        :return:
        """
        if field_name:
            ld = self.get_layer_definition(layer_number)
            for i in range(ld.GetFieldCount()):
                fd = ld.GetFieldDefn(i)
                if field_name == fd.GetName():
                    return FieldDefinition.from_ogr(fd)
        return None

    def get_field_definitions(self, field_names=None, layer_number=0):
        """Returns a list of field definitions

        :param field_names:
        :param layer_number: default is zero
        :return: list of field definitions
        :rtype: list of FieldDefinition
        """
        ld = self.get_layer_definition(layer_number)
        result = []
        for i in range(ld.GetFieldCount()):
            fd = ld.GetFieldDefn(i)
            if not field_names or fd.GetName() in field_names:
                result.append(FieldDefinition.from_ogr(fd))
        return result

    def get_field_definitions_data_frame(self, field_names=None, layer_number=0):
        """Return the field definitions in a pandas DataFrame

        :param field_names: field names. Default None means all fields
        :param layer_number: layer number, default is zero
        :return: a pandas DataFrame with field definitions
        :rtype: pandas.DataFrame
        """
        name_list = list()
        oft_type_list = list()
        oft_type_name_list = list()
        width_list = list()
        precision_list = list()
        oft_subtype_list = list()
        oft_subtype_name_list = list()
        nullable_list = list()
        default_list = list()
        for fd in self.get_field_definitions(field_names=field_names, layer_number=layer_number):
            name_list.append(fd.name)
            oft_type_list.append(fd.oft_type)
            oft_type_name_list.append(fd.get_field_type_name())
            width_list.append(fd.width)
            precision_list.append(fd.precision)
            oft_subtype_list.append(fd.oft_subtype)
            oft_subtype_name_list.append(fd.get_field_subtype_name())
            nullable_list.append(fd.nullable)
            default_list.append(fd.default)
        return pd.DataFrame({'name': name_list, 'type': oft_type_list, 'type_name': oft_type_name_list,
                             'width': width_list, 'precision': precision_list, 'subtype': oft_subtype_list,
                             'subtype_name': oft_subtype_name_list, 'nullable': nullable_list, 'default': default_list},
                            columns=['name', 'type', 'type_name', 'width', 'precision', 'subtype', 'subtype_name',
                                     'nullable', 'default'])

    def get_field_ids(self, layer_number=0):
        """Return field ids

        :param layer_number:
        :return: field ids
        :rtype: list
        """
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        ids = [feat.GetFID() for feat in ly]
        ly.ResetReading()
        return ids

    def get_field_values(self, field_names=None, layer_number=0):
        """Return field values

        :param field_names:
        :param layer_number:
        :return:
        """
        if isinstance(field_names, basestring):
            field_names = [field_names]
        if not field_names:
            field_names = self.get_field_names(layer_number)
        field_numbers = self.get_field_numbers(field_names, layer_number)
        if -1 in field_numbers:
            for i in field_numbers:
                if i == -1:
                    msg = 'Field {} does not exist'.format(field_names[i])
                    raise AttributeError(msg)
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        feat_list = [(feat.GetFID(), [feat.GetField(fn) for fn in field_numbers]) for feat in ly]
        ly.ResetReading()
        if feat_list:
            feat_list, values_list = list(zip(*feat_list))
        else:
            feat_list, values_list = list(), list()
        values_list = list(zip(*values_list))
        df = pd.DataFrame({field_names[i]: v for i, v in enumerate(values_list)}, index=feat_list,
                          columns=[field_names[i] for i in range(len(values_list))])
        df.index.name = 'FID'
        return df

    def has_field_names(self, field_names, layer_number=0):
        """Return a list of bool values with True for existing fields

        :param field_names:
        :param layer_number: default is zero
        :return: list of bool
        """
        if isinstance(field_names, basestring):
            field_names = [field_names]
        ld = self.get_layer_definition(layer_number)
        return [ld.GetFieldIndex(field_name) > -1 for field_name in field_names]

    def generate_field_name(self, field_name, layer_number=0):
        """Returns field_name if it does not exist, else append a digit from 0 to n to field_name and return it

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

    def rename_fields(self, target='', **kwargs):
        """Rename fields in a new `LayersWriter`

        Create a copy of this LayersSet with the new field names given in kwargs als `fieldsX`, where X is the layer
        number. In the following example the field `AREA` from layer number one will be renamed to 'AREAkm2'. The new
        LayerWriter created in the memory and returned to `lrs1`::

            lrs1 = lrs0.rename_fields(fields1={'AREA': 'AREAkm2'})

        For different layers::

            lrs1 = lrs0.rename_fields(fields0={'N': 'NAME'}, fields1={'AREA': 'AREAkm2'})

        Saving the new LayerWriter::

            lrs1 = lrs0.rename_fields(fields1={'AREA': 'AREAkm2'}, target='D:/tmp/example.shp', )

        :param target: file name or other source used in `LayerWriter`. If empty, creates a `LayerWriter` on memory.
        :param kwargs:

            - fields<X>: key used to address layer number <X>
            - all other kwargs are passed to LayerWriter
        :return: LayersWriter with the new field names
        """
        field_x_dict = {}
        for field_x in [f for f in list(kwargs.keys()) if f.startswith('fields')]:
            field_x_dict[field_x] = kwargs.pop(field_x)
        if not field_x_dict:
            msg = 'No parameter starting with field found: {}'.format(sorted(kwargs.keys()))
            raise ValueError(msg)

        target = target.strip()
        if target is None:
            target = ''
        if target and not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        lrs = LayersWriter(source=target)

        number_of_layers = self.dataset.GetLayerCount()
        layers_dict = dict()
        for field_x, fieldname_dict in list(field_x_dict.items()):
            if not field_x[-1].isdigit():
                layer_number0, field_x = 0, field_x + '0'
            else:
                f, n = field_x, ''
                while f and f[-1].isdigit():
                    n, f = f[-1] + n, f[:-1]
                layer_number0 = int(n)
            if layer_number0 >= number_of_layers:
                msg = 'fields {}: layer {} greater then the maximum number of layers'.format(field_x, str(layer_number0))
                raise ValueError(msg)
            layers_dict[layer_number0] = fieldname_dict

        for ilc in range(number_of_layers):
            ly_in = self.dataset.GetLayer(ilc)
            lrs.dataset.CopyLayer(ly_in, ly_in.GetName(), ['OVERWRITE=YES'])

        for ilc in range(number_of_layers):
            ly_out = lrs.dataset.GetLayer(ilc)
            ld_out = ly_out.GetLayerDefn()
            n_fields = ld_out.GetFieldCount()
            fd_out = [ld_out.GetFieldDefn(ifd) for ifd in range(n_fields)]
            for fd in fd_out:
                if fd.GetName() in layers_dict[ilc]:
                    fd.SetName(layers_dict[ilc][fd.GetName()])
        lrs.dataset.FlushCache()
        return lrs

    def fields(self, layer_number=0):
        """Returns a list of lists containing field name, type code, type name, width, and precision

        :param layer_number: default is zero
        :return: list of lists
        """
        ld = self.get_layer_definition(layer_number)
        for i in range(ld.GetFieldCount()):
            fd = ld.GetFieldDefn(i)
            yield fd

    # =========================================================================
    # Geometries
    # =========================================================================
    def _get_geometries(self, method, name, layer_number):
        """

        :param method:
        :param layer_number:
        :return:
        """
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        feat_list = [(feat.GetFID(), getattr(feat.GetGeometryRef(), method)()) for feat in ly]
        ly.ResetReading()
        feat_list, values_list = list(zip(*feat_list))
        sr = pd.Series(values_list, index=feat_list)
        sr.index.name = 'FID'
        sr.name = name
        return sr

    def get_geometry_type(self, layer_number=0):
        """Return the geometry type for the given layer number

        :param layer_number:
        :return:
        """
        return self.get_layer(layer_number).GetGeomType()

    def get_geometries(self, layer_number=0, fmt='wkb'):
        """Return the geometries

        :param layer_number:
        :param fmt: 'wkb' or 'wkt'
        :return:
        :rtype: pandas Series
        """
        if fmt.lower() == 'wkb':
            sr = self._get_geometries('ExportToWkb', name=DataFrameFeature.geometry_fieldname,
                                      layer_number=layer_number)
        elif fmt.lower() == 'wkt':
            sr = self._get_geometries('ExportToWkt', name=DataFrameFeature.geometry_fieldname,
                                      layer_number=layer_number)
        else:
            msg = 'Unknown geometry format {}: valid format are "wkt" or "wkb"'.format(fmt)
            raise ValueError(msg)
        return sr

    def get_geometries_and_field_values(self, field_names=None, layer_number=0, geometry_format='wkb'):
        """Return geometries and field values for the given layer number

        :param field_names:
        :param layer_number:
        :param geometry_format:
        :return:
        """
        if isinstance(field_names, basestring):
            field_names = [field_names]
        ly = self.get_layer(layer_number)
        if not field_names:
            field_names = self.get_field_names(layer_number)
        field_numbers = self.get_field_numbers(field_names, layer_number)
        field_names = [DataFrameFeature.geometry_fieldname] + field_names
        ly.ResetReading()
        geometry_format = str(geometry_format)
        if geometry_format.lower() == 'wkb':
            feat_list = [(feat.GetFID(), [feat.GetGeometryRef().ExportToWkb()] +
                          [feat.GetField(fn) for fn in field_numbers]) for feat in ly]
        elif geometry_format.lower() == 'wkt':
            feat_list = [(feat.GetFID(), [feat.GetGeometryRef().ExportToWkt()] +
                          [feat.GetField(fn) for fn in field_numbers]) for feat in ly]
        else:
            msg = 'Unknown geometry format {}: valid format are "wkt" or "wkb"'.format(geometry_format)
            raise ValueError(msg)
        ly.ResetReading()
        feat_list, values_list = list(zip(*feat_list))
        values_list = list(zip(*values_list))
        df = pd.DataFrame({field_names[i]: v for i, v in enumerate(values_list)}, index=feat_list, columns=field_names)
        df.index.name = 'FID'
        return df

    def get_geometries_areas(self, data_frame=True, layer_number=0):
        sr = self._get_geometries('GetArea', 'area', layer_number=layer_number)
        if data_frame:
            df = pd.DataFrame(sr.values.tolist(), columns=['area'])
            df.index.name = sr.index.name
            return df
        else:
            return sr

    def get_geometries_boundaries(self, fmt='wkt', layer_number=0):
        fmt = fmt.lower()
        geo_formats = {'gml': 'ExportToGML', 'isowkb': 'ExportToIsoWkb', 'isowkt': 'ExportToIsoWkt',
                       'json': 'ExportToJson', 'kml': 'ExportToKML', 'wkb': 'ExportToWkb', 'wkt': 'ExportToWkt'}
        if fmt not in list(geo_formats.keys()):
            msg = 'Format {} unknown'.format(str(fmt))
            raise ValueError(msg)
        sr = self._get_geometries('GetBoundary', 'boundary', layer_number=layer_number)
        return sr.apply(lambda g: getattr(g, geo_formats[fmt])())

    def get_geometries_coordinate_dimensions(self, layer_number=0):
        return self._get_geometries('GetCoordinateDimension', 'coordinate dimension', layer_number=layer_number)

    def get_geometries_curve(self, layer_number=0):
        return self._get_geometries('GetCurveGeometry', 'curve geometry', layer_number=layer_number)

    def get_geometries_dimensions(self, layer_number=0):
        return self._get_geometries('GetDimension', 'dimension', layer_number=layer_number)

    def geometries_export(self, fmt='wkt', layer_number=0):
        fmt = fmt.lower()
        geo_formats = {'gml': 'ExportToGML', 'isowkb': 'ExportToIsoWkb', 'isowkt': 'ExportToIsoWkt',
                       'json': 'ExportToJson', 'kml': 'ExportToKML', 'wkb': 'ExportToWkb', 'wkt': 'ExportToWkt'}
        if fmt not in list(geo_formats.keys()):
            msg = 'Format {} unknown'.format(str(fmt))
            raise ValueError(msg)
        return self._get_geometries(geo_formats[fmt], DataFrameFeature.geometry_fieldname, layer_number=layer_number)

    def get_geometries_centroids(self, data_frame=True, layer_number=0):
        sr = self._get_geometries('Centroid', 'centroid', layer_number=layer_number)
        index_name = sr.index.name
        sr = sr.apply(lambda p: p.GetPoints()[0])
        if data_frame:
            df = pd.DataFrame(sr.values.tolist(), columns=['x_center', 'y_center'])
            df.index.name = index_name
            return df
        else:
            return sr

    def get_geometries_envelopes(self, data_frame=True, layer_number=0):
        sr = self._get_geometries('GetEnvelope', 'envelope', layer_number=layer_number)
        return pd.DataFrame(sr.values.tolist(), columns=['xmin', 'xmax', 'ymin', 'ymax']) if data_frame else sr

    def get_geometries_envelopes_3d(self, data_frame=True, layer_number=0):
        sr = self._get_geometries('GetEnvelope3D', 'envelope 3D', layer_number=layer_number)
        columns = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        return pd.DataFrame(sr.values.tolist(), columns=columns) if data_frame else sr

    def get_geometries_count(self, layer_number=0):
        """Return the number of geometries for each record

        :param layer_number:
        :return: number of geometries in each feature
        :rtype: pandas Series
        """
        return self._get_geometries('GetGeometryCount', 'geom. count', layer_number=layer_number)

    def get_geometries_linear(self, layer_number=0):
        return self._get_geometries('GetLinearGeometry', 'linear geometry', layer_number=layer_number)

    def get_geometries_names(self, layer_number=0):
        """Return the geometries names for each record

        :param layer_number:
        :return: names of the geometries in each feature
        :rtype: pandas Series
        """
        return self._get_geometries('GetGeometryName', 'geom. name', layer_number=layer_number)

    def get_geometries_types(self, layer_number=0):
        return self._get_geometries('GetGeometryType', 'geom. type', layer_number=layer_number)

    def get_geometries_m(self, layer_number=0):
        return self._get_geometries('GetM', 'm', layer_number=layer_number)

    def get_geometries_x(self, layer_number=0):
        return self._get_geometries('GetX', 'x', layer_number=layer_number)

    def get_geometries_y(self, layer_number=0):
        return self._get_geometries('GetY', 'y', layer_number=layer_number)

    def get_geometries_z(self, layer_number=0):
        return self._get_geometries('GetZ', 'z', layer_number=layer_number)

    def get_geometries_point_count(self, layer_number=0):
        """Return zeros if geometries are not points, else the number of points in each geometry

        :param layer_number:
        :return:
        """
        return self._get_geometries('GetPointCount', 'point count', layer_number=layer_number)

    def get_geometries_points_2d(self, data_frame=True, layer_number=0):
        sr = self._get_geometries('GetPoint_2D', 'point 2D', layer_number=layer_number)
        if data_frame:
            columns = ['x', 'y']
            df = pd.DataFrame(sr.values.tolist(), columns=columns)
            df.index.name = 'FID'
            return df
        else:
            return sr

    def get_geometries_points(self, data_frame=True, layer_number=0):
        sr = self._get_geometries('GetPoint', 'point', layer_number=layer_number)
        if data_frame:
            columns = ['x', 'y', 'z']
            df = pd.DataFrame(sr.values.tolist(), columns=columns)
            df.index.name = 'FID'
            return df
        else:
            return sr

    def get_geometries_points_zm(self, data_frame=True, layer_number=0):
        sr = self._get_geometries('GetPointZM', 'point ZM', layer_number=layer_number)
        if data_frame:
            columns = ['x', 'y', 'z', 'm']
            df = pd.DataFrame(sr.values.tolist(), columns=columns)
            df.index.name = 'FID'
            return df
        else:
            return sr

    def get_geometries_spatial_references(self, layer_number=0):
        return self._get_geometries('GetSpatialReference', 'spatial reference', layer_number=layer_number)

    def get_geometries_wkb_size(self, layer_number=0):
        return self._get_geometries('WkbSize', 'wkb size', layer_number=layer_number)

    def get_geometry_union_as_wkb(self, layer_number=0):
        """Return the geometry union

        :param layer_number:
        :return:
        """
        ly = self.get_layer(layer_number)
        union = None
        ly.ResetReading()
        for feat in ly:
            geom = feat.GetGeometryRef().ExportToWkb()
            if geom:
                union = ogr.CreateGeometryFromWkb(geom) if not union else union.Union(ogr.CreateGeometryFromWkb(geom))
        return union.ExportToWkb()

    def is_geometry_3d(self, layer_number=0):
        return self._get_geometries('Is3D', 'Is3D', layer_number=layer_number)

    def is_geometry_empty(self, layer_number=0):
        return self._get_geometries('IsEmpty', 'IsEmpty', layer_number=layer_number)

    def is_geometry_measured(self, layer_number=0):
        return self._get_geometries('IsMeasured', 'IsMeasured', layer_number=layer_number)

    def is_geometry_ring(self, layer_number=0):
        return self._get_geometries('IsRing', 'IsRing', layer_number=layer_number)

    def is_geometry_simple(self, layer_number=0):
        return self._get_geometries('IsSimple', 'IsSimple', layer_number=layer_number)

    def is_geometry_valid(self, layer_number=0):
        return self._get_geometries('IsValid', 'IsValid', layer_number=layer_number)

    def is_geometry_point(self, layer_number=0):
        """Check whether the layer has a point geometry

        :param layer_number:
        :return:
        """
        return self.get_geometries_types(layer_number=layer_number).apply(lambda t: is_topology_0d(t))

    def is_geometry_polygon(self, layer_number=0):
        """Check whether the layer has a polygon geometry

        :param layer_number:
        :return:
        """
        g_types = self.get_geometries_types(layer_number=layer_number).apply(lambda t: is_topology_2d(t))
        return g_types

    def get_coordinate_system(self, fmt='wkt', layer_number=0):
        """Return the coordinate system

        Possible formats:

            * 'xml'
            * 'wkt'
            * 'usgs'
            * 'proj4'
            * 'prettywkt'
            * 'pci'
            * 'micoordsys'

        :param fmt:
        :param layer_number:
        :return:
        """
        ly_in = self.dataset.GetLayer(layer_number)
        return srs.export(ly_in.GetSpatialRef(), fmt=fmt)

    def get_extent(self, layer_number=0, scale=0.0):
        """Return the layer extension (bounding box)

        :param layer_number: default is zero
        :param scale: fraction of width and height to scale. Default is zero (no scaling). Negative values shrink the
            extention.
        :return: (xmin, xmax, ymin, ymax) as floats

        """
        xmin, xmax, ymin, ymax = self.get_layer(layer_number).GetExtent()
        if scale and scale != 0.0:
            dx, dy = xmax - xmin, ymax - ymin
            xmin -= dx * scale
            xmax += dx * scale
            ymin -= dy * scale
            ymax += dy * scale
        return xmin, xmax, ymin, ymax

    def transform(self, other=None, **kwargs):
        """Transform to another coordinate system
        :param other: another object returning srs as wkt from 'get_coordinate-system()'
        :param kwargs:

            Keys/values for the transformation:

            :key srs: (osgeo.osr.SpatialReference) srs object
            :key epsg: (str)
            :key epsga: (str)
            :key erm: (str)
            :key esri: (str)
            :key mi: (str)
            :key ozi: (str)
            :key pci: (str)
            :key proj4: (str)
            :key url: (str)
            :key usgs: (str)
            :key wkt: (str)
            :key xml: (str)

            Further key/values are used in LayersWriter

        :return:
        """
        sr_out = None
        if other:
            sr_out = srs.srs_from_wkt(other.get_coordinate_system())
        if not sr_out:
            sr_out = kwargs.pop('srs', None)
        if not sr_out:
            keys = ['epsg', 'epsga', 'erm', 'esri', 'mi', 'ozi', 'pci', 'proj4', 'url', 'usgs', 'wkt', 'xml']
            sr_out = [srs.get_srs(**{k: kwargs.pop(k, None)}) for k in keys if k in list(kwargs.keys())][0]
        if not sr_out:
            return None

        lrs_out = LayersWriter(**kwargs)

        for ilc, lyr_self in enumerate(self.layers()):
            ld_in = self.get_layer_definition(ilc)
            sr_self = lyr_self.GetSpatialRef()
            coord_trans = osr.CoordinateTransformation(sr_self, sr_out)
            fields_indices = self.get_field_numbers()
            fd_out = [FieldDefinition.from_ogr(ld_in.GetFieldDefn(i)) for i in fields_indices]
            ly_out = lrs_out.create_layer(str(ilc), sr_out, lyr_self.GetGeomType(), fd_out)
            ld_out = ly_out.GetLayerDefn()
            lyr_self.ResetReading()
            for feat in lyr_self:
                feat_new = ogr.Feature(ld_out)
                for idx in fields_indices:
                    feat_new.SetField(idx, feat.GetField(idx))
                g = feat.GetGeometryRef()
                geom_out = g.Clone()
                geom_out.Transform(coord_trans)
                feat_new.SetGeometry(geom_out)
                ly_out.CreateFeature(feat_new)
                feat = None
            lyr_self.ResetReading()
        lrs_out.dataset.FlushCache()
        return lrs_out

    def set_attribute_filter(self, filter_string, layer_number=0):
        """Set an attribute filter

        :param filter_string:
        :param layer_number:
        :return:
        """
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        ly.SetAttributeFilter(filter_string)

    def set_spatial_filter(self, lrs, layer_number=0):
        """Set a spatial filter

        :param lrs:
        :param layer_number:
        :return:
        """
        try:
            lrs = LayersReader(lrs)
            geometries = lrs.get_geometries(layer_number)
        except (TypeError, RuntimeError):
            try:
                geometries = lrs.get_geometries(layer_number)
            except (TypeError, RuntimeError):
                geometries = lrs
        geom = geometries_to_geometry_collection(geometries)
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        g = ogr.CreateGeometryFromWkb(geom)
        ly.SetSpatialFilter(g)

    def spatial_filter(self, lrs, layer_number=0, method='Intersects'):
        """Yield features according to a spatial filter

        :param lrs:
        :param layer_number:
        :param method:
        :return:
        """
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
            ly.ResetReading()

    def spatial_filter_to_layer(self, lrs, layer_number=0, method='Intersects', output_layers=''):
        """Copy the result of the spatial filter to a layer

        :param lrs:
        :param layer_number:
        :param method:
        :param output_layers:
        :return:
        """
        lrs = lrs.transform(other=self)
        fids = set([feat.GetFID() for feat in self.spatial_filter(lrs, layer_number, method)])
        return self.copy(target=output_layers, filter=lambda fid: fid in fids, ffields='FID')

    def buffer(self, buffer_dist, target=None, layer_number=0):
        """Return buffer

        :param buffer_dist:
        :param target:
        :param layer_number:
        :return:
        """
        lyr_in = self.get_layer(layer_number)
        lrs_out = LayersWriter(source=target)
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

    def centroid(self, target=None, drivername=None, **kwargs):
        """Return a new `LayersWriter` with centroids

        Return a LayersWriter with the centroids of the geometries in this LayersSet

        :param target: see `source` in LayersWriter
        :param drivername: see LayersWriter
        :param kwargs: see LayersWriter
        :return:
        """
        lrs_point = LayersWriter(source=target, drivername=drivername, **kwargs)
        for i in range(self.get_layer_count()):
            lyr_self = self.get_layer(i)
            ld_self = self.get_layer_definition(layer_number=i)
            field_definitions0 = []
            for ifd in range(ld_self.GetFieldCount()):
                field_definitions0.append(FieldDefinition.from_ogr(ld_self.GetFieldDefn(ifd)))
            lyr_point = lrs_point.create_layer(lyr_self.GetName(), prj=lyr_self.GetSpatialRef(), geom=ogr.wkbPoint,
                                               field_definitions=field_definitions0)
            feature_def = lyr_point.GetLayerDefn()
            lyr_self.ResetReading()
            for feat in lyr_self:
                feature = ogr.Feature(feature_def)
                for idx in range(len(field_definitions0)):
                    feature.SetField(idx, feat.GetField(idx))
                g = feat.GetGeometryRef().Centroid()
                feature.SetGeometry(g)
                lyr_point.CreateFeature(feature)
        return lrs_point

    def convex_hull(self, target=None):
        """Return the convex hull

        :param target:
        :return:
        """
        lrs = LayersWriter(source=target)
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

    def get_convex_hull_as_wkb(self, layer_number=0):
        """Return the convex hull as well known binary format

        :param layer_number:
        :return:
        """
        geom_col = ogr.Geometry(ogr.wkbGeometryCollection)
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        for feat in ly:
            geom_col.AddGeometry(feat.GetGeometryRef())

        # Calculate convex hull
        return geom_col.ConvexHull().ExportToWkb()

    def dissolve(self, **kwargs):
        """Dissolve to a new `LayersWriter`

        :param kwargs:
        :return:
        """
        from girs.feat.proc import dissolve
        return dissolve(self, **kwargs)


class LayersReader(LayersSet):
    """`LayersReader` inherits from `LayersSet`. It is responsible for creating a read only instance of a `OGRDataSet`.
    It has only a constructor.

    .. seealso::

        LayersSet

    """

    def __init__(self, source, drivername=None):
        """Create a read only instance of a `OGRDataSet`

        :param source: file name, database or any other valid source in `ogr.Open(source, 0)`
        :type: str
        """
        super(LayersReader, self).__init__()
        if drivername:
            drv = ogr.GetDriverByName(drivername)
            self.dataset = drv.Open(source, 0)
        else:
            self.dataset = ogr.Open(source, 0)
        if self.dataset is None:
            msg = 'Could not open {}'.format(str(source))
            raise ValueError(msg)


class LayersEditor(with_metaclass(ABCMeta, LayersSet)):
    """`LayersEditor` is an abstract class that inherits from `LayersSet` and is basis class of `LayersUpdate` and
    `LayersWrite`. It contains methods to edit an existing or new dataset.

    """

    def __init__(self):
        """Initializes the upper class

        """
        super(LayersEditor, self).__init__()

    def create_layer(self, name, prj, geom, field_definitions=()):
        """Create a new layer

        :param name: (str) name of the layer
        :param prj:
        :param geom:
        :param field_definitions:
        :return:
        """
        ly = self.dataset.CreateLayer(name, prj, geom)
        if not ly:
            raise Exception('Layer {} with projection {} and geometry {} could not be created'.format(name, prj, geom))
        for fd in field_definitions:
            try:
                ly.CreateField(fd.to_ogr())
            except TypeError as e:
                ly.CreateField(fd)
        return ly

    def field_calculator(self, field_name, calc_function, calc_field_names, layer_number=0):
        """Calculate field values for given field names.

        :param field_name:
        :param calc_function:
        :param calc_field_names:
        :param layer_number:
        :return:
        """
        ly = self.get_layer(layer_number)
        ly.ResetReading()
        idx = self.get_field_numbers(calc_field_names)
        for feat in ly:
            feat.SetField(field_name, calc_function(*[feat.GetField(i) for i in idx]))
            ly.SetFeature(feat)
        ly.ResetReading()

    def calculate_area(self, field_name='area', scale=1.0, layer_number=0):
        """Calculate feature areas and set them to the given field name

        :param field_name:
        :param scale:
        :param layer_number:
        :return:
        """
        if self.is_geometry_polygon(layer_number).all():
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

    def add_fields(self, **kwargs):
        """Add fields to this layers set

        :param kwargs: the key starting with `fields`
        :return:
        """
        fields_dict = {int(k[6:]) if k[6:] else 0: v for k, v in list(kwargs.items()) if k.lower().startswith('fields')}
        for layer_number in sorted(fields_dict.keys()):
            ly = self.get_layer(layer_number)
            try:
                field_definitions0 = fields_dict[layer_number]
                _ = iter(field_definitions0)
            except TypeError:
                field_definitions0 = [fields_dict[layer_number]]
            for fd0 in field_definitions0:
                fd1 = ogr.FieldDefn(fd0.name, fd0.oft_type)
                if fd0.width:
                    fd1.SetWidth(fd0.width)
                if fd0.precision:
                    fd1.SetPrecision(fd0.precision)
                if fd0.oft_subtype:
                    fd1.SetSubType(fd0.oft_subtype)
                if fd0.nullable:
                    fd1.SetNullable(fd0.nullable)
                if fd0.default:
                    fd1.SetDefault(fd0.default)
                ly.CreateField(fd1)

    def delete_fields(self, field_names, layer_number=0):
        """Delete fields from this layers set

        :param field_names:
        :param layer_number:
        :return:
        """
        ly = self.get_layer(layer_number)
        if ly.TestCapability(ogr.OLCDeleteFeature):
            for idx in sorted([i for i in self.get_field_numbers(field_names=field_names, layer_number=layer_number)
                               if i > -1], reverse=True):
                ly.DeleteField(idx)

    def delete_features(self, **kwargs):
        """
        :param kwargs:
            filter: filter function
            ffields: list of field names to apply the filter function, or a single field name
        """
        number_of_layers = self.dataset.GetLayerCount()
        layers_dict = {layer_number: [None]*3 for layer_number in range(number_of_layers)}
        for k, parameters_list in list(kwargs.items()):
            if k.startswith('filter'):
                idx = 0
            elif k.startswith('ffields'):
                idx = 1
            else:
                msg = 'option {} does not exist. Options are filter, and ffields'.format(k)
                raise ValueError(msg)
            if not k[-1].isdigit():
                layer_number = 0
                k = k + '0'
            else:
                f = k
                n = ''
                while f and f[-1].isdigit():
                    n = f[-1] + n
                    f = f[:-1]
                layer_number = int(n)
            if layer_number >= number_of_layers:
                msg = 'field {}: layer {} greater then the maximum number of layers'.format(k, str(layer_number))
                raise ValueError(msg)
            layers_dict[layer_number][idx] = parameters_list

        for ilc in range(number_of_layers):
            ly_in = self.dataset.GetLayer(ilc)
            ld_in = ly_in.GetLayerDefn()
            parameters = layers_dict[ilc]
            f_filter = parameters[0]
            f_fields = parameters[1]
            # Get filter fields indices
            if isinstance(f_fields, basestring):
                f_fields = [f_fields]
            ffields_indices = sorted([ld_in.GetFieldIndex(fn) if fn != 'FID' else 'FID' for fn in f_fields])
            if -1 in ffields_indices:
                msg = 'Filter field {} does not exist'.format(f_fields[ffields_indices.index(-1)])
                raise ValueError(msg)

            for feat in ly_in:
                feat_id = feat.GetFID()
                if f_filter and f_filter(*[feat_id if idx == 'FID' else feat.GetField(idx) for idx in ffields_indices]):
                    ly_in.DeleteFeature(feat_id)
            ly_in.ResetReading()


class LayersUpdate(LayersEditor):
    """`LayersUpdate` inherits from `LayersEditor`. It is responsible for creating an update instance of a `OGRDataSet`.
    It has only a constructor.

    .. seealso::

        - LayersSet
        - LayersEditor

    """

    def __init__(self, source):
        """

        :param source:
        """
        super(LayersUpdate, self).__init__()
        try:
            source = source.get_source()
        except AttributeError:
            pass
        self.dataset = ogr.Open(source, 1)


class LayersWriter(LayersEditor):
    """`LayersWriter` inherits from `LayersEditor`. It is responsible for creating a new instance of a `OGRDataSet`.
    It has only a constructor.

    .. seealso::

        - LayersSet
        - LayersEditor

    """

    def __init__(self, *args, **kwargs):
        """Instantiate a LayersWriter object

        Example:

        .. code-block:: python

                srs0 = get_srs(epsg=4326)
                fields0 = [FieldDefinition("Name", ogr.OFTString), FieldDefinition("Area", ogr.OFTReal)]
                fields1 = [FieldDefinition("Code", ogr.OFTInteger), FieldDefinition("Capital", ogr.OFTString)]
                LayersWriter(None, [ogr.wkbPolygon, srs0, fields0], [ogr.wkbPoint, srs0, fields1])
                LayersWriter('', [ogr.wkbPolygon, srs0, fields0], [ogr.wkbPoint, srs0, fields1])
                LayersWriter(D:/tmp/mfile.shp, ['', ogr.wkbPolygon, srs0, fields0])
                LayersWriter(D:/tmp/mfile.gml, ['park', ogr.wkbPolygon, srs0, fields0],
                             ['area', ogr.wkbPoint, srs0, fields1], drivername='GML')

        :param args: layer parameters with at least name and geometry type. Any of the following lists is valid:

            - [name, geometry type]
            - [name, geometry type, spatial reference system]
            - [name, geometry type, spatial reference system, list of girs.feat.layers.FieldDefinition]

        :param kwargs:

            :key source: (str) source name (e.g., file name with extension)
            :key drivername: a short driver name (see :meth:`FeatDrivers.get_driver_names()`). E.g., 'Memory'

        """
        super(LayersWriter, self).__init__()
        source = kwargs.pop('source', '')
        if source is None:
            source = ''
        dr = FeatDrivers.get_driver(source, kwargs.pop('drivername', None))
        self.dataset = dr.CreateDataSource(source)
        if self.dataset is None:
            if os.path.exists(source):
                raise ValueError("Data source {} already exists and will not be overwritten".format(str(source)))
            else:
                raise ValueError("Data source {} could not be created".format(str(source)))

        for parameters in args:
            n = len(parameters)
            name = parameters[0] if parameters is not None else ''
            geom = parameters[1]
            prj = parameters[2] if n > 2 else None
            field_definitions0 = parameters[3] if n > 3 else None
            self.create_layer(name, prj, geom, field_definitions0)


def delete_layers(source):
    """Delete layers defined in `source`

    Example::

        delete_layers('D:/tmp/country.shp')

    :param source: a data source, typically a file name (full path)
    :type source: str
    :return:
    """
    if os.path.exists(source):
        driver = FeatDrivers.get_driver(source=source)
        driver.DeleteDataSource(source)


def _get_unique_field_names(field_names):
    """Append a digit to fieldname duplicates
    :param field_names:
    :return: list of field names without duplicates
    """
    import collections
    repeated_fieldnames = {c: -2 for c, count in list(collections.Counter(field_names).items()) if count > 1}
    new_filed_names = list()
    for field_name in field_names:
        if field_name in repeated_fieldnames:
            repeated_fieldnames[field_name] += 1
            if repeated_fieldnames[field_name] >= 0:
                field_name += str(repeated_fieldnames[field_name])
        new_filed_names.append(field_name)
    return new_filed_names


def data_frame_to_layer(df, target=''):
    """Return a `LayersWriter` from the pandas DataFrame.

    The columns must have unique names.
    One column must be denominated according to DataFrameFeature.geometry_fieldname '_GEOM_'

    :param df: the DataFrame
    :type df: pandas.DataFrame
    :param target: an object LayerWriter, a file name of a new LayerWriter, or a dataset from a LayerWriter (writable)
    :type target: LayerWriter, str, or OGRDataSet
    :return: LayerWriter instance
    """

    # Check unique column names
    column_names = df.columns.tolist()
    if len(column_names) != len(set(column_names)):
        msg = 'Non unique column names: {}'.format(
            set([str(c) for c in column_names if column_names.count(c) > 1]))
        raise RuntimeError(msg)

    field_definitions0 = get_field_definitions_from_data_frame(df)
    df_geom_field_definition = [fd for fd in field_definitions0 if fd.name == DataFrameFeature.geometry_fieldname]
    if len(df_geom_field_definition) != 1:
        msg = 'No geometry field found' if len(df_geom_field_definition) == 0 else 'Geometry field not unique'
        raise RuntimeError(msg)
    else:
        df_geom_field_definition = df_geom_field_definition[0]

    # The geometry column may have one and the same layer in all records, as well as no duplicate FID inside the
    # columns' objects DataFrameGeometry
    sr_geom = df[df_geom_field_definition.name]
    if len(set([geom.get_layer() for geom in sr_geom])) != 1:
        msg = 'DataFrame contains different layers: {}'.format(set([geom.get_layer() for geom in sr_geom]))
        raise RuntimeError(msg)
    if len(set([geom.feature_id for geom in sr_geom])) != len(sr_geom):
        msg = 'DataFrame contains duplicated FIDs'
        raise RuntimeError(msg)

    # Get field names from the layer and from the data frame, deleting the geom column of the data frame
    lyr_in = sr_geom.iloc[0].get_layer()
    target = '' if target is None else target.strip()
    if target and not os.path.exists(os.path.dirname(target)):
        os.makedirs(os.path.dirname(target))
    lrs_out = LayersWriter(source=target)
    lyr_out = lrs_out.create_layer('', lyr_in.GetSpatialRef(), lyr_in.GetGeomType())
    field_definitions1 = list(field_definitions0)
    field_definitions1.remove(df_geom_field_definition)
    lrs_out.add_fields(fields0=field_definitions1)
    ldf_out = lrs_out.get_layer_definition()

    # copy features
    def create_feature(sr):
        g_wkb = sr.loc[df_geom_field_definition.name].get_geometry()
        sr = sr.drop(df_geom_field_definition.name)
        feat_new = ogr.Feature(ldf_out)
        for fn0, value in zip(sr.index, sr):
            feat_new.SetField(fn0, value)
        feat_new.SetGeometry(ogr.CreateGeometryFromWkb(g_wkb))
        lyr_out.CreateFeature(feat_new)
    # This is the slow part:
    df.apply(create_feature, axis=1)
    lrs_out.dataset.FlushCache()
    return lrs_out


# def data_frame_to_layer(df, target=''):
#     """Return a `LayersWriter` from the pandas DataFrame.
#
#     :param df: the DataFrame
#     :type df: pandas.DataFrame
#     :param target: an object LayerWriter, a file name of a new LayerWriter, or a dataset from a LayerWriter (writable)
#     :type target: LayerWriter, str, or OGRDataSet
#     :return: LayerWriter instance
#     """
#     # TODO: if column names in DataFrame are the same as in layers, use layer's field types instead of DataFrame types
#     # if df is None or df.dropna().empty:
#     #     return None
#
#     # Get column names repetitions
#     df.columns = _get_unique_field_names(df.columns.tolist())
#
#     field_definitions0 = get_field_definitions_from_data_frame(df)
#
#     df_field_names = df.columns.tolist()
#     df_geom_field_index_name = [(idx, field_name) for idx, field_name in enumerate(df_field_names)
#                                 if isinstance(df[field_name].iloc[0], DataFrameGeometry)]
#     if not df_geom_field_index_name:
#         msg = 'Column with type DataFrameGeometry not found'
#         raise RuntimeError(msg)
#     df_geom_field_index = df_geom_field_index_name[0][0]
#     df_geom_field_name = df_geom_field_index_name[0][1]
#     # The geometry column may have one and the same layer in all records, as well as no duplicate FID inside the
#     # columns' objects DataFrameGeometry
#
#     df_geom = df[df_geom_field_name]
#     if len(set([geom.get_layer() for geom in df_geom])) != 1:
#         msg = 'DataFrame contains different layers'
#         raise RuntimeError(msg)
#     if len(set([geom.feature_id for geom in df_geom])) != len(df_geom):
#         msg = 'DataFrame contains duplicated FIDs'
#         raise RuntimeError(msg)
#     # Get field names from the layer and from the data frame, deleting the geom column of the data frame
#     geom0 = df_geom.iloc[0]
#     lyr_in = geom0.get_layer()
#     field_names_df = df.columns.values.tolist()
#     del field_names_df[df_geom_field_index]
#     # Check same field type in the column
#     # field_definitions0 = [series_to_field_definition(df[fn]) for fn in field_names_df]
#
#     target = '' if target is None else target.strip()
#     if target and not os.path.exists(os.path.dirname(target)):
#         os.makedirs(os.path.dirname(target))
#     lrs_out = LayersWriter(target)
#     lyr_out = lrs_out.create_layer('', lyr_in.GetSpatialRef(), lyr_in.GetGeomType())
#     lrs_out.add_fields(fields0=field_definitions0)
#     ldf_out = lrs_out.get_layer_definition()
#
#     # copy features
#     def create_feature(sr):
#         g_wkb = sr.loc['GEOM'].get_geometry()
#         sr = sr.drop('GEOM')
#         feat_new = ogr.Feature(ldf_out)
#         for fn0, value in zip(sr.index, sr):
#             feat_new.SetField(fn0, value)
#         feat_new.SetGeometry(ogr.CreateGeometryFromWkb(g_wkb))
#         lyr_out.CreateFeature(feat_new)
#     # This is the slow part:
#     df.apply(create_feature, axis=1)
#     lrs_out.dataset.FlushCache()
#     return lrs_out


# def field_definitions(layer_definition):
#     """Yield field definitions
#
#     :param layer_definition: a layer definition
#     :type layer_definition: ogr.FeatureDefn
#     :return:
#     """
#     for i in range(layer_definition.GetFieldCount()):
#         yield layer_definition.GetFieldDefn(i)


# def series_to_field_definition(sr):
#     """Transform pandas Series name into field definition instances
#
#     Transform numpy types of pandas DataFrame columns into field
#
#     :param sr:
#     :return:
#     """
#     col_types = set([type(v) for v in sr.values])
#     field_name = sr.name
#     if len(col_types) != 1:
#         msg = 'Different types in column {}: {}'.format(field_name, ', '.join(list(col_types)))
#         raise ValueError(msg)
#     oft_type = FieldDefinition.data_frame_to_ogr_type_dict[type(sr.values[0])]
#     fd = FieldDefinition(name=field_name, oft_type=oft_type)
#     return fd
#

def get_field_definitions_from_data_frame(df):
    """Return a list of field definitions to the DataFrame columns.

    Return one instance of FieldDefinition for each column. For the geometry column, return a
    FieldDefinition with name 'DataFrameFeature.geometry_fieldname' and oft_type ogr.OFTString

    :param df: data frame with a geometry column
    :return: list of field definitions
    """
    sr_geom = df[DataFrameFeature.geometry_fieldname]
    geom_col_types = set([type(v) for v in sr_geom.values])
    if len(geom_col_types) != 1:
        msg = 'Different types in the geometry column: {}'.format(', '.join(list(geom_col_types)))
        raise ValueError(msg)
    data_frame_geom = sr_geom.values[0]
    data_frame_layer = data_frame_geom.data_frame_layer
    lyr = data_frame_layer.layer
    ldf = lyr.GetLayerDefn()
    layer_field_names = list()
    for i in range(ldf.GetFieldCount()):
        fd = ldf.GetFieldDefn(i)
        layer_field_names.append(fd.GetName())

    result = list()
    for field_name_df in df.columns:
        sr = df[field_name_df]
        col_types = set([type(v) for v in sr.values])
        if len(col_types) != 1:
            msg = 'Different types in column {}: {}'.format(field_name_df, ', '.join(list(col_types)))
            raise ValueError(msg)
        idx_layer = [idx for idx, field_name_lyr in enumerate(layer_field_names) if field_name_lyr == field_name_df]
        if idx_layer:
            fd = ldf.GetFieldDefn(idx_layer[0])
            result.append(FieldDefinition.from_ogr(fd))
        else:
            oft_type = FieldDefinition.numpy2oft(type(sr.values[0]))
            result.append(FieldDefinition(name=field_name_df, oft_type=oft_type))
    return result

