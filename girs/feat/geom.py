"""
the module ``girs.feat.geom`` creates geometries from strings and return them as an OGRGeometry object
"""
from __future__ import print_function
from builtins import range
from osgeo import ogr
from collections import OrderedDict

"""
OGRwkbGeometryType {
  wkbUnknown = 0, wkbPoint = 1, wkbLineString = 2, wkbPolygon = 3,
  wkbMultiPoint = 4, wkbMultiLineString = 5, wkbMultiPolygon = 6, wkbGeometryCollection = 7,
  wkbCircularString = 8, wkbCompoundCurve = 9, wkbCurvePolygon = 10, wkbMultiCurve = 11,
  wkbMultiSurface = 12, wkbCurve = 13, wkbSurface = 14, wkbPolyhedralSurface = 15,
  wkbTIN = 16, wkbTriangle = 17, wkbNone = 100, wkbLinearRing = 101,
  wkbCircularStringZ = 1008, wkbCompoundCurveZ = 1009, wkbCurvePolygonZ = 1010, wkbMultiCurveZ = 1011,
  wkbMultiSurfaceZ = 1012, wkbCurveZ = 1013, wkbSurfaceZ = 1014, wkbPolyhedralSurfaceZ = 1015,
  wkbTINZ = 1016, wkbTriangleZ = 1017, wkbPointM = 2001, wkbLineStringM = 2002,
  wkbPolygonM = 2003, wkbMultiPointM = 2004, wkbMultiLineStringM = 2005, wkbMultiPolygonM = 2006,
  wkbGeometryCollectionM = 2007, wkbCircularStringM = 2008, wkbCompoundCurveM = 2009, wkbCurvePolygonM = 2010,
  wkbMultiCurveM = 2011, wkbMultiSurfaceM = 2012, wkbCurveM = 2013, wkbSurfaceM = 2014,
  wkbPolyhedralSurfaceM = 2015, wkbTINM = 2016, wkbTriangleM = 2017, wkbPointZM = 3001,
  wkbLineStringZM = 3002, wkbPolygonZM = 3003, wkbMultiPointZM = 3004, wkbMultiLineStringZM = 3005,
  wkbMultiPolygonZM = 3006, wkbGeometryCollectionZM = 3007, wkbCircularStringZM = 3008, wkbCompoundCurveZM = 3009,
  wkbCurvePolygonZM = 3010, wkbMultiCurveZM = 3011, wkbMultiSurfaceZM = 3012, wkbCurveZM = 3013,
  wkbSurfaceZM = 3014, wkbPolyhedralSurfaceZM = 3015, wkbTINZM = 3016, wkbTriangleZM = 3017,
  wkbPoint25D = 0x80000001, wkbLineString25D = 0x80000002, wkbPolygon25D = 0x80000003, wkbMultiPoint25D = 0x80000004,
  wkbMultiLineString25D = 0x80000005, wkbMultiPolygon25D = 0x80000006, wkbGeometryCollection25D = 0x80000007
}"""

topology_0D = [ogr.wkbPoint, ogr.wkbPoint25D, # ogr.wkbPoint, ogr.wkbPointZM,
               ogr.wkbMultiPoint, ogr.wkbMultiPoint25D, # ogr.wkbMultiPointM, ogr.wkbMultiPointZM
               ]


topology_1D = [ogr.wkbLineString, ogr.wkbLineString25D, # ogr.wkbLineStringM, ogr.wkbLineStringZM,
               ogr.wkbMultiLineString, ogr.wkbMultiLineString25D, # ogr.wkbMultiLineStringM, ogr.wkbMultiLineStringZM,
               ogr.wkbCircularString, ogr.wkbCircularStringZ, # ogr.wkbCircularStringM, ogr.wkbCircularStringZM,
               # ogr.wkbCurve, ogr.wkbCurveZ, ogr.wkbCurveM, # ogr.wkbCurveZM,
               ogr.wkbMultiCurve, ogr.wkbMultiCurveZ, # ogr.wkbMultiCurveM, ogr.wkbMultiCurveZM,
               ogr.wkbCompoundCurve, ogr.wkbCompoundCurveZ, # ogr.wkbCompoundCurveM, ogr.wkbCompoundCurveZM,
               ogr.wkbLinearRing
               ]


topology_2D = [ogr.wkbPolygon, ogr.wkbPolygon25D, # ogr.wkbPolygonM, ogr.wkbPolygonZM,
               ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D, # ogr.wkbMultiPolygonM, ogr.wkbMultiPolygonZM,
               ogr.wkbCurvePolygon, ogr.wkbCurvePolygonZ, # ogr.wkbCurvePolygonM, ogr.wkbCurvePolygonZM,
               # ogr.wkbSurface, ogr.wkbSurfaceZ, # ogr.wkbSurfaceM, ogr.wkbSurfaceZM,
               ogr.wkbMultiSurface, ogr.wkbMultiSurfaceZ, # ogr.wkbMultiSurfaceM, ogr.wkbMultiSurfaceZM,
               # ogr.wkbPolyhedralSurface, ogr.wkbPolyhedralSurfaceZ, # ogr.wkbPolyhedralSurfaceM, ogr.wkbPolyhedralSurfaceZM,
               # ogr.wkbTIN, ogr.wkbTINZ, # ogr.wkbTINM, ogr.wkbTINZM
               ]

geometry_to_geometry_collection_dict = {
    ogr.wkbPoint: ogr.wkbMultiPoint,
    ogr.wkbPoint25D: ogr.wkbMultiPoint25D,
    # ogr.wkbPointM: ogr.wkbMultiPointM,
    # ogr.wkbPointZM: ogr.wkbMultiPointZM,

    ogr.wkbLineString: ogr.wkbMultiLineString,
    ogr.wkbLineString25D: ogr.wkbMultiLineString25D,
    # ogr.wkbLineStringM: ogr.wkbMultiLineStringM,
    # ogr.wkbLineStringZM: ogr.wkbMultiLineStringZM,

    # ogr.wkbCurve: ogr.wkbMultiCurve,
    # ogr.wkbCurveM: ogr.wkbMultiCurveM,
    # ogr.wkbCurveZ: ogr.wkbMultiCurveZ,
    # ogr.wkbCurveZM: ogr.wkbMultiCurveZM,

    ogr.wkbPolygon: ogr.wkbMultiPolygon,
    ogr.wkbPolygon25D: ogr.wkbMultiPolygon25D,
    # ogr.wkbPolygonM: ogr.wkbMultiPolygonM,
    # ogr.wkbPolygonZM: ogr.wkbMultiPolygonZM,

    # ogr.wkbSurface: ogr.wkbMultiSurface,
    # ogr.wkbSurfaceM: ogr.wkbMultiSurfaceM,
    # ogr.wkbSurfaceZ: ogr.wkbMultiSurfaceZ,
    # ogr.wkbSurfaceZM: ogr.wkbMultiSurfaceZM,
}


geometry_collection_to_geometry_dict = {
    ogr.wkbMultiPoint: ogr.wkbPoint,
    ogr.wkbMultiPoint25D: ogr.wkbPoint25D,
    # ogr.wkbMultiPointM: ogr.wkbPointM,
    # ogr.wkbMultiPointZM: ogr.wkbPointZM,

    ogr.wkbMultiLineString: ogr.wkbLineString,
    ogr.wkbMultiLineString25D: ogr.wkbLineString25D,
    # ogr.wkbMultiLineStringM: ogr.wkbLineStringM,
    # ogr.wkbMultiLineStringZM: ogr.wkbLineStringZM,

    # ogr.wkbMultiCurve: ogr.wkbCurve,
    # ogr.wkbMultiCurveM: ogr.wkbCurveM,
    # ogr.wkbMultiCurveZ: ogr.wkbCurveZ,
    # ogr.wkbMultiCurveZM: ogr.wkbCurveZM,

    ogr.wkbMultiPolygon: ogr.wkbPolygon,
    ogr.wkbMultiPolygon25D: ogr.wkbPolygon25D,
    # ogr.wkbMultiPolygonM: ogr.wkbPolygonM,
    # ogr.wkbMultiPolygonZM: ogr.wkbPolygonZM,

    # ogr.wkbMultiSurface: ogr.wkbSurface,
    # ogr.wkbMultiSurfaceM: ogr.wkbSurfaceM,
    # ogr.wkbMultiSurfaceZ: ogr.wkbSurfaceZ,
    # ogr.wkbMultiSurfaceZM: ogr.wkbSurfaceZM,
}


# =============================================================================
# Point
# =============================================================================
def create_point(*args):
    """

    :param args:
    :return:
    """
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint_2D(*args)
    return point


def create_point_z(*args):
    """

    :param args:
    :return:
    """
    point = ogr.Geometry(ogr.wkbPoint)
    point.AddPoint(*args)
    return point


def create_point_25d(*args):
    """

    :param args:
    :return:
    """
    point = ogr.Geometry(ogr.wkbPoint25D)
    point.AddPoint(*args)
    return point


def create_point_m(*args):
    """

    :param args:
    :return:
    """
    point = ogr.Geometry(ogr.wkbPointM)
    point.AddPointM(*args)
    return point


def create_point_zm(*args):
    """

    :param args:
    :return:
    """
    point = ogr.Geometry(ogr.wkbPointZM)
    point.AddPointZM(*args)
    return point


# =============================================================================
# MultiPoint
# =============================================================================
def create_multi_point(points):
    """

    :param points:
    :return:
    """
    multipoint = ogr.Geometry(ogr.wkbMultiPoint)
    for point in points:
        p = ogr.Geometry(ogr.wkbPoint)
        p.AddPoint(*point)
        multipoint.AddGeometry(p)
    return multipoint


def create_multi_point_25d(points):
    """

    :param points:
    :return:
    """
    multipoint = ogr.Geometry(ogr.wkbMultiPoint25D)
    for point in points:
        p = ogr.Geometry(ogr.wkbPoint25D)
        p.AddPoint(*point)
        multipoint.AddGeometry(p)
    return multipoint


def create_multi_point_m(points):
    """

    :param points:
    :return:
    """
    multipoint = ogr.Geometry(ogr.wkbMultiPointM)
    for point in points:
        p = ogr.Geometry(ogr.wkbPointM)
        p.AddPointM(*point)
        multipoint.AddGeometry(p)
    return multipoint


def create_multi_point_zm(points):
    """

    :param points:
    :return:
    """
    multipoint = ogr.Geometry(ogr.wkbMultiPointZM)
    for point in points:
        p = ogr.Geometry(ogr.wkbPointZM)
        p.AddPointZM(*point)
        multipoint.AddGeometry(p)
    return multipoint


# =============================================================================
# LineString
# =============================================================================
def create_line_string(line_string):
    """

    :param line_string:
    :return:
    """
    line = ogr.Geometry(ogr.wkbLineString)
    for p in line_string:
        line.AddPoint_2D(*p)
    return line


def create_line_string_25d(line_string):
    """

    :param line_string:
    :return:
    """
    line = ogr.Geometry(ogr.wkbLineString25D)
    for p in line_string:
        line.AddPoint(*p)
    return line


def create_line_string_m(line_string):
    """

    :param line_string:
    :return:
    """
    line = ogr.Geometry(ogr.wkbLineStringM)
    for p in line_string:
        line.AddPointM(*p)
    return line


def create_line_string_zm(line_string):
    """

    :param line_string:
    :return:
    """
    line = ogr.Geometry(ogr.wkbLineStringZM)
    for p in line_string:
        line.AddPointZM(*p)
    return line


# =============================================================================
# MultiLineString
# =============================================================================
def create_multiline_string(line_strings):
    """

    :param line_strings:
    :return:
    """
    multiline = ogr.Geometry(ogr.wkbMultiLineString)
    for line_string in line_strings:
        multiline.AddGeometry(create_line_string(line_string))
    return multiline


def create_multiline_string_25d(line_strings):
    """

    :param line_strings:
    :return:
    """
    multiline = ogr.Geometry(ogr.wkbMultiLineString25D)
    for line_string in line_strings:
        multiline.AddGeometry(create_line_string_25d(line_string))
    return multiline


def create_multiline_string_m(line_strings):
    """

    :param line_strings:
    :return:
    """
    multiline = ogr.Geometry(ogr.wkbMultiLineStringM)
    for line_string in line_strings:
        multiline.AddGeometry(create_line_string_m(line_string))
    return multiline


def create_multiline_string_zm(line_strings):
    """

    :param line_strings:
    :return:
    """
    multiline = ogr.Geometry(ogr.wkbMultiLineStringZM)
    for line_string in line_strings:
        multiline.AddGeometry(create_line_string_zm(line_string))
    return multiline


# =============================================================================
# LineString
# =============================================================================
def create_linear_ring(points):
    """

    :param points:
    :return:
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    if points[0] != points[-1]:
        points.append(points[0])
    for p in points:
        ring.AddPoint(*p)
    return ring


def create_linear_ring_2d(points):
    """

    :param points:
    :return:
    """
    return create_linear_ring(points)


def create_linear_ring_m(points):
    """

    :param points:
    :return:
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    if points[0] != points[-1]:
        points.append(points[0])
    for p in points:
        ring.AddPointM(*p)
    return ring


def create_linear_ring_zm(points):
    """

    :param points:
    :return:
    """
    ring = ogr.Geometry(ogr.wkbLinearRing)
    if points[0] != points[-1]:
        points.append(points[0])
    for p in points:
        ring.AddPointZM(*p)
    return ring


# =============================================================================
# Polygon
# =============================================================================
def create_polygon(rings):
    """

    :param rings:
    :return:
    """
    poly = ogr.Geometry(ogr.wkbPolygon)
    try:
        for r in rings:
            poly.AddGeometry(create_linear_ring(r))
    except (TypeError, AttributeError):
        poly.AddGeometry(create_linear_ring(rings))
    return poly


def create_polygon_25d(rings):
    """

    :param rings:
    :return:
    """
    poly = ogr.Geometry(ogr.wkbPolygon25D)
    try:
        for r in rings:
            poly.AddGeometry(create_linear_ring_2d(r))
    except (TypeError, AttributeError):
        poly.AddGeometry(create_linear_ring_2d(rings))
    return poly


def create_polygon_m(rings):
    """

    :param rings:
    :return:
    """
    poly = ogr.Geometry(ogr.wkbPolygonM)
    try:
        for r in rings:
            poly.AddGeometry(create_linear_ring_m(r))
    except (TypeError, AttributeError):
        poly.AddGeometry(create_linear_ring_m(rings))
    return poly


def create_polygon_zm(rings):
    """

    :param rings:
    :return:
    """
    poly = ogr.Geometry(ogr.wkbPolygonZM)
    try:
        for r in rings:
            poly.AddGeometry(create_linear_ring_zm(r))
    except (TypeError, AttributeError):
        poly.AddGeometry(create_linear_ring_zm(rings))
    return poly


# =============================================================================
# Polygon
# =============================================================================
def create_multipolygon(polygons):
    """

    :param polygons:
    :return:
    """
    multi_polygons = ogr.Geometry(ogr.wkbMultiPolygon)
    try:
        for rings in polygons:
            multi_polygons.AddGeometry(create_polygon(rings))
    except (TypeError, AttributeError):
        multi_polygons.AddGeometry(create_polygon(polygons))
    return multi_polygons


def create_multipolygon_25d(polygons):
    """

    :param polygons:
    :return:
    """
    multi_polygons = ogr.Geometry(ogr.wkbMultiPolygon25D)
    try:
        for rings in polygons:
            multi_polygons.AddGeometry(create_polygon_25d(rings))
    except (TypeError, AttributeError):
        multi_polygons.AddGeometry(create_polygon_25d(polygons))
    return multi_polygons


def create_multipolygon_m(polygons):
    """

    :param polygons:
    :return:
    """
    multi_polygons = ogr.Geometry(ogr.wkbMultiPolygonM)
    try:
        for rings in polygons:
            multi_polygons.AddGeometry(create_polygon_m(rings))
    except (TypeError, AttributeError):
        multi_polygons.AddGeometry(create_polygon_m(polygons))
    return multi_polygons


def create_multipolygon_zm(polygons):
    """

    :param polygons:
    :return:
    """
    multi_polygons = ogr.Geometry(ogr.wkbMultiPolygonZM)
    try:
        for rings in polygons:
            multi_polygons.AddGeometry(create_polygon_zm(rings))
    except (TypeError, AttributeError):
        multi_polygons.AddGeometry(create_polygon_zm(polygons))
    return multi_polygons


def is_topology_2d(geo_type):
    """Check whether the geometry type has 2D topology

    Polygons are: ogr.wkbPolygon, ogr.wkbPolygon25D, ogr.wkbPolygonM, ogr.wkbPolygonZM,
                  ogr.wkbMultiPolygon, ogr.wkbMultiPolygon25D, ogr.wkbMultiPolygonM, ogr.wkbMultiPolygonZM

    :param geo_type: ogr.wkbXXX
    :return: True if the geometry is a wkb polygon
    """
    return geo_type in topology_2D


def is_topology_1d(geo_type):
    """Check whether the geometry type has 1D topology

    Points are:

        - ogr.wkbLineString
        - ogr.wkbLineString25D
        - ogr.wkbLineStringM
        - ogr.wkbLineStringZM

    :param geo_type:
    :return: True if the geometry is a point
    """
    return geo_type in topology_1D


def is_topology_0d(geo_type):
    """Check whether the geometry type has 0D topology

    Points are:

        - ogr.wkbPoint
        - ogr.wkbPoint25D
        - ogr.wkbMultiPoint
        - ogr.wkbMultiPoint25D

    :param geo_type:
    :return: True if the geometry is a point
    """
    return geo_type in topology_0D


def is_geometry_collection(geo_type):
    return ogr.GeometryTypeToName(geo_type)[:5].upper() == 'MULTI'


def geometries_to_geometry_collection(geom_list):
    """Merge geometries

    :param geom_list:
    :return:
    """
    geom_col = ogr.Geometry(geometry_to_geometry_collection_type(geom_list[0].GetGeometryType()))
    for geom in geom_list:
        if 'multi' in geom.GetGeometryName().lower():
            for g_part in geom:
                geom_col.AddGeometry(g_part)
        else:
            geom_col.AddGeometry(geom)
    return geom_col


def geometry_collection_to_geometries(geom_col_wkb):
    """Return a list of geometries

    :param geom_col_wkb:
    :return:
    """
    geom_col = ogr.CreateGeometryFromWkb(geom_col_wkb)
    if geom_col.GetGeometryCount() > 0:
        return [g_part.ExportToWkb() for g_part in geom_col]
    else:
        return [geom_col_wkb]


def snap_to_grid(geom, ndigits=8):
    """2D and not measured

    :param geom:
    :param ndigits:
    :return:
    """
    n = geom.GetGeometryCount()
    is_geom_collection = n > 1
    geom_type = geom.GetGeometryType()
    geom_name = geom.GetGeometryName()
    if geom_name == 'LINEARRING':
        result = ogr.Geometry(ogr.wkbLinearRing)
    elif geom_name == 'POLYGON':
        result = ogr.Geometry(ogr.wkbPolygon)
        is_geom_collection = True
    elif geom_name == 'MULTIPOLYGON':
        result = ogr.Geometry(ogr.wkbMultiPolygon)
        is_geom_collection = True
    elif geom_name == 'MULTIPOINT':
        result = ogr.Geometry(ogr.wkbMultiPoint)
        is_geom_collection = True
    else:
        result = ogr.Geometry(geom_type)
    if is_geom_collection:
        for i in range(n):
            geom2 = snap_to_grid(geom.GetGeometryRef(i), ndigits)
            result.AddGeometry(geom2)
    else:
        r = list(range(geom.GetCoordinateDimension()))
        for i in range(0, geom.GetPointCount()):
            pt = geom.GetPoint(i)
            result.AddPoint_2D(*[round(pt[j], ndigits) for j in r])
    return result


def geometry_to_geometry_collection_type(geo_type):
    try:
        return geometry_to_geometry_collection_dict[geo_type]
    except KeyError:
        return geo_type


def geometry_collection_to_geometry_type(geo_type):
    try:
        return geometry_collection_to_geometry_dict[geo_type]
    except KeyError:
        return geo_type


def join_linestrings0(linestrings_wkb):
    """

    :param linestrings_wkb:
    :return:
    """
    linestrings = [ogr.CreateGeometryFromWkb(g) for g in linestrings_wkb]
    linestrings = [(g, g.GetPointCount()) for g in linestrings]
    points = [(g.GetPoint(0), g.GetPoint(n-1)) for g, n in linestrings]
    points_dict = {}
    for i, p in enumerate(points):
        p0, p1 = p[0], p[1]
        if p0 not in points_dict:
            points_dict[p0] = [(i, 0)]
        else:
            points_dict[p0].append((i, 0))
        if p1 not in points_dict:
            points_dict[p1] = [(i, -1)]
        else:
            points_dict[p1].append((i, -1))
    result = list()
    for p, ls_list in list(points_dict.items()):
        if len(ls_list) == 2:
            linestring0 = linestrings[ls_list[0][0]]
            linestring1 = linestrings[ls_list[1][0]]
            points0 = linestring0[0].GetPoints()
            points1 = linestring1[0].GetPoints()
            pos0 = ls_list[0][1]
            pos1 = ls_list[1][1]
            if pos0 == pos1:
                points1 = points1[::-1]
                pos1 = -1 if pos1 == 0 else 0
            if pos0 == 1:
                points0 = points0 + points1
            else:
                points0 = points1 + points0

            print(p, pos0, pos1, points0, points1)
            print(p, points0[pos0], points1[pos1])
            return
        else:
            for ls in ls_list:
                result.append(linestrings_wkb[ls[0]])


def join_linestrings2(linestrings_wkb):
    """

    :param linestrings_wkb:
    :return:
    """
    linestrings = [ogr.CreateGeometryFromWkb(g) for g in linestrings_wkb]
    linestrings = [(g, g.GetPointCount()) for g in linestrings]
    en_dict = {g: (g.GetPoint(0), g.GetPoint(n - 1)) for ils, (g, n) in enumerate(linestrings)}
    ne_dict = {}
    for g, (p0, p1) in list(en_dict.items()):
        if p0 not in ne_dict:
            ne_dict[p0] = [(g, 0)]
        else:
            ne_dict[p0].append((g, 0))
        if p1 not in ne_dict:
            ne_dict[p1] = [(g, -1)]
        else:
            ne_dict[p1].append((g, -1))
    found = True
    while found:
        for p, v in list(ne_dict.items()):
            if len(v) == 2:
                found = True
                g0, g1 = v[0][0], v[1][0]
                pos0, pos1 = v[0][1], v[1][1]
                points0 = g0.GetPoints()
                points1 = g1.GetPoints()
                if pos0 == pos1:
                    points1 = points1[::-1]
                points0 = points0 + points1 if pos0 == 1 else points1 + points0
                geom_new = ogr.Geometry(g0.GetGeometryType())
                for p in points0:
                    geom_new.AddPoint(*p)
                p00, p01 = en_dict[g0]
                p10, p11 = en_dict[g1]
                ne_dict[p00].remove[g0]
                ne_dict[p01].remove[g0]
                ne_dict[p10].remove[g1]
                ne_dict[p11].remove[g1]
                p0 = geom_new.GetPoint(0)
                p1 = geom_new.GetPoint(geom_new.GetPointCount() - 1)
                ne_dict[p0].append[geom_new]
                if not ne_dict[p00]:
                    del ne_dict[p00]
                if not ne_dict[p01]:
                    del ne_dict[p01]
                if not ne_dict[p10]:
                    del ne_dict[p10]
                if not ne_dict[p11]:
                    del ne_dict[p11]

                en_dict[geom_new] = [p0, p1]

                del en_dict[g0]
                del en_dict[g1]


def merge_edges(nd, n0, en_ordered_dict, ne_dict):
    """Merge both edges conencted to the point (node) n0

    :param nd: geometry dimension
    :param n0: point
    :param en_ordered_dict: edge-nodes dictionary
    :param ne_dict: node-edges dictionary
    :return:
    """
    edge0, edge1 = ne_dict[n0]
    points0 = edge0.GetPoints()  # return 3D-points
    points1 = edge1.GetPoints()  # return 3D-points

    p0_start = points0[0][:nd]
    p0_stop = points0[-1][:nd]
    p1_start = points1[0][:nd]
    p1_stop = points1[-1][:nd]

    assert (n0 == p0_start or n0 == p0_stop)
    assert (n0 == p1_start or n0 == p1_stop)

    if p0_stop == p1_start:
        points0 = points0 + points1[1:]
    elif p0_stop == p1_stop:
        points0 = points0 + points1[-2::-1]
    elif p0_start == p1_start:
        points0 = points0[:0:-1] + points1
    elif p0_start == p1_stop:
        points0 = points0[::-1] + points1[-2::-1]

    edge = ogr.Geometry(edge0.GetGeometryType())
    for p in points0:
        edge.AddPoint(*p[:nd])

    if p0_start in ne_dict and edge0 in ne_dict[p0_start]:
        ne_dict[p0_start].remove(edge0)
    if p0_stop in ne_dict and edge0 in ne_dict[p0_stop]:
        ne_dict[p0_stop].remove(edge0)
    if p1_start in ne_dict and edge1 in ne_dict[p1_start]:
        ne_dict[p1_start].remove(edge1)
    if p1_stop in ne_dict and edge1 in ne_dict[p1_stop]:
        ne_dict[p1_stop].remove(edge1)

    p0, p1 = edge.GetPoint(0)[:nd], edge.GetPoint(edge.GetPointCount() - 1)[:nd]
    ne_dict[p0].append(edge)
    ne_dict[p1].append(edge)

    if p0_start in ne_dict and not ne_dict[p0_start]:
        del ne_dict[p0_start]
    if p0_stop in ne_dict and not ne_dict[p0_stop]:
        del ne_dict[p0_stop]
    if p1_start in ne_dict and not ne_dict[p1_start]:
        del ne_dict[p1_start]
    if p1_stop in ne_dict and not ne_dict[p1_stop]:
        del ne_dict[p1_stop]

    en_ordered_dict[edge] = (p0, p1)
    if edge0 in en_ordered_dict:
        del en_ordered_dict[edge0]
    if edge1 in en_ordered_dict:
        del en_ordered_dict[edge1]


def join_linestrings(linestrings_wkb):
    """Join linestrings

    Join each pair of linestrings if they and only they are connected by a common point

    :param list of linestrings_wkb:
    :return: list of joined linestrings
    :rtype: list of linestrings_wkb
    """
    # build edge-node dictionary
    linestrings = [ogr.CreateGeometryFromWkb(g) for g in linestrings_wkb]
    nd = linestrings[0].CoordinateDimension()
    linestrings = [(g, (g.GetPoint(0)[:nd], g.GetPoint(g.GetPointCount() - 1)[:nd])) for ils, g in enumerate(linestrings)]
    en_ordered_dict = OrderedDict(linestrings)
    # build node-edge dictionary
    ne_dict = {}
    for g, (p0, p1) in list(en_ordered_dict.items()):
        if p0 not in ne_dict:
            ne_dict[p0] = [g]
        else:
            ne_dict[p0].append(g)
        if p1 not in ne_dict:
            ne_dict[p1] = [g]
        else:
            ne_dict[p1].append(g)
    found = True
    result = list()
    while en_ordered_dict and found:
        edge, (n0, n1) = en_ordered_dict.popitem(last=False)
        if len(ne_dict[n0]) == 2:
            found = True
            merge_edges(nd, n0, en_ordered_dict, ne_dict)
        elif len(ne_dict[n1]) == 2:
            found = True
            merge_edges(nd, n1, en_ordered_dict, ne_dict)
        else:
            found = False
            result.append(edge)
    return [g.ExportToWkb() for g in result]

