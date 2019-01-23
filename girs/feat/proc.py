from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import str
from builtins import zip
from builtins import range
from past.builtins import basestring
from past.utils import old_div
from builtins import object
import os
import time
from collections import OrderedDict
import numpy as np
import operator
import multiprocessing as mp
from osgeo import ogr
from scipy import spatial
from scipy.spatial.qhull import Voronoi
from girs.feat.layers import LayersReader, LayersWriter, LayersUpdate, delete_layers, FieldDefinition
from girs.feat.geom import is_geometry_collection, geometry_to_geometry_collection_type, \
    geometry_collection_to_geometry_type, geometry_collection_to_geometries, geometries_to_geometry_collection, \
    is_topology_1d, join_linestrings
from girs.srs import srs_from_wkt


def union_cascade(geometries_wkb, cpus=1):
    """Union cascade of a geometries' list
    It is assumed that the geometries have the same topological dimension but can be mixed collections ("MULTI") and
    non-collections.

    :param geometries_wkb: list of wkb-geometries
    :param as_collection: True/False
    :param cpus:
    :return:
    """
    uq_geometries_wkb = []
    for geom_wkb in geometries_wkb:
        geom = ogr.CreateGeometryFromWkb(geom_wkb)
        if is_geometry_collection(geom.GetGeometryType()):
            for geom_part in geom:
                uq_geometries_wkb.append([geom_part.ExportToWkb(), geom_part.GetEnvelope()])
        else:
            uq_geometries_wkb.append([geom_wkb, geom.GetEnvelope()])

    available_cpu_count = mp.cpu_count() if cpus > 1 else 1
    return UnionQuadtree(uq_geometries_wkb, '', cpus, available_cpu_count).get_geometry()


def _union_quadtree(geometries, level, cpu, max_cpu):
    """

    :param geometries:
    :param level:
    :param cpu:
    :param max_cpu:
    :return:
    """
    glock.acquire()
    time.sleep(0.1)
    glock.release()
    return UnionQuadtree(geometries, level, cpu, max_cpu).get_geometry()


def _init(lock):
    """

    :param lock:
    :return:
    """
    global glock
    glock = lock


def _parallelize_union(level, glist, requested_cpu_count, available_cpu_count):
    """

    :param level:
    :param glist:
    :param requested_cpu_count:
    :param available_cpu_count:
    :return:
    """
    geometries = []
    processes = min(requested_cpu_count, len(glist))
    mp.freeze_support()
    lock = mp.Manager().Lock()
    pool = mp.Pool(processes=processes, initializer=_init, initargs=(lock,))
    for i, g in enumerate(glist):
        def collect_results(result):
            geometries.append(result)
        level_i = level + str(i)
        available_cpu_count -= 1
        pool.apply_async(_union_quadtree, (g, level_i, requested_cpu_count, available_cpu_count),
                         callback=collect_results)
    pool.close()
    pool.join()
    return geometries


class UnionQuadtree(object):
    """

    """

    threshold = 500

    def __init__(self, geometries_wkb, level='', requested_cpu_count=1, available_cpu_count=1):
        """

        :param geometries_wkb: list of [geometry, extension], all non-collections
        :param level:
        :param requested_cpu_count:
        :param available_cpu_count:
        """
        requested_cpu_count = min(requested_cpu_count, available_cpu_count)
        try:
            env = list(zip(*list(zip(*geometries_wkb))[1]))
        except IndexError as e:
            raise e
        self.center = ((max(env[1]) + min(env[0])) * 0.5, (max(env[3]) + min(env[2])) * 0.5)
        del env
        if len(geometries_wkb) > UnionQuadtree.threshold and len(level) < 4:
            if requested_cpu_count > 1:
                geom_list = list()
                for i in range(4):
                    geometries0, geometries_wkb = self.filter_geometries(geometries_wkb, i)
                    if geometries0:
                        geom_list.append(geometries0)
                geom_list = _parallelize_union(level, geom_list, requested_cpu_count, available_cpu_count)
            else:
                geom_list = list()
                for i in range(4):
                    geometries0, geometries_wkb = self.filter_geometries(geometries_wkb, i)
                    if geometries0:
                        geom_list.append(UnionQuadtree(geometries0, level + str(i)).geometry)
                    del geometries0
            geom_list = [g for g in geom_list if g is not None]
        else:
            geom_list = [geom_wkb for geom_wkb, _ in geometries_wkb]

        geom_type = ogr.CreateGeometryFromWkb(geom_list[0]).GetGeometryType()
        geom_type = geometry_to_geometry_collection_type(geom_type)
        geom = ogr.Geometry(geom_type)
        for g in geom_list:
            geom1 = ogr.CreateGeometryFromWkb(g)
            geom_name = geom1.GetGeometryName().lower()
            if 'multi' in geom_name:
                for geom_part in geom1:
                    geom.AddGeometry(geom_part)
            else:
                geom.AddGeometry(geom1)
        try:
            self.geometry = geom.UnionCascaded().ExportToWkb()
        except RuntimeError:
            geom1 = None
            for geom_wkb in geom_list:
                geom = ogr.CreateGeometryFromWkb(geom_wkb)
                if not geom1:
                    geom1 = geom
                else:
                    geom1 = geom1.Union(geom)
            self.geometry = geom1.ExportToWkb()

    def get_geometry(self):
        return self.geometry

    def filter_geometries(self, geometries0, quadrant):
        fnc = [lambda cx, cy: cx <  self.center[0] and cy <  self.center[1],
               lambda cx, cy: cx <  self.center[0] and cy >= self.center[1],
               lambda cx, cy: cx >= self.center[0] and cy <  self.center[1],
               lambda cx, cy: cx >= self.center[0] and cy >= self.center[1]]
        filter0 = fnc[quadrant]
        geometries1 = []
        geometries2 = []
        for i in range(len(geometries0)):
            geom_list = geometries0[i]
            env = geom_list[1]
            cx0, cy0 = (env[1] + env[0]) * 0.5, (env[3] + env[2]) * 0.5
            if filter0(cx0, cy0):
                geometries1.append(geom_list)
            else:
                geometries2.append(geom_list)
        return geometries1, geometries2


def idw_weights(xy_in, xy_out, nnn=None, p=2):
    """Return the Shepard (idw) interpolation

    :param xy_in: list of coordinate tuples
    :param xy_out: list of coordinate tuples
    :param nnn: integer, number of nearest neighbors
    :param p:
    :return: z-value or list of z values
    """
    n_out = len(xy_out)
    n_in = len(xy_in)
    w_out = np.zeros(n_in * n_out)
    w_out = w_out.reshape((n_out, n_in))
    nnn = min(nnn, len(xy_in)) if nnn else len(xy_in)
    if nnn == 1:
        w_out[:][0] = 1.0
    else:
        distances, indices = spatial.KDTree(xy_in).query(xy_out, k=nnn)
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if distances[i][0] < 1e-10:  # distances are sorted (nearest first)
                w_out[i][0] = 1.0
            else:
                w = 1.0 / dist**p
                w /= np.sum(w)
                w_out[i] = w
    return w_out


def idw(xy_in, z_in, xy_out, nnn=None, p=2):
    """Return the Shepard (idw) interpolation

    :param xy_in: list of coordinate tuples
    :param z_in: list of values
    :param xy_out: list of coordinate tuples
    :param nnn: integer, number of nearest neighbors
    :param p:
    :return: z-value or list of z values
    """
    nnn = min(nnn, len(xy_in)) if nnn else len(xy_in)
    assert len(xy_in) == len(z_in)
    z_in = np.asarray(z_in)
    distances, indices = spatial.KDTree(xy_in).query(xy_out, k=nnn)
    if z_in.ndim == 1:
        z_out = np.zeros(len(indices))
    else: #z_in.ndim == 2:
        z_out = np.zeros((len(indices), z_in.shape[1]))
    for i, (dist, idx) in enumerate(zip(distances, indices)):
        if nnn == 1:
            wz = z_in[idx]
        elif dist[0] < 1e-10:  # distances are sorted (nearest first)
            wz = z_in[idx[0]]
        else:
            w = 1.0 / dist**p
            w /= np.sum(w)
            wz = np.dot(w, z_in[idx])
        z_out[i] = wz
    return z_out

# def idw(xy_in, z_in, xy_out, nnn=None, p=2):
#     """Return the IDW interpolation
#
#     :param xy_in: list of coordinate tuples
#     :param z_in: list of values
#     :param xy_out: list of coordinate tuples
#     :param nnn: integer, number of nearest neighbors
#     :param p:
#     :return: z-value or list of z values
#     """
#     assert len(xy_in) == len(z_in)
#
#     z_in = np.asarray(z_in)
#     nnn = min(nnn, len(xy_in))
#     distances, indices = spatial.KDTree(xy_in).query(xy_out, k=nnn)
#     try:
#         n_in = len(z_in[0])
#     except TypeError:
#         z_in = [z_in]
#         n_in = 1
#     z_out = np.ndarray(shape=(len(indices), n_in), dtype=float)
#
#     z_out.fill(0.0)
#     for i, (dist, idx) in enumerate(zip(distances, indices)):
#         if nnn == 1:
#             wz = z_in[idx]
#         elif dist[0] < 1e-10:  # distances are sorted (nearest first)
#             wz = z_in[idx[0]]
#         else:
#             w = 1.0 / dist**p
#             w /= np.sum(w)
#             wz = np.dot(w, z_in[idx])
#         z_out[i] = wz
#
#     return z_out if len(xy_out) > 1 else z_out[0]


def dissolve(source, **kwargs):
    """Dissolve features in a layers set.

    Features are dissolved according to the parameter `fields`.
    Remaining fields are defined by the parameter `stats`.
    Both parameters `fields` and `stats` are appended by the layer number.
    Examples::

        lrs1 = LayersReader(shp1())
        # Dissolved by field 'NAME_3', no remaining field
        lrs2 = lrs1.dissolve(fields0='NAME_3')
        # Dissolved by field 'NAME_3', no remaining field, saved to file
        lrs2 = lrs1.dissolve(target='D/tmp/diss.shp', fields0='NAME_3')
        # Dissolved by field 'NAME_3', remaining field 'NAME_2' with statistics 'last'
        lrs2 = lrs1.dissolve(fields0='NAME_3', stats0=[('NAME_2', 'last')])

    The following statistics are possible: `first`, `last`, `count`, `min`, `max`, `sum`, `mean`, `range`, and `std`


    :param source: LayersSet or filename of a LayersSet
    :param kwargs:
        :key cpus: number of cpus to be used (default 1). Hint. cpus <= total number of cpus - 1
        :key target: dataset source. Default is '' (memory). Examples: target='', target='D:tmp/output.shp'
        :key drivername: driver short name. Example: drivername='GML'
        :key as_collection: if True return geometries as collections (MULTIPOINT, MULTIPOLYGON, etc.)
        :key fields: dictionary with layer number as key and a list of field names
        :key stats: dictionary with layer number as key and a list of statistics [(field name, stats, output field name)].

            If the list is defined, field name and stats are obligatory. The resulting output field name will be field name

            Examples:

            * {0: [('NAME3', 'count')]}: results in a count of field 'NAME3' saved under 'NAME3'
            * {0: [('NAME3', 'count', 'NAME3COUNT')]}: count of field 'NAME3' saved under 'NAME3COUNT'
            * {0: [('NAME3', 'count', True)]}: count of field 'NAME3' saved under 'NAME3_COUNT'. The stats name is appended to the field name preceded by underline.
            * {0: [('NAME3', 'first')], [('AREA', 'sum', 'AREAkm2)]}: results in a count of field 'NAME3' saved under 'NAME3' and summed area of the field 'AREA' saved under 'AREAAREAkm2'

    :return: LayersWriter
    :rtype: LayersWriter
    """
    target = kwargs.pop('target', '')
    drivername = kwargs.pop('drivername', None)
    cpus = kwargs.pop('cpus', 1)
    as_collection = kwargs.pop('as_collection', True)

    try:
        lrs_in = LayersReader(source)
    except RuntimeError:
        lrs_in = source
    try:
        lrs_out = LayersWriter(source=target, drivername=drivername)
    except RuntimeError:
        lrs_out = target

    layer_diss_dict = kwargs.pop('fields', {})
    try:
        list(layer_diss_dict.items())
    except AttributeError:
        if isinstance(layer_diss_dict, basestring):
            layer_diss_dict = {0: [layer_diss_dict]}
        else:
            layer_diss_dict = {0: layer_diss_dict}

    layer_stats_dict = kwargs.pop('stats', {})
    try:
        list(layer_stats_dict.keys())
    except AttributeError:
        layer_stats_dict = {0: layer_stats_dict}

    stats_methods_dict = {
        'first': lambda values: values[0],
        'last': lambda values: values[-1],
        'count': lambda values: len(values),
        'min': lambda values: min(values),
        'max': lambda values: max(values),
        'sum': lambda values: sum(values),
        'mean': lambda values: np.mean(values),
        'range': lambda values: np.ptp(values),
        'std': lambda values: np.std(values),
    }

    for ilyr, lyr_in in enumerate(lrs_in.layers()):

        ldf_in = lyr_in.GetLayerDefn()
        fd_in = [ldf_in.GetFieldDefn(i) for i in range(ldf_in.GetFieldCount())]
        fd_ogr_layer_dict = {fd.GetName(): fd for fd in fd_in}

        # =====================================================================
        # Process dissolve fields
        # =====================================================================
        diss_keys_dict = dict()
        if ilyr in layer_diss_dict:
            diss_keys_in = list()
            diss_keys_out = list()
            diss_keys_dict = dict()  # simple bijective map
            for diss_parameter in layer_diss_dict[ilyr]:
                if isinstance(diss_parameter, basestring):
                    diss_parameter = [diss_parameter]
                diss_keys_in.append(diss_parameter[0])
                try:
                    diss_keys_out.append(diss_parameter[1])
                    diss_keys_dict[diss_parameter[0]] = diss_parameter[1]
                    diss_keys_dict[diss_parameter[1]] = diss_parameter[0]
                except IndexError:
                    diss_keys_out.append(diss_parameter[0])
                    diss_keys_dict[diss_parameter[0]] = diss_parameter[0]
            fd_diss = [FieldDefinition.from_ogr(fd_ogr_layer_dict[n]) for n in diss_keys_in if n in fd_ogr_layer_dict]
            for fd in fd_diss:
                fd.name = [diss_keys_out[ik] for ik, name_in in enumerate(diss_keys_in) if name_in == fd.name][0]
        else:
            fd_diss = []

        # =====================================================================
        # Process statistics fields
        # =====================================================================
        stats_keys_in = list()
        stats_keys_dict = dict()
        stats_methods = list()
        fd_stats = list()
        if ilyr in list(layer_stats_dict.keys()):
            stats_keys_out = list()
            stats_keys_dict = dict()
            for stats_parameter in layer_stats_dict[ilyr]:
                if isinstance(stats_parameter, basestring):
                    stats_parameter = [stats_parameter]
                stats_methods.append(stats_parameter[1])
                stats_keys_in.append(stats_parameter[0])
                try:
                    stats_keys_out.append(stats_parameter[2])
                    stats_keys_dict[stats_parameter[0]] = stats_parameter[2]
                    stats_keys_dict[stats_parameter[2]] = stats_parameter[0]
                except IndexError:
                    stats_keys_out.append(stats_parameter[0])
                    stats_keys_dict[stats_parameter[0]] = stats_parameter[0]
            stats_methods = [stats_methods_dict[m] for m in stats_methods]
            # Sort fields according to stats_keys_out
            fd_stats = [FieldDefinition.from_ogr(fd_ogr_layer_dict[n]) for n in stats_keys_in if n in fd_ogr_layer_dict]
            for fd in fd_stats:
                fd.name = [stats_keys_out[ik] for ik, name_in in enumerate(stats_keys_in) if name_in == fd.name][0]
            # If the list has different lengths, a field name was wrong
            if len(fd_stats) != len(stats_keys_in):
                msg = ', '.join([n for n in stats_keys_in if n not in [fd.name for fd in fd_stats]])
                msg = 'Unknown field name: {}'.format(msg)
                raise ValueError(msg)
            # Set output field names for statistics fields

        lyr_out = lrs_out.create_layer('', prj=lyr_in.GetSpatialRef(), geom=lyr_in.GetGeomType())
        for fd in fd_diss:
            lyr_out.CreateField(fd.to_ogr())
        for fd in fd_stats:
            lyr_out.CreateField(fd.to_ogr())

        geometry_dict = dict()
        attributes_dict = dict()
        fd_diss_names = [diss_keys_dict[fd.name] for fd in fd_diss]
        fd_stat_names = [stats_keys_dict[fd.name] for fd in fd_stats]
        idx_diss = [i for i in range(ldf_in.GetFieldCount()) if ldf_in.GetFieldDefn(i).GetName() in fd_diss_names]
        idx_stat = {ldf_in.GetFieldDefn(i).GetName(): i
                    for i in range(ldf_in.GetFieldCount()) if ldf_in.GetFieldDefn(i).GetName() in fd_stat_names}  # TODO
        idx_stat = [idx_stat[name] for name in stats_keys_in]
        lyr_in.ResetReading()
        for i, feat in enumerate(lyr_in):
            geom_wkb = feat.GetGeometryRef()
            if geom_wkb is None:
                continue
            geom_wkb = geom_wkb.ExportToWkb()
            diss_values = tuple([feat.GetField(fid) for fid in idx_diss])
            if diss_values not in geometry_dict:
                geometry_dict[diss_values] = []
            geometry_dict[diss_values].append(geom_wkb)
            if diss_values not in attributes_dict:
                attributes_dict[diss_values] = []
            attributes_dict[diss_values].append([feat.GetField(fid) for fid in idx_stat])
            del feat
        lyr_in.ResetReading()

        ldf_out = lyr_out.GetLayerDefn()
        fds = list(geometry_dict.keys())
        if as_collection:
            ldf_out.SetGeomType(geometry_to_geometry_collection_type(ldf_out.GetGeomType()))
            for fd in fds:
                feature = ogr.Feature(ldf_out)
                geom_wkb = union_cascade(geometry_dict.pop(fd), cpus)
                geom_wkb = ogr.CreateGeometryFromWkb(geom_wkb)
                if not is_geometry_collection(geom_wkb.GetGeometryType()):
                    geom = geometries_to_geometry_collection([geom_wkb])
                    geom_wkb = ogr.CreateGeometryFromWkb(geom.ExportToWkb())
                feature.SetGeometry(geom_wkb)
                idx = 0
                for value in fd:
                    feature.SetField(idx, value)
                    idx += 1
                if stats_methods:
                    stats_values = list()
                    fd_att = list(zip(*attributes_dict[fd]))
                    for i, v in enumerate(fd_att):
                        value = stats_methods[i](v)
                        stats_values.append(value)
                    for i, value in enumerate(stats_values):
                        feature.SetField(idx + i, value)
                lyr_out.CreateFeature(feature)
                del feature
        else:
            ldf_out.SetGeomType(geometry_collection_to_geometry_type(ldf_out.GetGeomType()))
            for fd in fds:
                geom_wkb = union_cascade(geometry_dict.pop(fd), cpus)
                try:
                    geometries = geometry_collection_to_geometries(geom_wkb)
                    if is_topology_1d(ogr.CreateGeometryFromWkb(geometries[0]).GetGeometryType()):
                        # merge linestrings share common points into one linestring
                        geometries = join_linestrings(geometries)
                except RuntimeError:
                    geometries = [geom_wkb]
                for geom_wkb in geometries:
                    feature = ogr.Feature(ldf_out)
                    feature.SetGeometry(ogr.CreateGeometryFromWkb(geom_wkb))
                    idx = 0
                    for value in fd:
                        feature.SetField(idx, value)
                        idx += 1
                    if stats_methods:
                        for i, value in enumerate([stats_methods[i](v) for i, v in enumerate(zip(*attributes_dict[fd]))]):
                            feature.SetField(idx + i, value)
                    lyr_out.CreateFeature(feature)
                    del feature

    lrs_out.dataset.FlushCache()
    return lrs_out


def thiessen(points, thiessen, points_layer_number=0, point_id_field_names=None, envelope=None,
             clip_region=None, clip_region_layer_number=0, expand=0.1):
    """Calculate the Voronoi diagram (or Thiessen polygons) with the given points.

    If both envelope and polygon_shp are given, envelope will be discarded

    :param points: (string) shapefile with the points, for which the Thiessen polygons are created
    :param thiessen: (string) shapefile with polygons, and with coordinate system and attributes from point_shp
    :param points_layer_number:
    :param point_id_field_names: (list) list of field names to be assigned in thiessen_shp
    :param envelope: (list) [minX, maxX, minY, maxY] envelope used to clip the thiessen polygons
    :param clip_region: (string) shapefile with the polygon(s) used to clip the thiessen polygons
    :param clip_region_layer_number:
    :param expand:
    :return:
    """

    def merge_extent(ext0, ext1=None, point=None):
        if not ext1:
            ext1 = [point[0], point[1], point[0], point[1]] if point else ext0
        return [min(ext0[0], ext1[0]), max(ext0[1], ext1[1]), min(ext0[2], ext1[2]), max(ext0[3], ext1[3])]

    # Check given point field name is not valid
    lrs_thiessen = LayersWriter(source=thiessen)

    lrs_point = LayersReader(points)
    lyr_point = lrs_point.get_layer(points_layer_number)
    ext_point = lyr_point.GetExtent()

    # Get the clip polygon, The clip object can be 1) the given polygon, 2) the give bounding box (envelope) or
    # 3) an envelope expanding 10% of the points envelope
    if clip_region:
        from . import proc as gp
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
        # poly = 'POLYGON((' + '{} {}, {} {}, {} {}, {} {}, {} {}))'.format(
        #     envelope[0], envelope[2], envelope[1], envelope[2],
        #     envelope[1], envelope[3], envelope[0], envelope[3], envelope[0], envelope[2])
        # clip_geometry_wkb = ogr.Geometry(poly).ExportToWkb()
        poly = [[envelope[0], envelope[2]], [envelope[1], envelope[2]],
            [envelope[1], envelope[3]], [envelope[0], envelope[3]], [envelope[0], envelope[2]]]
        from girs.feat.geom import create_polygon
        clip_geometry_wkb = create_polygon(poly).ExportToWkb()

    clip_geometry = ogr.CreateGeometryFromWkb(clip_geometry_wkb)

    # Merge the envelope of the given points with the given envelope or polygons envelope
    merge_extent(ext_point, clip_geometry.GetEnvelope())

    if point_id_field_names:
        point_id_field_names = list(set(f[0] for f in lrs_point.get_field_definitions()) - set(point_id_field_names))

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
            print('ERROR Thiessen')
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
        lrs_thiessen = LayersUpdate(thiessen)
        for layer_number in range(lrs_thiessen.get_layer_count()):
            lrs_thiessen.delete_fields(point_id_field_names, layer_number)
        del lrs_thiessen


def get_thiessen_weights(point_shp, point_shp_id, polygon_shp, polygon_shp_id,
                         valid_points=None, work_directory=None):
    """Create Thiessen weights of points in area

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

    1)  determination of Thiessen polygons

    2)  intersection of Thiessen polygons with areas in polygon_shp

    3)  determination of the weights of the intersections for each area. The
        weights of each area must sum up to one.

    point_shp: full qualified name of the features used to create Thiessen
        polygons. The geometry must be point

    point_shp_id: name of the field containing unique record
        values. It is recommended a user defined field instead of using feature IDs (FID).
        The field type must be number or string

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

    :param point_shp:
    :param point_shp_id:
    :param polygon_shp:
    :param polygon_shp_id:
    :param valid_points:
    :param work_directory:
    :return:

    """
    from girs.util.osu import create_tmp_directory, remove_directory  # TODO: remove it

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
    point_id_field = lys_point.get_field_definition(point_shp_id)
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
    if not lys_polygon.is_geometry_polygon().all():
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
    lys_polygon = LayersUpdate(polygon_shp)

    # create a_area_field and calculate the area
    lys_polygon.add_fields([[a_area_field, ogr.OFTReal, 18]])
    lys_polygon.calculate_area(a_area_field)

    # delete all fields from polygon_shp but polygon_shp_id and a_area_field
    fields = list(set([f[0] for f in lys_polygon.get_field_definitions()]) - set([a_area_field, polygon_shp_id]))
    lys_polygon.delete_fields(fields)

    # ========== copy point_shp, create new and delete unused fields ========
    # copy point
    lys_point.copy(tmp_point0_shp)
    lys_point.destroy()

    lys_point = LayersUpdate(tmp_point0_shp)

    # create t_area_field
    lys_point.add_fields([[t_area_field, ogr.OFTReal, 18]])

    if point_shp_id == polygon_shp_id:
        # rename the field and create a new shapefile
        point_shp_id += '_t'
        lys_point.rename_fields(tmp_point0_shp, [[]])

    # delete all fields from point_shp but point_shp_id and t_area_field
    fields = list(set([f[0] for f in lys_point.get_field_definitions()]) - set([point_shp_id, t_area_field]))
    lys_point.delete_fields(fields)

    # =========================================================================
    polygon_weights = {}  # polygons

    def point_filter(value, values):
        return value in values

    keys = sorted(list(valid_points.keys()), key=operator.itemgetter(0))
    for k in keys:
        date_key = k[0], k[1]
        print('calculating', date_key)

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
        dissolve(tmp_thiessen1, tmp_thiessen2, fields)

        # calculate Thiessen polygon areas
        lys_thiessen = LayersUpdate(tmp_thiessen2)
        lys_thiessen.calculate_area(t_area_field)

        # calculate polygons weights
        for ly in lys_thiessen.layers():
            ly.ResetReading()
            for feat in ly:
                a_area = feat.GetField(a_area_field)  # polygon area
                t_area = feat.GetField(t_area_field)  # intersection area
                a_code = feat.GetField(polygon_shp_id)  # polygon id
                t_code = feat.GetField(point_shp_id)  # point id
                weight = old_div(t_area, a_area)

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


def merge(layers, target=None, layers_numbers=None, field_names=None):
    """Merge features from given layers

    :param layers: a list of ogr layers, ogr datasets, or filenames
    :param target:
    :param layers_numbers: list of layers numbers used for layers ogr-dataset or filename.
        If layers_numbers = None, the first layer will be used (layers_number=0). If the length of layers_numbers is
        less then the length of layers, the list layers_numbers will be completed with zeros.
    :param field_names:
    :return:
    """
    n_layers = len(layers)
    layers0 = list()
    for lrs in layers:
        try:
            layers0.append(LayersReader(lrs))
        except (TypeError, RuntimeError):
            layers0.append(lrs)
    layers = layers0

    if not layers_numbers:
        layers_numbers = [0] * n_layers
    elif len(layers_numbers) < n_layers:
        layers_numbers = layers_numbers + [0] * (n_layers - len(layers_numbers))

    if not field_names:
        field_names = list(set([fn for field_names in [lrs.get_field_names() for lrs in layers] for fn in field_names]))

    field_definitions = OrderedDict()
    srs, geo_type = None, None
    for i, lrs in enumerate(layers):
        for fd in lrs.get_field_definitions():
            fd_name = fd.name
            if fd_name in field_names and fd_name not in field_definitions:
                field_definitions[fd_name] = fd
        if not srs:
            srs = srs_from_wkt(lrs.get_coordinate_system(layer_number=layers_numbers[i]))
            geo_type = lrs.get_geometry_type(layer_number=layers_numbers[i])

    lrs_out = LayersWriter(['', geo_type, srs, list(field_definitions.values())], source=target)

    for lrs in layers:
        field_names = list(set(lrs.get_field_names()).intersection(set(field_definitions.keys())))
        df_values = lrs.get_geometries_and_field_values(field_names)
        for row in df_values.itertuples():
            kwargs = dict()
            for j, f in enumerate(field_names):
                kwargs[f] = str(row[j+2])  # 0: index, 1: _GEOM_
            kwargs['geom'] = row._1
            lrs_out.create_feature(**kwargs)
    return lrs_out


def intersection(source0, source1, target):
    """Return the intersection of `source0` and `source1`

    :param source0:
    :param source1:
    :param target:
    :return:
    """
    lrs0 = LayersReader(source0)
    lrs1 = LayersReader(source1)
    lrs2 = LayersWriter(target)
    ly0 = lrs0.get_layer()
    ly1 = lrs1.get_layer()
    lyt = lrs2.create_layer(str(0), prj=ly0.GetSpatialRef(), geom=ly0.GetGeomType())
    lyt = ly0.Intersection(ly1, lyt)
    return lyt

# def dissolve(source, **kwargs):
#     """Dissolve features in a layers set.
#
#     :param source: LayersSet or filename of a LayersSet
#     :param kwargs:
#         target: filename. Default is '' (memory)
#         fields: name or list of names of fields used to dissolve
#         stats<X>: dictionary for layer <X> : field name: statistics
#         stats_to_name: if True appends the statistic to the field name
#         cpus: default 1
#         The remaining kwargs are fieldname=statistics
#     :return:
#     """
#     stats_methods_dict = {
#         'first': lambda values: values[0],
#         'last': lambda values: values[-1],
#         'count': lambda values: len(values),
#         'min': lambda values: min(values),
#         'max': lambda values: max(values),
#         'sum': lambda values: sum(values),
#         'mean': lambda values: np.mean(values),
#         'range': lambda values: np.ptp(values),
#         'std': lambda values: np.std(values),
#     }
#
#     target = kwargs.pop('target', '')
#     fields = kwargs.pop('fields', None)
#     cpus = kwargs.pop('cpus', 1)
#     stats_to_name = kwargs.pop('stats_to_name', False)
#     try:
#         lrs_in = LayersReader(source)
#     except RuntimeError:
#         lrs_in = source
#
#     try:
#         lrs_out = LayersWriter(source=target)
#     except RuntimeError:
#         lrs_out = target
#
#     for ilyr, lyr_in in enumerate(lrs_in.layers()):
#
#         ldf_in = lyr_in.GetLayerDefn()
#         lyr_out = lrs_out.create_layer('', prj=lyr_in.GetSpatialRef(), geom=lyr_in.GetGeomType())
#
#         # for fd in get_field_definitions(ldf_in):
#         #     # TODO create only the necessary, not all fields
#         #     lyr_out.CreateField(fd)
#         # ldf_out = lyr_in.GetLayerDefn()
#
#         if fields:
#             fd_diss = [(i, fd) for i, fd in enumerate(field_definitions(ldf_in))]
#             fd_diss = [fd for fd in fd_diss if fd[1].GetName() in fields]
#             # Sort field definitions by names given in fields
#             fd_diss = [fd for (f, fd) in sorted(zip(fields, fd_diss), key=lambda x: x[0])]
#         else:
#             fd_diss = []
#
#         stats_keys = [k for k in kwargs.keys() if k not in fields]
#         stats_methods = [stats_methods_dict[kwargs[k]] for k in stats_keys]
#         if stats_keys:
#             fd_stats = [(i, fd) for i, fd in enumerate(field_definitions(ldf_in))]
#             fd_stats = [fd for fd in fd_stats if fd[1].GetName() in stats_keys]
#             # Sort field definitions by names given in fields
#             fd_stats = [fd for (f, fd) in sorted(zip(fields, fd_stats), key=lambda x: x[0])]
#         else:
#             fd_stats = []
#
#         geometry_dict = dict()
#         attributes_dict = dict()
#         lyr_in.ResetReading()
#         for i, feat in enumerate(lyr_in):
#             geom = feat.GetGeometryRef()
#             if geom is None:
#                 continue
#             geom_wkb = geom.ExportToWkb()
#             fd = tuple([feat.GetField(fid[0]) for fid in fd_diss])
#             if fd not in geometry_dict:
#                 geometry_dict[fd] = []
#             geometry_dict[fd].append(geom_wkb)
#             if fd not in attributes_dict:
#                 attributes_dict[fd] = []
#             attributes_dict[fd].append([feat.GetField(fid[0]) for fid in fd_stats])
#             del feat
#
#         for fd in fd_diss:
#             lyr_out.CreateField(fd[1])
#         for fd in fd_stats:
#             lyr_out.CreateField(fd[1])
#         for fd in field_definitions(lyr_out.GetLayerDefn()):
#             name = fd.GetName()
#             if name in stats_keys:
#                 if stats_to_name:
#                     fd.SetName(name + kwargs[name].capitalize())
#                 else:
#                     fd.SetName(name)
#
#         ldf_out = lyr_out.GetLayerDefn()
#         fds = geometry_dict.keys()
#         for fd in fds:
#             feature = ogr.Feature(ldf_out)
#             feature.SetGeometry(ogr.CreateGeometryFromWkb(union_cascade(geometry_dict.pop(fd), cpus)))
#             idx = 0
#             for value in fd:
#                 feature.SetField(idx, value)
#                 idx += 1
#             if stats_methods:
#                 for i, value in enumerate([stats_methods[i](v) for i, v in enumerate(zip(*attributes_dict[fd]))]):
#                     feature.SetField(idx + i, value)
#             lyr_out.CreateFeature(feature)
#             del feature
#
#     lrs_out.dataset.FlushCache()
#     return lrs_out
