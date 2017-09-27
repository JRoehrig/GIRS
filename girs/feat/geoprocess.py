import time
import numpy as np
import multiprocessing as mp
from osgeo import ogr
from layers import LayersReader, LayersWriter, get_field_definitions


def union_cascade(geometries, cpus=1):
    uq_geometries = []
    for geom_wkb in geometries:
        geom = ogr.CreateGeometryFromWkb(geom_wkb)
        geom_type = geom.GetGeometryType()
        env = geom.GetEnvelope()
        if geom_type == ogr.wkbPolygon:
            xmin, xmax, ymin, ymax = env
            ext_env = (xmin, xmax, ymin, ymax, ((xmax + xmin) * 0.5, (ymax + ymin) * 0.5))
            uq_geometries.append([geom_wkb, geom_type, ext_env])
        elif geom_type == ogr.wkbMultiPolygon:
            geom = ogr.CreateGeometryFromWkb(geom_wkb)
            for geom_part in geom:
                xmin, xmax, ymin, ymax = geom_part.GetEnvelope()
                ext_env = (xmin, xmax, ymin, ymax, ((xmax + xmin) * 0.5, (ymax + ymin) * 0.5))
                uq_geometries.append([geom_part.ExportToWkb(), ogr.wkbPolygon, ext_env])
    available_cpu_count = mp.cpu_count() if cpus > 1 else None
    return UnionQuadtree(uq_geometries, '', cpus, available_cpu_count).get_geometry()


def _union_quadtree(geometries, level, cpu, max_cpu):
    glock.acquire()
    time.sleep(0.1)
    glock.release()
    return UnionQuadtree(geometries, level, cpu, max_cpu).get_geometry()


def _init(lock):
    global glock
    glock = lock


def _parallelize_union(level, glist, requested_cpu_count, available_cpu_count):
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
        pool.apply_async(_union_quadtree, (g, level_i, requested_cpu_count, available_cpu_count), callback=collect_results)
    pool.close()
    pool.join()
    return geometries


class UnionQuadtree(object):

    threshold = 500

    def __init__(self, geometries, level='', requested_cpu_count=1, available_cpu_count=1):
        requested_cpu_count = min(requested_cpu_count, available_cpu_count)
        env = zip(*zip(*geometries)[2])
        self.center = ((max(env[1]) + min(env[0])) * 0.5, (max(env[3]) + min(env[2])) * 0.5)
        del env
        if len(geometries) > UnionQuadtree.threshold and len(level) < 4:
            if requested_cpu_count > 1:
                glist = list()
                for i in range(4):
                    geometries0, geometries = self.filter_geometries(geometries, i)
                    if geometries0:
                        glist.append(geometries0)
                glist = _parallelize_union(level, glist, requested_cpu_count, available_cpu_count)
            else:
                glist = list()
                for i in range(4):
                    geometries0, geometries = self.filter_geometries(geometries, i)
                    if geometries0:
                        glist.append(UnionQuadtree(geometries0, level + str(i)).geometry)
                    del geometries0
            glist = [g for g in glist if g is not None]
        else:
            glist = [geom_wkb for geom_wkb, _, _ in geometries]
        self.geometry = self.union(glist)

    def get_geometry(self):
        return self.geometry

    def filter_geometries(self, geometries0, quadrant):
        fnc = [lambda center: center[0] <  self.center[0] and center[1] <  self.center[1],
               lambda center: center[0] <  self.center[0] and center[1] >= self.center[1],
               lambda center: center[0] >= self.center[0] and center[1] <  self.center[1],
               lambda center: center[0] >= self.center[0] and center[1] >= self.center[1]]
        filter = fnc[quadrant]
        geometries1 = []
        geometries2 = []
        for i in range(len(geometries0)):
            geom_list = geometries0[i]
            if filter(geom_list[2][4]):
                geometries1.append(geom_list)
            else:
                geometries2.append(geom_list)
        return geometries1, geometries2

    def union(self, glist):
        geom_multi = ogr.Geometry(ogr.wkbMultiPolygon)
        while glist:
            g_wkb = glist.pop()
            g = ogr.CreateGeometryFromWkb(g_wkb)
            geom_type = g.GetGeometryType()
            if geom_type == ogr.wkbPolygon:
                geom_multi.AddGeometry(g)
            elif geom_type == ogr.wkbMultiPolygon:
                for g_part in g:
                    geom_multi.AddGeometry(g_part)
        return geom_multi.UnionCascaded().ExportToWkb()


def dissolve(source, **kwargs):
    """

    :param source:
    :param kwargs:
        target:
        fields:
        cpus:
    :return:
    """
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

    target = kwargs.pop('target', '')
    fields = kwargs.pop('fields', None)
    cpus = kwargs.pop('cpus', 1)

    try:
        lrs_in = LayersReader(source)
    except RuntimeError:
        lrs_in = source

    try:
        lrs_out = LayersWriter(target)
    except RuntimeError:
        lrs_out = target

    for lyr_in in lrs_in.layers():

        ldf_in = lyr_in.GetLayerDefn()

        lyr_out = lrs_out.create_layer('', prj=lyr_in.GetSpatialRef(), geom=lyr_in.GetGeomType())
        if fields:
            fd_diss = [(i, fd) for i, fd in enumerate(get_field_definitions(ldf_in))]
            fd_diss = [fd for fd in fd_diss if fd[1].GetName() in fields]
            # Sort field definitions by names given in fields
            fd_diss = [fd for (f, fd) in sorted(zip(fields, fd_diss), key=lambda x: x[0])]
        else:
            fd_diss = []

        stats_keys = [k for k in kwargs.keys() if k not in fields]
        stats_methods = [stats_methods_dict[kwargs[k]] for k in stats_keys]
        if stats_keys:
            fd_stats = [(i, fd) for i, fd in enumerate(get_field_definitions(ldf_in))]
            fd_stats = [fd for fd in fd_stats if fd[1].GetName() in stats_keys]
            # Sort field definitions by names given in fields
            fd_stats = [fd for (f, fd) in sorted(zip(fields, fd_stats), key=lambda x: x[0])]
        else:
            fd_stats = []

        geometry_dict = dict()
        attributes_dict = dict()
        lyr_in.ResetReading()
        for i, feat in enumerate(lyr_in):
            geom = feat.GetGeometryRef()
            if geom is None:
                continue
            geom_wkb = geom.ExportToWkb()
            fd = tuple([feat.GetField(fid[0]) for fid in fd_diss])
            if fd not in geometry_dict:
                geometry_dict[fd] = []
            geometry_dict[fd].append(geom_wkb)
            if fd not in attributes_dict:
                attributes_dict[fd] = []
            attributes_dict[fd].append([feat.GetField(fid[0]) for fid in fd_stats])
            del feat

        for fd in fd_diss:
            lyr_out.CreateField(fd[1])
        for fd in fd_stats:
            lyr_out.CreateField(fd[1])
        for fd in get_field_definitions(lyr_out.GetLayerDefn()):
            name = fd.GetName()
            if name in stats_keys:
                fd.SetName(name + '_' + kwargs[name].upper())

        ld_out = lyr_out.GetLayerDefn()

        fds = geometry_dict.keys()
        for fd in fds:
            feature = ogr.Feature(ld_out)
            feature.SetGeometry(ogr.CreateGeometryFromWkb(union_cascade(geometry_dict.pop(fd), cpus)))
            idx = 0
            for value in fd:
                feature.SetField(idx, value)
                idx += 1
            if stats_methods:
                for i, value in enumerate([stats_methods[i](v) for i, v in enumerate(zip(*attributes_dict[fd]))]):
                    feature.SetField(idx + i, value)
            lyr_out.CreateFeature(feature)
            del feature


