__author__ = 'roehrig'

from osgeo import osr


def srs_from_epsg(epsg):
    """
        epsg: integer or in the format 'EPSG:12345'
    """
    try:
        epsg = int(epsg)
    except ValueError:
        epsg = int(epsg.split(':')[-1].strip())
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs


def srs_from_epsga(epsga):
    srs = osr.SpatialReference()
    srs.ImportFromEPSGA(epsga)
    return srs


def srs_from_erm(erm):
    srs = osr.SpatialReference()
    srs.ImportFromERM(erm)
    return srs


def srs_from_esri(esri):
    srs = osr.SpatialReference()
    srs.ImportFromESRI(esri)
    return srs


def srs_from_mi_coord_sys(mi):
    srs = osr.SpatialReference()
    srs.ImportFromMICoordSys(mi)
    return srs


def srs_from_ozi(ozi):
    srs = osr.SpatialReference()
    srs.ImportFromOzi(ozi)
    return srs


def srs_from_pci(*args):
    srs = osr.SpatialReference()
    srs.ImportFromPCI(*args)
    return srs


def srs_from_proj4(proj4):
    srs = osr.SpatialReference()
    a = srs.ImportFromProj4(proj4)
    return srs


def srs_from_url(url):
    srs = osr.SpatialReference()
    srs.ImportFromUrl(url)
    return srs


def srs_from_usgs(*args):
    srs = osr.SpatialReference()
    srs.ImportFromUSGS(*args)
    return srs


def srs_from_wkt(wkt):
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return srs


def srs_from_xml(xml):
    srs = osr.SpatialReference()
    srs.ImportFromXML(xml)
    return srs


# =============================================================================
#
# =============================================================================
def get_srs(**kwargs):
    """
    kwargs:
        keys: 'epsg', 'epsga', 'erm', 'esri', 'mi', 'ozi', 'pci', 'proj4', 'url', 'usgs', 'wkt', 'xml'
        values: the coordinate system
    """
    srs_from_dict = {
        'epsg': srs_from_epsg,
        'epsga': srs_from_epsga,
        'erm': srs_from_erm,
        'esri': srs_from_esri,
        'mi': srs_from_mi_coord_sys,
        'ozi': srs_from_ozi,
        'pci': lambda v: srs_from_pci(*v),
        'proj4': srs_from_proj4,
        'url': srs_from_url,
        'usgs': lambda v: srs_from_usgs(*v),
        'wkt': srs_from_wkt,
        'xml': srs_from_xml
    }
    if kwargs and len(kwargs) == 1:
        k, v = kwargs.iteritems().next()
        if k in srs_from_dict:
            return srs_from_dict[k](v)
    return None


def get_utm_zone(prj, lon, lat):
    srs = osr.SpatialReference()
    srs.ImportFromWkt(prj)
    if srs.IsGeographic() and srs.GetAttrValue('GEOGCS') == 'WGS 84':
        return int(1+(lon+180.0)/6.0), lat >= 0.0
    else:
        return 0, None

