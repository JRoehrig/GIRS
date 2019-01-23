from osgeo import osr

_srs_export_dict = {
    'xml': 'ExportToXML',
    'wkt': 'ExportToWkt',
    'usgs': 'ExportToUSGS',
    'proj4': 'ExportToProj4',
    'prettywkt': 'ExportToPrettyWkt',
    'pci': 'ExportToPCI',
    'micoordsys': 'ExportToMICoordSys'
}


def export(srs, fmt='wkt'):
    """Export the spatial reference to a string format. Default is 'wkt' (well known type)

    Possible formats:

        * 'xml'
        * 'wkt'
        * 'usgs'
        * 'proj4'
        * 'prettywkt'
        * 'pci'
        * 'micoordsys'


    :param srs: spatial reference
    :type srs: osgeo.osr.SpatialReference
    :param fmt: export format
    :type fmt: str
    :return: spatial reference in the required format
    """
    return getattr(srs, _srs_export_dict[fmt.lower()])()


def srs_from_epsg(epsg):
    """Return the spatial reference from EPSG GCS or PCS code

    :param epsg: code or 'EPSG:XXXXX' where XXXXX is the epsg code
    :type epsg: int, str
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    try:
        epsg = int(epsg)
    except ValueError:
        epsg = int(epsg.split(':')[-1].strip())
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    return srs


def srs_from_epsga(epsga):
    """Return the spatial reference from EPSG GCS or PCS code

    :param epsga: code or 'EPSGA:XXXXX' where XXXXX is the epsg code
    :type epsga: int, str
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    try:
        epsga = int(epsga)
    except ValueError:
        epsga = int(epsga.split(':')[-1].strip())
    srs = osr.SpatialReference()
    srs.ImportFromEPSGA(epsga)
    return srs


def srs_from_erm(proj, datum, units):
    """Return the spatial reference from ERMapper projection definitions

    :param proj: projection name, such as "NUTM11" or "GEOGRAPHIC"
    :type proj: str
    :param datum: datum name, such as "NAD83"
    :type datum: str
    :param units: linear units "FEET" or "METERS"
    :type units: str
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    srs = osr.SpatialReference()
    srs.ImportFromERM(*[proj, datum, units])
    return srs


def srs_from_esri(prj_filename):
    """Return the spatial reference from ESRI .prj format(s)

    :param prj_filename:  ESRI projection filename .prj
    :type prj_filename: str
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    srs = osr.SpatialReference()
    srs.ImportFromESRI(prj_filename)
    return srs


def srs_from_mi_coord_sys(mapinfo):
    """Return the spatial reference from Mapinfo style CoordSys

    :param mapinfo:  Mapinfo coordinate system
    :type mapinfo: str
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    srs = osr.SpatialReference()
    srs.ImportFromMICoordSys(mapinfo)
    return srs


def srs_from_ozi(ozi):
    """Return the spatial reference from OziExplorer projection definition

    :param ozi: strings containing the whole OziExplorer .MAP file
    :type ozi: list
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    srs = osr.SpatialReference()
    srs.ImportFromOzi(ozi)
    return srs


def srs_from_pci(*args):
    """Return the spatial reference from PCI projection definition

    :param proj: NULL terminated string containing the definition. Looks like "pppppppppppp Ennn" or "pppppppppppp Dnnn", where "pppppppppppp" is a projection code, "Ennn" is an ellipsoid code, "Dnnn" - a datum code.
    :type proj: str
    :param units: Grid units code ("DEGREE" or "METRE"). If NULL "METRE" will be used.
    :type units: str
    :param parameters: 17 coordinate system parameters
    :type parameters: list
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    srs = osr.SpatialReference()
    srs.ImportFromPCI(*args)
    return srs


def srs_from_proj4(proj4):
    """Return the spatial reference from PROJ.4 coordinate string

    :param proj4: PROJ.4 coordinate string
    :type proj4: str
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    srs = osr.SpatialReference()
    srs.ImportFromProj4(proj4)
    return srs


def srs_from_url(url):
    """Return the spatial reference from a url

    :param url: url
    :type url: str
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    srs = osr.SpatialReference()
    srs.ImportFromUrl(url)
    return srs


def srs_from_usgs(proj, zone, parameters, datum):
    """Return the spatial reference from the USGS projection definition

    :param proj: projection system code, used in GCTP
    :type proj: int
    :param zone: zone for UTM and State Plane projection systems
    :type zone: int
    :param parameters: list of 15 coordinate system parameters
    :type parameters: list of float
    :param datum: spheroid
    :type datum: int
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    # TODO:
    # :param angle_format: one of USGS_ANGLE_DECIMALDEGREES, USGS_ANGLE_PACKEDDMS, or USGS_ANGLE_RADIANS (default is USGS_ANGLE_PACKEDDMS)
    # :type angle_format: int
    srs = osr.SpatialReference()
    srs.ImportFromUSGS(proj, zone, parameters, datum)
    return srs


def srs_from_wkt(wkt):
    """Return the spatial reference from wkt string

    :param wkt: wkt string
    :type wkt: str
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    return srs


def srs_from_xml(xml):
    """Return the spatial reference from XML format (GML only currently).

    :param xml: wkt string
    :type xml: str
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    srs = osr.SpatialReference()
    srs.ImportFromXML(xml)
    return srs


_srs_from_dict = {
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


def get_srs(**kwargs):
    """Return the spatial reference from given format

    :param kwargs:

        - keys: 'epsg', 'epsga', 'erm', 'esri', 'mi', 'ozi', 'pci', 'proj4', 'url', 'usgs', 'wkt', 'xml'
        - values: coordinate system parameters for the given keys
    :return: spatial reference
    :rtype: osgeo.osr.SpatialReference
    """
    if kwargs and len(kwargs) == 1:
        k = list(kwargs.keys())[0]
        if k in _srs_from_dict:
            return _srs_from_dict[k](kwargs[k])
    return None


def is_same_srs(srs0, srs1):
    """

    :param srs0: spatial reference system
    :type srs0: str or osr.SpatialReference
    :param srs1: spatial reference system
    :type srs1: str or osr.SpatialReference
    :return:
    """

    try:
        srs0 = srs_from_wkt(srs0)
    except TypeError:
        pass
    try:
        srs1 = srs_from_wkt(srs1)
    except TypeError:
        pass
    if srs0.AutoIdentifyEPSG() == 0 and srs1.AutoIdentifyEPSG() == 0:
        return srs0.GetAuthorityCode(None) == srs1.GetAuthorityCode(None)
    else:
        # TODO: this is not good!
        try:
            return srs0.export() == srs1.export()
        except AttributeError:
            return srs0.ExportToWkt() == srs1.ExportToWkt()




def get_utm_zone_from_wgs84(wkt, lon, lat):
    """

    :param wkt:
    :type wkt:
    :param lon:
    :type lon:
    :param lat:
    :type lat:
    :return:
    :rtype
    """
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    if srs.IsGeographic() and srs.GetAttrValue('GEOGCS') == 'WGS 84':
        return int(1+(lon+180.0)/6.0), lat >= 0.0
    else:
        return 0, None

