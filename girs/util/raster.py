import os
from osgeo import gdal


def driver_dictionary():
    drivers_dict = {}
    for i in range(gdal.GetDriverCount()):
        drv = gdal.GetDriver(i)
        if drv.GetMetadataItem(gdal.DCAP_RASTER):
            extensions = drv.GetMetadataItem(gdal.DMD_EXTENSIONS)
            extensions = extensions.split() if extensions else [None]
            for ext in extensions:
                if ext:
                    if ext.startswith('.'):
                        ext = ext[1:]
                    ext = ext.lower()
                    for ext in [e for e in ext.split('/')]:
                        if ext not in drivers_dict:
                            drivers_dict[ext] = []
                    drivers_dict[ext].append(drv.ShortName)
                else:
                    if None not in drivers_dict:
                        drivers_dict[None] = []
                    drivers_dict[None].append(drv.ShortName)
    return drivers_dict


def get_driver(filename=None, driver_short_name=None):
    """
    If driver_short_name is given, return the corresponding driver.
    filename can be 'ME' or a file name. If a file name is given, guess
    the driver, returning it only when the correspondence is one to one,
    else return None.

    Filename suffixes, which should work:
    ace2: ACE2, asc: AAIGrid, bin: NGSGEOID, blx: BLX, bmp: BMP, bt: BT, cal: CALS, ct1: CALS, dat: ZMap, ddf: SDTS,
    dem: USGSDEM, dt0: DTED, dt1: DTED, dt2: DTED, e00: E00GRID, gen: ADRG, gff: GFF, grb: GRIB, grc: NWT_GRC,
    gsb: NTv2, gtx: GTX, gxf: GXF, hdf: HDF4, hdf5: HDF5, hf2: HF2, hgt: SRTMHGT, jpeg: JPEG, jpg: JPEG, kea: KEA,
    kro: KRO, lcp: LCP, map: PCRaster, mem: JDEM, mpl: ILWIS, n1: ESAT, nat: MSGN, ntf: NITF, pix: PCIDSK, png:
    PNG, pnm: PNM, ppi: IRIS, rda: R, rgb: SGI, rik: RIK, rst: RST, rsw: RMF, sdat: SAGA, tif: GTiff, tiff: GTiff,
    toc: RPFTOC, vrt: VRT, xml: ECRGTOC, xpm: XPM, xyz: XYZ,

    Filenames with ambiguous suffixes:
    gif: GIF, BIGGIF
    grd: GSAG, GSBG, GS7BG, NWT_GRD
    hdr: COASP, MFF, SNODAS
    img: HFA, SRP
    nc: GMT, netCDF
    ter: Leveller, Terragen

    Without suffix: SAR_CEOS, CEOS, JAXAPALSAR, ELAS, AIG, GRASSASCIIGrid, MEM, BSB, DIMAP, AirSAR, RS2, SAFE,
    HDF4Image, ISIS3, ISIS2, PDS, VICAR, TIL, ERS, L1B, FIT, INGR, COSAR, TSX, MAP, KMLSUPEROVERLAY, SENTINEL2, MRF,
    DOQ1, DOQ2, GenBin, PAux, MFF2, FujiBAS, GSC, FAST, LAN, CPG, IDA, NDF, EIR, DIPEx, LOSLAS, CTable2, ROI_PAC, ENVI,
    EHdr, ISCE, ARG, BAG, HDF5Image, OZI, CTG, DB2ODBC, NUMPY

    :param filename:
    :param driver_short_name:
    :return:
    """
    if driver_short_name:
        driver = gdal.GetDriverByName(driver_short_name)
        if not driver:
            raise ValueError('Could not find driver for short name {}'.format(driver_short_name))
    elif not filename:
        driver = gdal.GetDriverByName('MEM')
    else:
        try:
            driver = gdal.IdentifyDriver(filename)
        except RuntimeError:
            driver = None
        if not driver:
            drivers_dict = driver_dictionary()
            try:
                driver_short_name = drivers_dict[filename.split('.')[-1]]
                if len(driver_short_name) == 1:
                    driver = gdal.GetDriverByName(driver_short_name[0])
                else:
                    raise ValueError('Ambiguous file name {} with possible drivers: {}'.format(
                        filename, ', '.join(driver_short_name)))
            except KeyError:
                raise ValueError('Could not find driver for file {}'.format(filename))
    return driver





# def get_driver_name_from_file_name(filename):
#     if filename:
#         return gdal.GetDriverByName(filename)
#         # s = splitext(filename)
#         # if len(s) > 1:
#         #     suffix = s[1].lower() if s[1] != '' else s[0]
#         #     if suffix.startswith('.'):
#         #         suffix = suffix[1:]
#         #     return GDAL_DRIVER_MAP[suffix] if suffix in GDAL_DRIVER_MAP else None
#     return None
#
#
# def get_driver_name(driver=None, output_raster=None):
#     if driver:
#         return driver
#     if output_raster:
#         return get_driver_name_from_file_name(output_raster)
#     return 'MEM'


# def get_driver(**kwargs):
#     # Default driver
#     driver_name = 'MEM'
#     if 'driver' in kwargs:
#         driver_name = kwargs['driver']
#     if 'output_raster' in kwargs:
#         output = kwargs['output_raster']
#         if not driver_name:
#             driver_name = get_driver_name_from_file_name(output)
#             if driver_name is None:
#                 raise Exception('Could not find driver for file {}'.format(output))


# def identify_driver(filename):
#     return gdal.IdentifyDriver(filename)

