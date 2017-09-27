from gdalnumeric import BandReadAsArray
from girs.rast.raster import RasterReader, RasterWriter
from girs.util.raster import get_driver


def align_raster():
    pass


def _unformat_calculate_inputs(rasters, bands):

    filename_raster_dict = {}

    # auxiliary functions
    def _to_raster_and_bands(rv):
        try:  # rv is a file name
            if rv in filename_raster_dict:
                r = filename_raster_dict[rv]
            else:
                r = Raster(rv)
                filename_raster_dict[rv] = r
        except:
            if isinstance(rv, Raster):
                if rv.get_filename() in filename_raster_dict:
                    r = filename_raster_dict[rv]
                else:
                    r = rv
                    filename_raster_dict[rv.get_filename()] = r
            else:
                try:
                    return _to_raster_and_bands(rv[0])
                except:
                    raise
        if bands is not None:
            if bands == 'all':
                b = range(1, r.get_number_of_bands()+1)
            else:
                try:
                    b = [int(bands)]
                except:
                    b = bands
        else:
            b = 1
        return [r, b]

    rd = {}
    for raster_key, raster_value in rasters.items():
        rd[raster_key] = _to_raster_and_bands(raster_value)
    return rd


def _check_number_of_bands(rasters):
    n_bands = None
    for raster_key, raster_value in rasters.items():
        if n_bands is None:
            n_bands = len(raster_value[1])
            k0 = raster_key
        if n_bands != len(raster_value[1]):
            raise Exception('Raster {} and rasters {} defined different number of bands'.format(k0, raster_key))
    return n_bands


def _check_for_equal_projections_and_transformations(rasters):
    pr = None
    tr = None
    for raster_key, raster_value in rasters.items():
        r = raster_value[0]
        if pr is None:
            pr = r.get_coordinate_system()
            tr = r.get_geotransform()
            k0 = raster_key
        if pr != r.get_coordinate_system():
            raise Exception('Rasters {} and {} have different projections.'.format(k0, raster_key))
        if tr != r.get_geotransform():
            raise Exception('Rasters {} and {} have different transformations.'.format(k0, raster_key))
    return pr, tr


def _get_block(band):
    block_sizes = band.GetBlockSize()
    x_block_size = block_sizes[0]
    y_block_size = block_sizes[1]

    x_size = band.XSize
    y_size = band.YSize
    for yoff in range(0, y_size, y_block_size):
        ysize = y_block_size if yoff + y_block_size < y_size else y_size - yoff
        for xoff in range(0, x_size, x_block_size):
            xsize = x_block_size if xoff + x_block_size < x_size else x_size - xoff
            data = band.ReadAsArray(xoff, yoff, xsize, ysize)


def get_symbol_and_bands(key):
    digits = ''
    while key[-1].isdigit():
        digits += key[-1]
        key = key[:-1]
    try:
        return key, (int(digits),)
    except ValueError:
        return key, (1,)


def calculate(calc, **kwargs):
    """
    See also:
        https://svn.osgeo.org/gdal/trunk/gdal/swig/python/scripts/gdal_calc.py
        https://github.com/rveciana/geoexamples/blob/master/python/gdal-performance/classification_blocks.py

    Args:
        calc (str): algebraic operation supported by numpy arrays "(R1+R2)/(R1-R2)", where R1 and R2 are the keys used
            in the argument rasters
        rasters (dictionary):
            key (str):
            value (str / Raster or list):
                1) rasters file name, Raster object, or list with the follwing elements:
                    - rasters file name of Raster object
                    - band number or list of band numbers

        kwargs:
            'band_numbers': 'all' or a band number
            'output_raster' (string): filename of the new rasters or 'MEM' (default).
            'driver' (string): driver name
    """
    # load the dictionary with rasters

    if 'band_numbers' in kwargs:
        band_numbers = kwargs['band_numbers']
        del kwargs['band_numbers']
    else:
        band_numbers = None

    if 'output_raster' in kwargs:
        output_raster = kwargs['output_raster']
        del kwargs['output_raster']
    else:
        output_raster = ''

    if 'driver' in kwargs:
        driver = kwargs['driver']
        del kwargs['driver']
    else:
        driver = None

    driver = get_driver(output_raster, driver)
    driver = driver.ShortName

    symbol_bands_raster = [get_symbol_and_bands(k) + (RasterReader(v),) for k, v in kwargs.items()]
    if band_numbers:
        symbol_bands_raster = [[s, band_numbers, r] for s, n, r in symbol_bands_raster]

    number_of_bands = [len(n) for _, n, _ in symbol_bands_raster]
    assert number_of_bands[1:] == number_of_bands[:-1]
    number_of_bands = number_of_bands[0]

    for rz in zip(*[r.get_raster_size() for _, _, r in symbol_bands_raster]):
        assert rz[1:] == rz[:-1]

    raster_parameters = symbol_bands_raster[0][2].get_parameters()
    raster_parameters.driverShortName = driver

    r_out = RasterWriter(output_raster, raster_parameters)
    for i in range(number_of_bands):
        arr = eval(calc, {s: r.get_masked_array(n[i]) for s, n, r in symbol_bands_raster})
        arr._sharedmask = False
        arr[arr.mask] = r_out.get_nodata(i+1)
        print arr
        r_out.set_array(arr, i+1)
    return r_out

