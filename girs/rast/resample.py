import gdal

from girs.rast.raster import RasterReader, RasterWriter, get_driver


def resample(input_raster, pixel_sizes, resample_alg=gdal.GRA_NearestNeighbour, **kwargs):
    """
    Args:
        input_raster: file name of a rasters or a Raster object

        pixel_sizes (list, str or Raster): list of pixel_width, pixel_height, file name of a rasters, or Raster object

        kwargs:
            'output_raster' (string): filename of the new rasters or 'MEM' (default).
            'driver' (string): driver name

    """


    try:
        pixel_width, pixel_height = float(pixel_sizes), float(pixel_sizes)
    except:
        try:
            pixel_width, pixel_height = pixel_sizes
            pixel_width, pixel_height = float(pixel_width), float(pixel_height)
        except:
            try:
                raster0 = RasterReader(pixel_sizes)
            except:
                raster0 = pixel_sizes
            pixel_width, pixel_height = raster0.get_pixel_size()
            del raster0

    try:
        input_raster = RasterReader(input_raster)
    except:
        pass

    x_min, x_max, y_min, y_max = input_raster.get_extent()

    driver = get_driver(kwargs['output_raster'] if 'output_raster' in kwargs else None,
                        kwargs['driver'] if 'driver' in kwargs else None)
    driver = driver.ShortName

    len_x = x_max-x_min
    len_y = y_max-y_min
    u_max, v_max = int(len_x/pixel_width), int(len_y/pixel_height)
    if ((len_x - u_max * pixel_width) / pixel_width) > 0.5:
        u_max += 1
    if ((len_y - v_max * pixel_height) / pixel_height) > 0.5:
        v_max += 1

    gt_out = list(input_raster.get_geotransform())  # make it mutable
    gt_out[1], gt_out[5] = pixel_width, -pixel_height

    name = kwargs['output_raster'] if kwargs and 'output_raster' in kwargs else 'MEM'
    raster_parameters = input_raster.get_parameters()
    raster_parameters.RasterXSize = u_max
    raster_parameters.RasterYSize = v_max
    raster_parameters.geo_trans = gt_out
    raster_parameters.driverShortName = driver
    raster_out = RasterWriter(name, raster_parameters)

    gdal.ReprojectImage(input_raster.dataset, raster_out.dataset, input_raster.get_coordinate_system(), raster_out.get_coordinate_system(), resample_alg)

    return raster_out


