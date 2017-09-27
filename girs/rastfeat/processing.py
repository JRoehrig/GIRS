from osgeo import gdal

from girs import Layers
from girs.rast.raster import Raster


def burn_feature(r_in, shp_in, r_out):
    """
    Args:
        r_in (string):

        shp_in (string): filename of the shapefile. The geometry must be line or polygon

        r_out (string):
    """

    # Get the union of the polygons and its extent
    lrs = Layers(shp_in)
    xmin, xmax, ymin, ymax = lrs.get_extent()

    r_in = Raster(r_in)
    rb_in_nodata = r_in.get_nodata()
    n_bands = r_in.get_number_of_bands()

    (u_min, u_max, v_min, v_max), geo_trans = r_in.extent_world_to_pixel(xmin, xmax, ymin, ymax)

    # rasterize the shapefile
    ds0_shp = gdal.GetDriverByName('MEM').Create('', u_max-u_min, v_max-v_min, 1, gdal.GDT_Byte)
    ds0_shp.SetGeoTransform(geo_trans)
    ds0_shp.SetProjection(r_in.get_coordinate_system())

    gdal.RasterizeLayer(ds0_shp, [1], lrs.get_layer(), burn_values=[1])

    array_shp = ds0_shp.GetRasterBand(1).ReadAsArray(0, 0, u_max-u_min, v_max-v_min)
    # array_shp = np.logical_not(array_shp)
    dr_out = gdal.GetDriverByName('GTiff')
    ds_out = dr_out.Create(r_out, u_max-u_min, v_max-v_min, n_bands,  gdal.GDT_Byte)
    # ds_out = dr_out.Create(r_out, u_max-u_min, v_max-v_min, r_in.number_of_bands(), r_in.band_data_type())
    ds_out.SetGeoTransform(geo_trans)
    ds_out.SetProjection(r_in.get_coordinate_system())

    for i in range(r_in.get_number_of_bands()):
        rb_out = ds_out.GetRasterBand(i+1)
        # Write the array into the band
        array_in = r_in.clip_array(i + 1, u_min, v_min, u_max - u_min, v_max - v_min)
        rb_out.WriteArray(np.multiply(array_in,  array_shp))
        rb_out.SetNoDataValue(rb_in_nodata[i])

    gdal.ErrorReset()


