Welcome to GIRS's documentation!
================================

``girs`` (Geographic Information and Remote Sensing) is basically a wrapper around osgeo/gdal and osgeo/ogr written in python with additional  geoprocessing algorithms. It was initially developed as part of a toolset to support geoprocessing for water resources management, but then splitted into a standalone general module on geoprocessing. 


The module is divided into the following packages:

* feat: feature (vector) data geoprocessing
* raster: raster data geoprocessing
* rasterfeat: geoprocessing combining features and rasters
* util: utilities and general tools