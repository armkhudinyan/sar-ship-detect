from pathlib import Path
from typing import List, Tuple
import rasterio as rio
from rasterio.io import DatasetReader
from rasterio.crs import CRS
from rasterio.features import shapes
from rasterio import Affine
from numpy import ndarray
from shapely.geometry import shape
from geopandas import GeoDataFrame, GeoSeries
from matplotlib import pyplot as plt
import pandas as pd


def write_tif(raster_array: ndarray, src: DatasetReader, transform: Affine, 
               crs: CRS, dest_file: Path) -> None:
    '''Write  raster files into GeoTif 
    
    raster_array : 2d array 
    src          : source file
    transform    : geo_transform file
    crs          : coordinate reference system, e.g. 'EPSG':4326
    dest_file    : destination file path and name
    '''
    # get output raster size
    height, width = raster_array.shape
    # define the output raster parameters
    out_meta = src.meta.copy()
    export_tif = raster_array.astype('float32')
    # update the metadata
    out_meta.update({'dtype'    : 'float32'
                ,    'width'    : width
                ,    'height'   : height
                ,    'crs'      : crs #CRS.from_epsg(4326), #out_meta['crs'],
                ,    'transform': transform
    })
    # Write the array to raster GeoTIF 
    with rio.open(dest_file, "w", **out_meta) as dest:
        dest.write(export_tif, 1)  # data array should be in (band, rows, cols) order 
    print(f'saved tif: {dest}')


def GCPs2GeoTransform(src: DatasetReader, scale_factor: float=1, 
                      gt_style: str='Affine') -> Tuple[Affine,CRS]:
    """
    Reads GCP's with `rasterio` and returns geotransform 
    in either `Affine` style or in `GDAL_geotransform` style
    and `crs` data
    """
    GCPs = src.get_gcps()[0]
    transf_from_gcps = rio.transform.from_gcps(GCPs) # type: ignore
    
    if gt_style == 'Affine':
        transform = rio.Affine(
            # X-coord / long
            transf_from_gcps.a*scale_factor, 
            transf_from_gcps.b*scale_factor, 
            transf_from_gcps.c,
            # Y-coord / lat
            transf_from_gcps.d*scale_factor, 
            transf_from_gcps.e*scale_factor, 
            transf_from_gcps.f
        )
    elif gt_style == 'GDAL': 
        transform = rio.Affine(
            # X-coord / long
            transf_from_gcps.c*scale_factor, 
            transf_from_gcps.a*scale_factor, 
            transf_from_gcps.b,
            # Y-coord / lat
            transf_from_gcps.f*scale_factor, 
            transf_from_gcps.d*scale_factor, 
            transf_from_gcps.e
        )
    else:
        raise ValueError("Wrong input for 'gt_style', use 'Affine' or 'GDAL'")
    crs = src.get_gcps()[1]
    return transform, crs


def plot_gcps(GCPs: List) -> None:
    '''
    Plot GCPs extracted with rasterio.get_gcps()
    '''
    row_id = []
    col_id = []
    gcps_x = []
    gcps_y = []

    for i in range(len(GCPs)):
        row_id.append(GCPs[i].row)
        col_id.append(GCPs[i].col)
        gcps_x.append(GCPs[i].x)
        gcps_y.append(GCPs[i].y)
        
    df = pd.DataFrame([row_id, col_id, gcps_x, gcps_y]).T
    df.columns = ['row', 'col', 'x', 'y']                        # type: ignore
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    ax1.scatter(df.x, df.y, s=20, c='b', marker="s", label='GCPs') # type: ignore
    ax1.scatter(df.x[0],df.y[0], s=30, c='r', marker="o", label='first') # type: ignore
    plt.legend(loc='upper left');
    plt.show()


def raster2shape(segmented_map: ndarray, dest: Path, affine: Affine) -> None:
    geometries = [
            {'properties': {'raster_val': v}, 'geometry': polygons}
            for polygons, v
            in shapes(
                segmented_map.astype('uint8'), 
                mask=segmented_map!=0, 
                transform=affine)]

    # geometries = list(vectorized)
    # _imageCoord_to_lonLat(transform, geometries) # geographic geometries

    crs = 'EPSG:4326'
    # Vector analysis with geopanda
    gpSeries = GeoSeries(
        [shape(geom['geometry']) 
        for geom in geometries])
    gpDFrame = GeoDataFrame.from_features(gpSeries.geometry, crs=crs)
    # write shapefile
    gpDFrame.to_file(dest, driver ='ESRI Shapefile')


def _imageCoord_to_lonLat(transform, geometries):
    '''
    Applies/transforms the coordinates of polygon geometries
    '''
    for geom in geometries:
        for polygon in geom['geometry']['coordinates']:
            for i, col_row in enumerate(polygon):
                polygon[i] = _colRow_to_lonLat(transform, *col_row)


def _colRow_to_lonLat(transform, col, row):
    return  (rio.transform.xy(transform, col, row))  # type: ignore