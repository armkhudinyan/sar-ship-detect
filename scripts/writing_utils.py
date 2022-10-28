from rasterio.features import shapes
from shapely.geometry import shape
from geopandas import GeoDataFrame, GeoSeries

def raster2shape(cluster_map, dest, product, affine):
    geometries = [
            {'properties': {'raster_val': v}, 'geometry': polygons}
            for polygons, v
            in shapes(
                cluster_map.astype('uint8'), 
                mask=cluster_map!=0, 
                transform=affine)]

    # geometries = list(vectorized)
    # NOTE:  write function to transform image coordinated into geo coordinates
    imageCoord_to_lonLat(product, geometries) # geo geometries

    # write shapefile
    crs = 'EPSG:4326'
    # Vector analysis with geopanda
    gpSeries = GeoSeries(
        [shape(geom['geometry']) 
        for geom in geometries])
    gpDFrame = GeoDataFrame.from_features(gpSeries.geometry, crs=crs)
    # write shapefile
    gpDFrame.to_file(dest, driver ='ESRI Shapefile')


