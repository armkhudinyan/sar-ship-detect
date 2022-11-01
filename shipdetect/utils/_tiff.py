from pathlib import Path
from typing import Tuple
from rasterio.io import DatasetReader
from rasterio.crs import CRS
from rasterio import Affine
import rasterio as rio
from numpy import ndarray


def write_tif(
    raster_array: ndarray,
    src: DatasetReader,
    transform: Affine,
    crs: CRS,
    dest_file: Path,
) -> None:
    """Write  raster files into GeoTif

    raster_array : 2d array
    src          : source file
    transform    : geo_transform file
    crs          : coordinate reference system, e.g. 'EPSG':4326
    dest_file    : destination file path and name
    """
    # get output raster size
    height, width = raster_array.shape
    # define the output raster parameters
    out_meta = src.meta.copy()
    export_tif = raster_array.astype("float32")
    # update the metadata
    out_meta.update(
        {
            "dtype": "float32",
            "width": width,
            "height": height,
            "crs": crs,  # CRS.from_epsg(4326), #out_meta['crs'],
            "transform": transform,
        }
    )
    # Write the array to raster GeoTIF
    with rio.open(dest_file, "w", **out_meta) as dest:
        dest.write(export_tif, 1)  # data array should be in (band, rows, cols) order
    print(f"saved tif: {dest}")


def GCPs2GeoTransform(
    src: DatasetReader, scale_factor: float = 1, gt_style: str = "Affine"
) -> Tuple[Affine, CRS]:
    """
    Reads GCP's with `rasterio` and returns geotransform
    in either `Affine` style or in `GDAL_geotransform` style
    and `crs` data
    """
    GCPs = src.get_gcps()[0]
    transf_from_gcps = rio.transform.from_gcps(GCPs)  # type: ignore

    if gt_style == "Affine":
        transform = rio.Affine(
            # X-coord / long
            transf_from_gcps.a * scale_factor,
            transf_from_gcps.b * scale_factor,
            transf_from_gcps.c,
            # Y-coord / lat
            transf_from_gcps.d * scale_factor,
            transf_from_gcps.e * scale_factor,
            transf_from_gcps.f,
        )
    elif gt_style == "GDAL":
        transform = rio.Affine(
            # X-coord / long
            transf_from_gcps.c * scale_factor,
            transf_from_gcps.a * scale_factor,
            transf_from_gcps.b,
            # Y-coord / lat
            transf_from_gcps.f * scale_factor,
            transf_from_gcps.d * scale_factor,
            transf_from_gcps.e,
        )
    else:
        raise ValueError("Wrong input for 'gt_style', use 'Affine' or 'GDAL'")
    crs = src.get_gcps()[1]
    return transform, crs
