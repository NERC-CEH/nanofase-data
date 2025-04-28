"""
Utility functions for nfdata.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import rasterio
from affine import Affine
from rasterio.coords import BoundingBox
from rasterio.crs import CRS
from shapely.geometry import Polygon, box


@dataclass
class RasterInfo:
    bounds: BoundingBox
    crs: CRS
    transform: Affine
    res: Tuple[float, float]
    width: int
    height: int
    dtype: str
    count: int
    shape: tuple
    nodata: Optional[float]
    bbox: Polygon
    arr: Union[np.ndarray, np.ma.MaskedArray]


def extract_raster_info(path: Union[str, Path],
                        masked: bool = True) -> RasterInfo:
    """
    Extract data and info from a raster file and store
    it in a RasterInfo dataclass so that it can be closed
    """
    with rasterio.open(path) as src:
        return RasterInfo(
            bounds=src.bounds,
            crs=src.crs,
            transform=src.transform,
            res=src.res,
            width=src.width,
            height=src.height,
            dtype=src.dtypes[0],
            count=src.count,
            shape=src.shape,
            nodata=src.nodata,
            bbox=box(*src.bounds),
            arr=src.read(1, masked=masked)
        )
