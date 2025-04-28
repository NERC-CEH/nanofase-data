"""
Unit tests for routing processes
"""
import numpy as np
import rasterio
from rasterio.transform import from_origin

import nfdata.routing
from nfdata._errors import RoutingError


def test_calculating_flow_dir_from_dem():
    """
    Test that calculating flow direction from a conditioned
    DEM works as expected
    """
    # An example DEM with a high point in the central cell
    dem = np.array([
        [1,  2,  3],
        [10, 91, 5],
        [5,  43, 42]
    ])
    expected_flow_dir = np.array([
        [128, 128, 128],
        [32,  64,  128],
        [32,  8,   128]
    ])
    flow_dir = nfdata.routing.calculate_flow_dir(dem, res=(1, 1))
    # Assert the the calculated flow direction equals the expected
    np.testing.assert_equal(flow_dir, expected_flow_dir)


def test_calculating_flow_dir_from_unconditioned_dem():
    """
    Test that calculating flow direction from an unconditioned
    DEM (with a pit) without asking for the DEM to be conditioned
    raises an error
    """
    # An example DEM with a low point in the central cell
    dem = np.array([
        [1,  2,  3],
        [10, -5, 5],
        [5,  43, 42]
    ])
    try:
        _ = nfdata.routing.calculate_flow_dir(dem, res=(1, 1))
        assert False
    except RoutingError:
        # If an exception has been raised, this test should pass
        assert True


def test_calculating_flow_dir_from_dem_with_nodata():
    """
    Test that calculating flow direction from a conditioned
    DEM with nodata values works as expected. Also tests that
    a masked array is returned with nodata cells masked.
    """
    # An example DEM with a nodata cell. The calculation routine
    # assumes by default that nodata=np.nan
    dem = np.array([
        [np.nan, 2,  3],
        [10,     91, 5],
        [5,      43, 42]
    ])
    expected_flow_dir = np.ma.masked_equal([
        [-1, 128, 128],
        [64, 32,  128],
        [32, 8,   128]
    ], -1)
    flow_dir = nfdata.routing.calculate_flow_dir(dem, res=(1, 1))
    # Assert the the calculated flow direction equals the expected
    np.testing.assert_equal(flow_dir, expected_flow_dir)


def test_calculating_flow_dir_with_dem_conditioning(tmp_path):
    """
    Test that calculating flow direction from a DEM that we
    ask to be conditioned works as expected.
    """
    # An example DEM with a low point in the central cell
    dem = np.array([
        [2,  2,  3],
        [10, 1, 5],
        [5,  43, 42]
    ], dtype=np.float64)
    dem = np.pad(dem, 1, constant_values=-1)
    # Write this to a temporary raster file (because Whitebox
    # requires a file to read from)
    meta = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': dem.shape[1],
        'height': dem.shape[0],
        'count': 1,
        'crs': 'EPSG:4326',
        'transform': from_origin(0, 3, 1, 1),
        'nodata': -9999
    }
    temp_dem = tmp_path / 'dem.tif'
    with rasterio.open(temp_dem, 'w', **meta) as dst:
        dst.write(dem, 1)
    # Do the conditioning
    dem_path, _ = nfdata.routing.condition_dem(
        temp_dem, config={'save_dem_to_path':
                          tmp_path / 'conditioned_dem.tif'},
    )
    # Read the conditioned DEM
    with rasterio.open(dem_path) as src:
        conditioned_dem = src.read(1)
    # For the breaching to have worked, the central value (1) must have
    # been raised above the centre-middle value (2)
    assert conditioned_dem[2, 2] > conditioned_dem[1, 2]
