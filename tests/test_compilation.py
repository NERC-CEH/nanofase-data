"""
Tests for compiling input data, largely using the example dataset
"""
from pathlib import Path
import pytest
import numpy as np
from nfdata.compiler import Compiler
from netCDF4 import Dataset


@pytest.fixture
def example_config():
    return Path(__file__).parent.parent / 'config.create.example.yaml'


def test_compiling_example_data(example_config, tmpdir):
    """
    Test that compiling the example data works as expected
    """
    # Init the compiler with the example config file
    compiler = Compiler('create', example_config)
    # Set the output files to be in the temporary directory
    compiler.config['output']['nc_file'] = str(tmpdir / 'data.nc')
    compiler.config['output']['constants_file'] = str(tmpdir / 'constants.nml')
    try:
        # Create the NetCDF and constants files. If there is no error
        # then the test should pass
        compiler.create()
        assert True
    except Exception:
        # If an exception has been raised, this test should fail
        assert False


def test_compiling_example_data_without_flowdir(example_config, tmpdir):
    """
    Test that compiling the example data without flow direction
    works as expected
    """
    # Init the compiler with the example config file
    compiler = Compiler('create', example_config)
    # Set the output files to be in the temporary directory
    compiler.config['output']['nc_file'] = str(tmpdir / 'data.nc')
    compiler.config['output']['constants_file'] = str(tmpdir / 'constants.nml')
    # Remove the flow_dir from the config
    del compiler.config['flow_dir']
    # Run the compiler
    compiler.create()
    # Load the NetCDF file and check that flow_dir is present
    # and doesn't contain any 0 values
    with Dataset(compiler.config['output']['nc_file']) as nc:
        flow_dir = nc.variables['flow_dir'][:]
        # Check that the flow_dir is not all zeros
        assert np.any(flow_dir != 0)
