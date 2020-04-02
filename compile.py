#!/usr/bin/env python
"""Compiling data sources into NetCDF file for use in NanoFASE model."""
from compiler import Compiler
import sys
import os

# Get config file location from command line args
if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    sys.exit('You must provide a config file as a command line argument.')

path = sys.path[0]          # Path to script directory for model_vars.yaml file
compiler = Compiler(config_file, os.path.join(path, 'model_vars.yaml'))

print("Setting up dataset...")
print('\t...parsing flow_dir')
compiler.parse_flow_dir()                               # Used to create grid and hence needed to set up NetCDF file
print('\t...creating NetCDF file')
compiler.setup_netcdf_dataset()
compiler.parse_spatial_var('is_estuary', save=True)     # Tidal bounds need setting before routing
compiler.vars_spatial.remove('is_estuary')              # Make sure we don't create is_estuary twice
# Route the waterbody network, including setting inflows, outflows and headwaters
print('\t...routing water bodies')
compiler.routing()

print("Creating variables...")

# Constants
print('\t...constants')
compiler.parse_constants()

# Spatial, non-temporal data
for var in compiler.vars_spatial:
    print(f'\t...{var}')
    compiler.parse_spatial_var(var)

# Spatial data with 1 record dimension (that isn't time)
for var in compiler.vars_spatial_1d:
    print(f'\t...{var}')
    compiler.parse_spatial_1d_var(var)

# Spatial point data
for var in compiler.vars_spatial_point:
    print(f'\t...{var}')
    compiler.parse_spatial_point_var(var)

# Spatiotemporal data
for var in compiler.vars_spatiotemporal:
    print(f'\t...{var}')
    compiler.parse_spatiotemporal_var(var)

print(f'Done! Data saved to...\n\t'
      f'...{compiler.config["output"]["nc_file"]}\n\t'
      f'...{compiler.config["output"]["constants_file"]}')
