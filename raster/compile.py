#!/usr/bin/env python
"""Compiling data sources into NetCDF file for use in NanoFASE model."""
from compiler import Compiler
import sys

if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    config_file = 'config.yaml'

compiler = Compiler(config_file, 'model_vars.yaml')
print("Setting up dataset...")
print('\t...parsing flow_dir')
compiler.parse_flow_dir()
print('\t...creating NetCDF file')
compiler.setup_netcdf_dataset()
compiler.parse_spatial_var('is_estuary', save=True)     # Tidal bounds need setting before routing
compiler.vars_spatial.remove('is_estuary')
print('\t...routing water bodies')
compiler.routing()

print("Creating variables...")

# Constants
print('\t...constants')
compiler.parse_constants()

# Spatial, non-temporal data
for var in compiler.vars_spatial:
    print('\t...{0}'.format(var))
    compiler.parse_spatial_var(var)

# Spatial data with 1 record dimension (that isn't time)
for var in compiler.vars_spatial_1d:
    print('\t...{0}'.format(var))
    compiler.parse_spatial_1d_var(var)

# Spatial point data
for var in compiler.vars_spatial_point:
	print('\t...{0}'.format(var))
	compiler.parse_spatial_point_var(var)

# Spatial, temporal data
for var in compiler.vars_spatiotemporal:
    print('\t...{0}'.format(var))
    compiler.parse_spatiotemporal_var(var)

print("Done! Data saved to data.nc.")