#!/usr/bin/env python
"""Compiling data sources into NetCDF file for use in NanoFASE model."""
from compiler import Compiler

compiler = Compiler('config.yaml', 'model_vars.yaml')
print("Creating variables...\n\t...flow_dir")
compiler.parse_flow_dir()
compiler.setup_netcdf_dataset()

# Spatial, non-temporal data
for var in compiler.vars_spatial:
    print('\t...{0}'.format(var))
    compiler.parse_spatial_var(var)

for var in compiler.vars_spatial_1d:
    print('\t...{0}...NOT YET!'.format(var))

# Spatial, temporal data
for var in compiler.vars_spatiotemporal:
    print('\t...{0}'.format(var))
    compiler.parse_spatiotemporal_var(var)
