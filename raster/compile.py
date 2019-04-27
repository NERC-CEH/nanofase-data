#!/usr/bin/env python
"""Compiling data sources into NetCDF file for use in NanoFASE model."""
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform
from rasterio.crs import CRS
from shapely.geometry import box
from netCDF4 import Dataset
from pint import UnitRegistry
import pandas as pd
import yaml
import numpy as np
import sys
import util
from compiler import Compiler

compiler = Compiler()
compiler.parse_flow_dir()
compiler.setup_netcdf_dataset()

# with open('./config.yaml', 'r') as f:
#     try:
#         config = yaml.load(f, Loader=yaml.BaseLoader)
#     except yaml.YAMLError as e:
#         print(e)

# # Get the list of variables and their names, units etc
# vars, vars_constant, vars_spatial, vars_spatiotemporal = util.var_lookup(config)

# # Setup the unit registry
# ureg = UnitRegistry()
# # Define the timestep as a unit, based on that given in config file
# ureg.define('timestep = {0} * second'.format(config['time']['dt']))

# Flow direction first - this defines the bounds of the grid we're working to
# print("Creating variables...\n\t...flow_dir")
# if config['flow_dir']['type'] == 'raster':
#     grid = rasterio.open(config['flow_dir']['path'])
#     # If a CRS has been specified for the flowdir raster, use this instead of the raster's
#     # internal CRS. This is useful if a raster has an ill-defined CRS
#     grid_crs = CRS.from_user_input(config['flow_dir']['crs']) if 'crs' in config['flow_dir'] else grid.crs
#     # Only projected rasters allowed for the moment
#     if grid_crs.is_geographic:
#         print('Sorry, the flow_dir raster must be projected, not geographic. I got a geographic CRS: \'{0}\'.'.format(grid_crs))
#         sys.exit()
#     grid_bbox = box(*grid.bounds)                           # Create Shapely box from bounds, to clip other rasters to
# elif config['flow_dir']['type'] == 'csv':
#     print("CSV flow direction not currently supported. Please provide a raster file.")
#     sys.exit()
# else:
#     print("Unrecognised file type {0} for variable flow_dir. Type should be rs, csv or nc.".format(config['flow_dir']['type']))
#     sys.exit()

# Now we have our grid structure, we can create the NetCDF4 output file, set some dimensions,
# and route the reaches
# nc, grid_mask = util.setup_netcdf_dataset(config, grid, grid_crs)

# Spatial, non-temporal data
for var in compiler.vars_spatial:
    print('\t...{0}'.format(var))
    compiler.parse_spatial_var(var)
    # # Is the data supplied in raster or csv form?
    # if config[var]['type'] == 'raster':
    #     # Open the raster and clip to extent of grid (defined by flowdir raster)
    #     with rasterio.open(config[var]['path'], 'r') as rs:
    #         out_img, out_transform = mask(rs, [grid_bbox], crop=True)
    #     values = np.ma.masked_where(grid_mask, out_img[0])
    #     # Should the array be clipped?
    #     if 'clip' in vars[var]:
    #         try:
    #             min = float(vars[var]['clip'][0])
    #         except ValueError:
    #             min = None
    #         try:
    #             max = float(vars[var]['clip'][1])
    #         except ValueError:
    #             max = None
    #         np.clip(values, min, max, out=values)
    #     # Check if we're converting units
    #     from_units = ureg(vars[var]['units'] if 'units' in vars[var] else vars[var]['to_units'])
    #     to_units = ureg(vars[var]['to_units'])
    #     if from_units != to_units:
    #         print('\t\t...converting {0.units:~P} to {1.units:~P}'.format(from_units, to_units))
    #     # Do the conversion
    #     values = from_units * values
    #     values.ito(to_units)
    #     # Fill the NetCDF variable with the clipped raster (without the units)
    #     nc_var[:] = values.magnitude
    # elif config[var]['type'] == 'csv':
    #     # TODO
    #     pass
    # else: 
    #     print("Unrecognised file type {0} for variable {1}. Type should be rs, csv or nc.".format(config[var]['type'], var))

# Spatial, temporal data
for var in compiler.vars_spatiotemporal:
    print('\t...{0}'.format(var))
    compiler.parse_spatiotemporal_var(var)
    # # Is this a raster or CSV?
    # if vars[var]['type'] == 'raster':
    #     values = []
    #     nc_var = util.setup_netcdf_var(var, vars[var], nc)
    #     # Check if we're converting units (the actual converting is done per timestep, below)
    #     from_units = ureg(vars[var]['units'] if 'units' in vars[var] else vars[var]['to_units'])
    #     to_units = ureg(vars[var]['to_units'])
    #     if from_units != to_units:
    #         print('\t\t...converting {0.units:~P} to {1.units:~P}'.format(from_units, to_units))
    #     # If the {t} tag is in the path, there must be one raster file per timestep
    #     if '{t}' in vars[var]['path']:
    #         t_min = 0 if 't_min' not in vars[var] else int(vars[var]['t_min'])
    #         for t in range(t_min, int(config['time']['n']) + t_min):
    #             with rasterio.open(vars[var]['path'].replace('{t}',str(t)), 'r') as rs:
    #                 out_img, out_transform = mask(rs, [grid_bbox], crop=True)
    #             values = np.ma.masked_where(grid_mask, out_img[0])
    #             # Should the array be clipped?
    #             if 'clip' in vars[var]:
    #                 try:
    #                     min = float(vars[var]['clip'][0])
    #                 except ValueError:
    #                     min = None
    #                 try:
    #                     max = float(vars[var]['clip'][1])
    #                 except ValueError:
    #                     max = None
    #                 np.clip(values, min, max, out=values)
    #             # Convert units if "units" specified in config, to the to_units in model_vars
    #             values = from_units * values
    #             values.ito(to_units)
    #             # Add this time step to the NetCDF file as a masked array
    #             nc_var[t-1,:,:] = values.magnitude
       
    # elif vars[var]['type'] == 'csv':
    #     df = pd.read_csv(config[var]['path'], header=0)
    #     values = []
    #     nc_var = util.setup_netcdf_var(var, vars[var], nc)
    #     # Check if we're converting units (the actual converting is done per timestep, below)
    #     from_units = ureg(vars[var]['units'] if 'units' in vars[var] else vars[var]['to_units'])
    #     to_units = ureg(vars[var]['to_units'])
    #     if from_units != to_units:
    #         print('\t\t...converting {0.units:~P} to {1.units:~P}'.format(from_units, to_units))
    #     # Loop through the timesteps
    #     for t in range(1,df['t'].max()+1):
    #         df_t = df[df['t'] == t]
    #         pt = df_t.pivot_table(index='y', columns='x', values=var)
    #         values = np.ma.masked_where(grid_mask, pt.values)
    #         # Check the pivot table's shape is that of the grid we're using
    #         if pt.shape != grid.shape:
    #             print("Inconsistent shape between {0} csv file and overall grid system ({1} and {2}). Check indices set correctly.".format(var, pt.shape, grid.shape))
    #             sys.exit()
    #         # Should the array be clipped?
    #         if 'clip' in vars[var]:
    #             try:
    #                 min = float(vars[var]['clip'][0])
    #             except ValueError:
    #                 min = None
    #             try:
    #                 max = float(vars[var]['clip'][1])
    #             except ValueError:
    #                 max = None
    #             np.clip(values, min, max, out=values)
    #         # Convert units if "units" specified in config, to the to_units in model_vars
    #         values = from_units * values
    #         values.ito(to_units)
    #         # Add this time step to the NetCDF file as a masked array
    #         nc_var[t-1,:,:] = values.magnitude
    # else:
    #     print("Unrecognised file type {0} for variable {1}. Type should be rs, csv or nc.".format(config[var]['type'], var))
