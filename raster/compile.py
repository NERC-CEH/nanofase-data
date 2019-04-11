#!/usr/bin/env python
"""Compiling data sources into NetCDF file for use in NanoFASE model."""
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform
from rasterio.crs import CRS
from shapely.geometry import box
from netCDF4 import Dataset
import pandas as pd
import yaml
import numpy as np
import sys
import util

with open('config.yaml', 'r') as f:
    try:
        config = yaml.load(f, Loader=yaml.BaseLoader)
    except yaml.YAMLError as e:
        print(e)

# Get the list of variables and their names, units etc
var_lookup = util.var_lookup()

# Flow direction first - this defines the bounds of the grid we're working to
if config['flow_dir']['type'] == 'raster':
    grid = rasterio.open(config['flow_dir']['path'])
    # If a CRS has been specified for the flowdir raster, use this instead of the raster's
    # internal CRS. This is useful if a raster has an ill-defined CRS
    grid_crs = CRS.from_user_input(config['flow_dir']['crs']) if 'crs' in config['flow_dir'] else grid.crs
    # Only projected rasters allowed for the moment
    if grid_crs.is_geographic:
        print('Sorry, the flow_dir raster must be projected, not geographic. I got a geographic CRS: \'{0}\'.'.format(grid_crs))
        sys.exit()
    grid_bbox = box(*grid.bounds)                           # Create Shapely box from bounds, to clip other rasters to
    flowdir_arr = grid.read(1, masked=True)                 # Get the flow direction array from the raster
    grid_mask = np.ma.getmask(flowdir_arr)                  # Use the extent of the flow direction array to create a mask for all other data


elif config['flow_dir']['type'] == 'csv':
    print("CSV flow direction not currently supported. Please provide a raster file.")
    sys.exit()
else:
    print("Unrecognised file type {0} for variable flow_dir. Type should be rs, csv or nc.".format(config['flow_dir']['type']))
    sys.exit()

# Now we have our grid structure, we can create the NetCDF4 output file and set some dimensions
nc = util.setup_netcdf_dataset(config, grid, grid_crs)

# Spatial, non-temporal data
for var in ['soil_bulk_density']:
    # Create the variable in the NetCDF file
    nc_var = nc.createVariable(var, 'f4',('y','x'))
    nc_var.grid_mapping = 'crs'
    if 'standard_name' in var_lookup[var]:
        nc_var.standard_name = var_lookup[var]['standard_name']
    nc_var.units = var_lookup[var]['units']
    # Is the data supplied in raster or csv form?
    if config[var]['type'] == 'raster':
        # Open the raster and clip to extent of grid (defined by flowdir raster)
        with rasterio.open(config[var]['path'], 'r') as rs:
            out_img, out_transform = mask(rs, [grid_bbox], crop=True)
        # Fill the NetCDF variable with the clipped raster
        nc_var[:] = np.ma.masked_where(grid_mask, out_img[0])
    elif config[var]['type'] == 'csv':
        # TODO
        pass
    else: 
        print("Unrecognised file type {0} for variable {1}. Type should be rs, csv or nc.".format(config[var]['type'], var))

# Spatial, temporal data
for var in ['runoff']:
    
    # Is this a raster or CSV?
    if config[var]['type'] == 'raster':
        rs = rasterio.open(config[var]['path'], 'r')
        out_img, out_transform = mask(rs, [grid_bbox], crop=True)
        # TODO need to make this temporal
        # values.append(out_img)
    elif config[var]['type'] == 'csv':
        df = pd.read_csv(config[var]['path'], header=0)
        values = []
        nc_var = nc.createVariable(var, 'f4', ('t','y','x'))
        nc_var.standard_name = var_lookup[var]['standard_name']
        nc_var.units = var_lookup[var]['units']
        nc_var.grid_mapping = 'crs'
        for t in range(1,df['t'].max()+1):
            df_t = df[df['t'] == t]
            pt = df_t.pivot_table(index='y', columns='x', values=var)
            # Check the pivot table's shape is that of the grid we're using
            if pt.shape != grid.shape:
                print("Inconsistent shape between {0} csv file and overall grid system ({1} and {2}). Check indices set correctly.".format(var, pt.shape, grid.shape))
                sys.exit()
            # Add this time step to the NetCDF file as a masked array
            nc_var[t-1,:,:] = np.ma.masked_where(grid_mask, pt.values)
    else:
        print("Unrecognised file type {0} for variable {1}. Type should be rs, csv or nc.".format(config[var]['type'], var))
