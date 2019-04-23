from netCDF4 import Dataset
import numpy as np
import yaml
import sys

def setup_netcdf_dataset(config, grid, crs):
    """Create NetCDF file, add required dimensions and coordinate variables"""
    nc = Dataset(config['output_file'], 'w', format='NETCDF4')
    nc.title = "Input data for NanoFASE model"
    nc.Conventions = 'CF-1.6'
    crs_var = nc.createVariable('crs','i4')
    crs_var.spatial_ref = crs.to_wkt()          # QGIS/ArcGIS recognises spatial_ref to define CRS
    crs_var.crs_wkt = crs.to_wkt()              # Latest CF conventions say crs_wkt can be used
    # Time dimensions and coordinate variable
    t_dim = nc.createDimension('t', None)
    t = nc.createVariable('t','i4',('t',))
    t.units = "seconds since {0} 00:00:00".format(config['time']['start_date'])
    t.standard_name = 'time'
    t.calendar = 'gregorian'
    t[:] = [i*int(config['time']['dt']) for i in range(int(config['time']['n']))]
    # x dimension and coordinate variable
    x_dim = nc.createDimension('x', grid.width)
    x = nc.createVariable('x','f4',('x',))
    x.units = 'm'
    x.standard_name = 'projection_x_coordinate'
    x.axis = 'X'
    x[:] = [grid.bounds.left + i * grid.res[0] for i in range(grid.width)]
    # y dimension and coordinate variable
    y_dim = nc.createDimension('y', grid.height)
    y = nc.createVariable('y','f4',('y',))
    y.units = 'm'
    y.standard_name = 'projection_y_coordinate'
    y.axis = 'Y'
    y[:] = [grid.bounds.top - i * grid.res[1] for i in range(grid.height)]
    # Add the flow direction
    flow_dir = grid.read(1, masked=True)                # Get the flow direction array from the raster
    grid_mask = np.ma.getmask(flow_dir)                       # Use the extent of the flow direction array to create a mask for all other data
    nc_var = nc.createVariable('flow_dir', 'i4', ('y', 'x'))
    nc_var.long_name = 'flow direction of water in grid cell'
    nc_var[:] = flow_dir
    # Route the reaches using the flow direction
    routed_reaches = routing(flow_dir)
    # Return the NetCDF file
    return nc, grid_mask


def outflow_from_flow_dir(flow_dir, x, y):
    """Get the outflow cell reference given the current cell
    reference and a flow direction."""
    xy_out = {
        1: (x+1, y),
        2: (x+1, y+1),
        4: (x, y+1),
        8: (x-1, y+1),
        16: (x-1, y),
        32: (x-1, y-1),
        64: (x, y-1),
        128: (x+1, y-1)
    }
    return xy_out[flow_dir]


def routing(flow_dir_array):
    pass
    # x, y = np.meshgrid(np.arange(flow_dir_array.shape[0]), np.arange(flow_dir_array.shape[1]))
    # outflow_array = outflow_from_flow_dir(flow_dir_array, x, y)
    # outflow_var = setup_netcdf_var('outflow')
    # for x, xy in enumerate(flow_dir_array):
    #     for y, flow_dir in enumerate(xy):
    #         if flow_dir[x][y] != np.ma.masked:
    #             # Create an outflow array



def setup_netcdf_var(var_name, var_dict, nc):
    """Create and fill attributes in NetCDF file for given variable."""
    fill_value = float(var_dict['fill_value']) if 'fill_value' in var_dict else None
    dims = tuple(var_dict['dims']) if 'dims' in var_dict else ()
    vartype = var_dict['vartype'] if 'vartype' in var_dict else 'f4'
    nc_var = nc.createVariable(var_name, vartype, dims, fill_value=fill_value)
    if 'standard_name' in var_dict:
        nc_var.standard_name = var_dict['standard_name']
    if 'long_name' in var_dict:
        nc_var.long_name = var_dict['long_name']
    if 'source' in var_dict:
        nc_var.source = var_dict['source']
    if 'references' in var_dict:
        nc_var.references = var_dict['references']
    nc_var.units = var_dict['to_units']
    nc_var.grid_mapping = 'crs'
    return nc_var

def var_lookup(config):
    # Open the internal config file to create variable attributes (model_vars.yaml)
    with open('./model_vars.yaml', 'r') as f:
        try:
            vars = yaml.load(f, Loader=yaml.BaseLoader)
        except yaml.YAMLError as e:
            print(e)
    # Add config options to the user-defined var dict (config.yaml)
    for k, v in config.items():
        if k in vars:
            vars[k].update(v)
    # Get a list of constants, spatial and spatiotemporal variables
    vars_constant = [k for k, v in vars.items() if ('dims' not in v) or (v['dims'] == None)]
    vars_spatial = [k for k, v in vars.items() if ('dims' in v) and (v['dims'] == ['y', 'x'])]
    vars_spatiotemporal = [k for k, v in vars.items() if ('dims' in v) and (v['dims'] == ['t', 'y', 'x'])]
    return vars, vars_constant, vars_spatial, vars_spatiotemporal