from netCDF4 import Dataset
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from shapely.geometry import box
import yaml
from pint import UnitRegistry
import sys

class Compiler:

    def __init__(self, config_path, model_vars_path):
        """Initialise the compiler by reading the config and model_var files,
        combining these and generating list of vars according to dimensionality.
        Also set up the unit registry and define 'timestep' as a unit."""
        with open(config_path, 'r') as config_file, open(model_vars_path, 'r') as model_vars_file:
            try:
                self.config = yaml.load(config_file, Loader=yaml.BaseLoader)
                self.vars = yaml.load(model_vars_file, Loader=yaml.BaseLoader)
            except yaml.YAMLError as e:
                print(e)
        # Combine config and model_vars
        for k, v in self.config.items():
            if k in self.vars:
                self.vars[k].update(v)
        # Get a list of constants, spatial and spatiotemporal variables
        self.vars_constant = []
        self.vars_spatial = []
        self.vars_spatial_1d = []
        self.vars_spatiotemporal = []
        for k, v in self.vars.items():
            if ('dims' not in v) or (v['dims'] == None):
                self.vars_constant.append(k)
            elif ('dims' in v) and (v['dims'] == ['y', 'x']):
                self.vars_spatial.append(k)
            elif ('dims' in v) and (v['dims'] == ['t', 'y', 'x']):
                self.vars_spatiotemporal.append(k)
            elif ('dims' in v) and (all(x in v['dims'] for x in ['y', 'x'])) and (len(v['dims']) == 3):
                self.vars_spatial_1d.append(k)
        # Setup the unit registry
        self.ureg = UnitRegistry()
        # Define the timestep as a unit, based on that given in config file
        self.ureg.define('timestep = {0} * second'.format(self.config['time']['dt']))
    

    def parse_flow_dir(self):
        """Parse the flow direction variable and define the CRS and mask for other
        variables based on this. This step must be performed before NetCDF file is generated."""
        if self.config['flow_dir']['type'] == 'raster':
            self.grid = rasterio.open(self.config['flow_dir']['path'])
            # If a CRS has been specified for the flowdir raster, use this instead of the raster's
            # internal CRS. This is useful if a raster has an ill-defined CRS
            self.grid_crs = CRS.from_user_input(self.config['flow_dir']['crs']) if 'crs' in self.config['flow_dir'] else self.grid.crs
            # Only projected rasters allowed for the moment
            if self.grid_crs.is_geographic:
                print('Sorry, the flow_dir raster must be projected, not geographic. I got a geographic CRS: \'{0}\'.'.format(self.grid_crs))
                sys.exit()
            self.grid_bbox = box(*self.grid.bounds)                         # Create Shapely box from bounds, to clip other rasters to
        elif self.config['flow_dir']['type'] == 'csv':
            print("CSV flow direction not currently supported. Please provide a raster file.")
            sys.exit()
        else:
            print("Unrecognised file type {0} for variable flow_dir. Type should be rs, csv or nc.".format(self.config['flow_dir']['type']))
            sys.exit()


    def setup_netcdf_dataset(self):
        """Create NetCDF file, add required dimensions, coordinate variables and the
        flow direction variable."""
        self.nc = Dataset(self.config['output_file'], 'w', format='NETCDF4')
        self.nc.title = "Input data for NanoFASE model"
        self.nc.Conventions = 'CF-1.6'
        crs_var = self.nc.createVariable('crs','i4')
        crs_var.spatial_ref = self.grid_crs.to_wkt()            # QGIS/ArcGIS recognises spatial_ref to define CRS
        crs_var.crs_wkt = self.grid_crs.to_wkt()                # Latest CF conventions say crs_wkt can be used
        # Time dimensions and coordinate variable
        t_dim = self.nc.createDimension('t', None)
        t = self.nc.createVariable('t','i4',('t',))
        t.units = "seconds since {0} 00:00:00".format(self.config['time']['start_date'])
        t.standard_name = 'time'
        t.calendar = 'gregorian'
        t[:] = [i*int(self.config['time']['dt']) for i in range(int(self.config['time']['n']))]
        # x dimension and coordinate variable
        x_dim = self.nc.createDimension('x', self.grid.width)
        x = self.nc.createVariable('x','f4',('x',))
        x.units = 'm'
        x.standard_name = 'projection_x_coordinate'
        x.axis = 'X'
        x[:] = [self.grid.bounds.left + i * self.grid.res[0] for i in range(self.grid.width)]
        # y dimension and coordinate variable
        y_dim = self.nc.createDimension('y', self.grid.height)
        y = self.nc.createVariable('y','f4',('y',))
        y.units = 'm'
        y.standard_name = 'projection_y_coordinate'
        y.axis = 'Y'
        y[:] = [self.grid.bounds.top - i * self.grid.res[1] for i in range(self.grid.height)]
        # Add the flow direction
        flow_dir = self.grid.read(1, masked=True)                   # Get the flow direction array from the raster
        self.grid_mask = np.ma.getmask(flow_dir)                    # Use the extent of the flow direction array to create a mask for all other data
        nc_var = self.nc.createVariable('flow_dir', 'i4', ('y', 'x'))
        nc_var.long_name = 'flow direction of water in grid cell'
        nc_var[:] = flow_dir
        # Route the reaches using the flow direction
        # routed_reaches = routing(flow_dir)

    def setup_netcdf_var(self, var_name):
        var_dict = self.vars[var_name]
        fill_value = float(var_dict['fill_value']) if 'fill_value' in var_dict else None
        dims = tuple(var_dict['dims']) if 'dims' in var_dict else ()
        vartype = var_dict['vartype'] if 'vartype' in var_dict else 'f4'
        nc_var = self.nc.createVariable(var_name, vartype, dims, fill_value=fill_value)
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


    def parse_raster(self, var_name, units, path=None):
        """Parse a variable (or timestep of a variable) given by raster."""
        var_dict = self.vars[var_name]
        if path is None:
            path = var_dict['path']
        # Open the raster and clip to extent of grid (defined by flowdir raster)
        with rasterio.open(path, 'r') as rs:
            out_img, out_transform = mask(rs, [self.grid_bbox], crop=True, filled=False)
        values = np.ma.masked_where(self.grid_mask, out_img[0])
        # Should the array be clipped?
        if 'clip' in var_dict:
            try:
                min = float(var_dict['clip'][0])
            except ValueError:
                min = None
            try:
                max = float(var_dict['clip'][1])
            except ValueError:
                max = None
            np.clip(values, min, max, out=values)
        # Do the unit conversion
        values = units[0] * values
        values.ito(units[1])
        # Return the values
        return values


    def parse_spatial_var(self, var_name):
        """Create and fill attributes in NetCDF file for given variable."""
        nc_var = self.setup_netcdf_var(var_name)
        var_dict = self.vars[var_name]

         # Check if we're converting units
        from_units = self.ureg(var_dict['units'] if 'units' in var_dict else var_dict['to_units'])
        to_units = self.ureg(var_dict['to_units'])
        if from_units != to_units:
            print('\t\t...converting {0.units:~P} to {1.units:~P}'.format(from_units, to_units))

        # Is the data supplied in raster or csv form?
        if var_dict['type'] in ['raster', 'nc']:
            # Parse the raster
            values = self.parse_raster(var_name, (from_units, to_units))
            # Fill the NetCDF variable with the clipped raster (without the units)
            nc_var[:] = values.magnitude
        elif var_dict['type'] == 'csv':
            # TODO
            print("Sorry, only raster spatial variables supported at the moment. Variable: {0}.".format(var_name))
        else: 
            print("Unrecognised file type {0} for variable {1}. Type should be raster, csv or nc.".format(var_dict['type'], var_name))


    def parse_spatiotemporal_var(self, var_name):
        nc_var = self.setup_netcdf_var(var_name)
        var_dict = self.vars[var_name]

        # Check if we're converting units (the actual converting is done per timestep, below)
        from_units = self.ureg(var_dict['units'] if 'units' in var_dict else var_dict['to_units'])
        to_units = self.ureg(var_dict['to_units'])
        if from_units != to_units:
            print('\t\t...converting {0.units:~P} to {1.units:~P}'.format(from_units, to_units))

        # Is this a raster or CSV?
        if var_dict['type'] == 'raster':
            # If the {t} tag is in the path, there must be one raster file per timestep
            if '{t}' in var_dict['path']:
                # Zero-indexed or higher?
                t_min = 0 if 't_min' not in var_dict else int(var_dict['t_min'])
                # Loop through the time steps and parse raster for each
                for t in range(t_min, int(self.config['time']['n']) + t_min):
                    path = var_dict['path'].replace('{t}',str(t))
                    values = self.parse_raster(var_name, (from_units, to_units), path)
                    # Add this time step to the NetCDF file as a masked array
                    nc_var[t-1,:,:] = values.magnitude
            else:
                print("Spatiotemporal variable ({0}) in raster format must be provided by one raster file per time step, with path denoted by /{t/}".format(var_name))
           
        elif var_dict['type'] == 'csv':
            df = pd.read_csv(var_dict['path'], header=0)

            # Loop through the timesteps and create pivot table to obtain spatial array for each
            for t in range(1,df['t'].max()+1):
                df_t = df[df['t'] == t]
                pt = df_t.pivot_table(index='y', columns='x', values=var_name)
                values = np.ma.masked_where(self.grid_mask, pt.values)
                # Check the pivot table's shape is that of the grid we're using
                if pt.shape != self.grid.shape:
                    print("Inconsistent shape between {0} csv file and overall grid system ({1} and {2}). Check indices set correctly.".format(var, pt.shape, grid.shape))
                    sys.exit()
                # Should the array be clipped?
                if 'clip' in var_dict:
                    try:
                        min = float(var_dict['clip'][0])
                    except ValueError:
                        min = None
                    try:
                        max = float(var_dict['clip'][1])
                    except ValueError:
                        max = None
                    np.clip(values, min, max, out=values)
                # Convert units if "units" specified in config, to the to_units in model_vars
                values = from_units * values
                values.ito(to_units)
                # Add this time step to the NetCDF file as a masked array
                nc_var[t-1,:,:] = values.magnitude
        else:
            print("Unrecognised file type {0} for variable {1}. Type should be rs, csv or nc.".format(config[var]['type'], var))