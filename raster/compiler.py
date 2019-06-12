import netCDF4
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import rasterio
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
from shapely.geometry import box
import yaml
from pint import UnitRegistry
import sys
from router import Router

class Compiler:

    def __init__(self, config_path, model_vars_path, land_use_config_path):
        """Initialise the compiler by reading the config and model_var files,
        combining these and generating list of vars according to dimensionality.
        Also set up the unit registry and define 'timestep' as a unit."""
        with open(config_path, 'r') as config_file, open(model_vars_path, 'r') as model_vars_file, open(land_use_config_path, 'r') as land_use_config_file:
            try:
                self.config = yaml.load(config_file, Loader=yaml.BaseLoader)
                self.vars = yaml.load(model_vars_file, Loader=yaml.BaseLoader)
                self.land_use_config = yaml.load(land_use_config_file, Loader=yaml.BaseLoader)
            except yaml.YAMLError as e:
                print(e)
        # Combine config and model_vars
        for k, v in self.config.items():
            if k in self.vars:
                self.vars[k].update(v)
        # Add the separate land use category convertor array
        self.vars['land_use']['cat_conv_dict'] = self.land_use_config
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
        # Create empty dict ready to save vars to memory
        self.saved_vars = {}
    

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
        x[:] = [self.grid.bounds.left + i * self.grid.res[0] + 0.5 * self.grid.res[0] for i in range(self.grid.width)]
        # y dimension and coordinate variable
        y_dim = self.nc.createDimension('y', self.grid.height)
        y = self.nc.createVariable('y','f4',('y',))
        y.units = 'm'
        y.standard_name = 'projection_y_coordinate'
        y.axis = 'Y'
        y[:] = [self.grid.bounds.top - i * self.grid.res[1] - 0.5 * self.grid.res[1] for i in range(self.grid.height)]
        # Grid dimension (2D), max number of waterbodies per cell and grid bounds
        d_dim = self.nc.createDimension('d', 2)
        w_dim = self.nc.createDimension('w', 7)
        box_dim = self.nc.createDimension('box', 4)
        # Grid properties - shape
        grid_shape = self.nc.createVariable('grid_shape', 'i4', ('d',))
        grid_shape.units = ''
        grid_shape.long_name = 'number of grid cells along each (x,y) grid axis'
        grid_shape[:] = self.grid.shape
        # Grid resolution
        grid_res = self.nc.createVariable('grid_res', 'f4', ('d',))
        grid_res.units = ''
        grid_res.long_name = 'number of grid cells along each grid axis'
        grid_res[:] = self.grid.res
        # Grid bounds
        grid_bounds = self.nc.createVariable('grid_bounds', 'f4', ('box',))
        grid_bounds.units = ''
        grid_bounds.long_name = 'bounding box of the grid'
        grid_bounds[:] = self.grid.bounds
        # Add the flow direction
        self.flow_dir = self.grid.read(1, masked=True)              # Get the flow direction array from the raster
        self.grid_mask = np.ma.getmask(self.flow_dir)               # Use the extent of the flow direction array to create a mask for all other data
        nc_var = self.nc.createVariable('flow_dir', 'i4', ('y', 'x'))
        nc_var.long_name = 'flow direction of water in grid cell'
        nc_var[:] = self.flow_dir


    def setup_netcdf_var(self, var_name, extra_dims=None):
        var_dict = self.vars[var_name]
        fill_value = float(var_dict['fill_value']) if 'fill_value' in var_dict else None
        dims = tuple(var_dict['dims']) if 'dims' in var_dict else ()
        # If an extra dimension has been supplied (e.g. a record dim), create this
        # before adding to the variable
        if extra_dims is not None:
            for dim in extra_dims:
                if dim[0] in dims:
                    nc_dim = self.nc.createDimension(dim[0], dim[1])
                else:
                    print("Cannot find extra dimension {0} for NetCDF variable {1} in config file.".format(dim[0], var_name))
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


    def parse_spatial_var(self, var_name, save=False):
        # Create and fill attributes in NetCDF file for given variable.
        nc_var = self.setup_netcdf_var(var_name)
        var_dict = self.vars[var_name]

         # Check if we're converting units
        from_units = self.ureg(var_dict['units'] if 'units' in var_dict else var_dict['to_units'])
        to_units = self.ureg(var_dict['to_units'])
        if from_units != to_units:
            print('\t\t...converting {0.units:~P} to {1.units:~P}'.format(from_units, to_units))

        # Is the data supplied in raster or csv form?
        if var_dict['type'] in ['raster', 'nc']:
            # Parse the raster (clip, convert units)
            values = self.parse_raster(var_name, (from_units, to_units))
            # Fill the NetCDF variable with the clipped raster (without the units)
            nc_var[:] = values.magnitude
            if save:
                self.saved_vars[var_name] = values.magnitude
        elif var_dict['type'] == 'csv':
            # TODO
            print("Sorry, only raster spatial variables supported at the moment. Variable: {0}.".format(var_name))
        else: 
            print("Unrecognised file type {0} for variable {1}. Type should be raster, csv or nc.".format(var_dict['type'], var_name))


    def parse_spatial_1d_var(self, var_name):
        var_dict = self.vars[var_name]
        record_dim = [d for d in var_dict['dims'] if d not in ['x', 'y']][0]    # Get the dim that isn't x or y

        # If this is land use, we need to do some pre-processing first to convert supplied land use categories
        # to the NanoFASE land use categories
        if var_name == 'land_use':
            self.parse_land_use(record_dim)
        else:
            nc_var = self.setup_netcdf_var(var_name)

            from_units = self.ureg(var_dict['units'] if 'units' in var_dict else var_dict['to_units'])
            to_units = self.ureg(var_dict['to_units'])
            if from_units != to_units:
                print('\t\t...converting {0.units:~P} to {1.units:~P}'.format(from_units, to_units))

            if var_dict['type'] == 'raster':
                # If the {record_dim} tag is in the path, there must be one raster per record dim
                if '{' + record_dim + '}' in var_dict['path']:
                    print("Sorry, record dimension of spatial 1d variables must be given as separate bands, for the moment.")
                else:
                    print("one band per record dim")
            else:
                print("Unrecognised file type {0} for variable {1}. Type should be raster for 1d spatial variables.".format(config[var]['type'], var))

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
            print("Unrecognised file type {0} for variable {1}. Type should be raster or csv.".format(config[var]['type'], var))


    def parse_land_use(self, cat_dim):
        """Convert the supplied raster of land use categories to a multi-band array of NanoFASE land
        use categories."""
        var_dict = self.vars['land_use']
        cat_conv_dict = self.vars['land_use']['cat_conv_dict']
        nf_cats = self.vars['land_use']['cats']

        # Open the supplied land use raster
        # Open the raster and clip to extent of grid (defined by flowdir raster)
        with rasterio.open(self.vars['land_use']['path'], 'r') as rs:
            out_img, out_transform = mask(rs, [self.grid_bbox], crop=True, filled=False)
            src_arr = out_img[0]

            # Prepare a dict of lists (empty for the moment) to store arrays to be summed
            # to create final high resolution array to downsample
            nf_cat_arrs = {}
            for nf_cat in nf_cats:
                nf_cat_arrs.update({nf_cat: []})

            # Loop through CLC categories and create NF cat rasters from them
            for key, conv in cat_conv_dict.items():
                # CLC "boolean" (1/0) raster for this category
                src_cat_rs = np.where(src_arr == int(key), float(1), float(0))
                # Get name of NF cat and fraction of this CLC cat contributing to it - tuple of not?
                for nf_cat in conv:
                    if type(nf_cat) is list:
                        nf_cat_name = nf_cat[0]
                        frac_contr_to_nf_cat = float(nf_cat[1])
                    else:
                        nf_cat_name = nf_cat
                        frac_contr_to_nf_cat = 1
                    
                    # Add this CLC cat, multiplied by the fraction of it contributing to
                    # the NF cat, to the list of CLC rasters to be combined into this NF cat
                    nf_cat_arrs[nf_cat_name].append(src_cat_rs * frac_contr_to_nf_cat)
                        
            # Sum all the contributors to each NF cat
            nf_final_arrs = { name:np.sum(nf_cat_arr, axis=0) for name, nf_cat_arr in nf_cat_arrs.items() }

            # Reproject the higher res NF cat to NF model res (defined by grid rs),
            # using the average resampling method to get the fraction cover for each cell.
            # Store output in cats array to fill final raster file (one NF cat per band).
            cats = {}
            for name, old_arr in nf_final_arrs.items():
                if old_arr.shape == src_arr.shape:
                    # Re-read flowdir as reproject fills new_arr
                    new_arr = self.grid.read(1, masked=True)
                    # Reproject. Remember we're not converting CRS here, so clc_rs
                    # CRS can be used as src and dst
                    reproject(
                        source=old_arr,
                        destination=new_arr,
                        src_transform=rs.transform,
                        dst_transform=self.grid.transform,
                        src_crs=rs.crs,    
                        dst_crs=rs.crs,
                        resampling=Resampling.average
                    )
                    new_arr = np.ma.masked_where(self.grid_mask, new_arr)   # Reclip the array
                    cats[name] = new_arr                                    # Store this cat

        # Create the NetCDF variable, set cat names in attribute (in order of land use cat dimension)
        # and fill the variable. Loop through cats manually as fill values not set properly when
        # filling entire variable at once
        nc_var = self.setup_netcdf_var('land_use', [(cat_dim, len(cats))])
        nc_var.cat_names = list(cats.keys())
        for l, (cat_name, cat) in enumerate(cats.items()):
            nc_var[l,:,:] = cat


    def routing(self):
        """Use the flow direction to route the waterbody network."""
        # Create the masked arrays to being with, such that they're masked by the grid mask
        router = Router(self.flow_dir)              # Set up the router to deal with common routing tasks
        outflow_arr = np.ma.zeros((*self.flow_dir.shape, 2), dtype=np.dtype('i2'))
        outflow_arr.mask = self.grid_mask           # Set the grid mask
        inflows_arr = np.ma.array(np.ma.empty((*self.flow_dir.shape, 7, 2), dtype=np.dtype('i2')), mask=True)      # Max of seven inflows
        n_waterbodies = np.ma.array(np.ma.empty(self.flow_dir.shape, dtype=np.dtype('i2')), mask=True)
        is_headwater = np.ma.array(np.ma.empty(self.flow_dir.shape, dtype=np.dtype('i1')), mask=True)
        # waterbody_code = np.ma.array(np.ma.empty(self.flow_dir.shape, dtype=np.dtype(('U', 70))), mask=True)

        # Use the flow direction to set outflow and inflows to each cell
        for index, cell in np.ndenumerate(self.flow_dir):
            y, x = index[0] + 1, index[1] + 1
            if not self.grid_mask[index]:       # Only for non-masked elements
                outflow_arr[index] = router.outflow_from_flow_dir(x, y)
                inflows_arr[index] = router.inflows_from_flow_dir(x, y)
                n_waterbodies[index], is_headwater[index] = router.n_waterbodies_from_inflows(x, y, outflow_arr[index], inflows_arr[index])
                # waterbody_code[index] = router.generate_waterbody_code(x,
                #                                                        y,
                #                                                        outflow_arr[index],
                #                                                        inflows_arr[index],
                #                                                        self.saved_vars['is_estuary'][index],
                #                                                        is_headwater[index])
        
        # Create NetCDF vars for these arrays. Firstly, outflow
        nc_var = self.nc.createVariable('outflow', np.dtype('i2'), ('y', 'x', 'd'))
        nc_var.long_name = 'index of grid cell outflow'
        nc_var.units = ''
        nc_var.grid_mapping = 'crs'
        nc_var[:] = outflow_arr
        # Inflows
        nc_var = self.nc.createVariable('inflows', np.dtype('i2'), ('y', 'x', 'w', 'd'))
        nc_var.long_name = 'indices of grid cell inflows'
        nc_var.units = ''
        nc_var.grid_mapping = 'crs'
        nc_var[:] = inflows_arr
        # Number of waterbodies per cell
        nc_var = self.nc.createVariable('n_waterbodies', np.dtype('i2'), ('y', 'x'))
        nc_var.long_name = 'number of waterbodies in grid cell'
        nc_var.units = ''
        nc_var.grid_mapping = 'crs'
        nc_var[:] = n_waterbodies
        # Is cell a headwater?
        nc_var = self.nc.createVariable('is_headwater', np.dtype('i1'), ('y', 'x'))
        nc_var.long_name = 'is this cell a headwater?'
        nc_var.units = ''
        nc_var.grid_mapping = 'crs'
        nc_var[:] = is_headwater
         

        # # Set the outflows array for each cell
        # for index, cell in np.ndenumerate(self.flow_dir):
        #     y, x = index[0] + 1, index[1] + 1
        #     if not self.flow_dir.mask[index]:       # Only for non-masked elements
        #         outflow_arr[index] = self.outflow_from_flow_dir(x, y, cell)[::-1]

        # Use that to set the inflows array to each cell
        # for j, xy in enumerate(outflow_arr):
        #     for i, outflow in enumerate(xy):
        #         if not self.flow_dir.mask[j,i]:
        #             x_in, y_in = i + 1, j + 1
        #             l, m = outflow[1] - 1, outflow[0] - 1
        #             k = inflows_arr[m,l].count(axis=0)          # How many inflow cells already set
        #             inflows_arr[m,l,k] = [y_in,x_in]

        # Use the number of inflows for each cell to set number of waterbodies
        # n_waterbodies_arr = inflows_arr.count(axis=2)[:,:,0]
        # n_waterbodies_arr = np.ma.masked_array(n_waterbodies_arr, mask=self.flow_dir.mask)
        # max_n_waterbodies_per_cell = n_waterbodies_arr.max()
        # headwaters = np.ma.array(np.full((*self.flow_dir.shape, 7), False), mask=True)      # Max of seven waterbodies per cell
        # stream_order = np.ma.array(np.empty((*self.flow_dir.shape, 7), dtype=int), mask=True)
        # # If number of inflows is zero, it must be a headwater
        # for index, n_waterbodies in np.ndenumerate(n_waterbodies_arr):
        #     if not self.flow_dir.mask[index]:
        #         if n_waterbodies == 0:
        #             n_waterbodies_arr[index] = 1
        #             headwaters[index][0] = True
        #             stream_order[index][0] = 1

        # # Use the number of waterbodies to create an array of inflows to each waterbody
        # waterbody_inflows_arr = np.ma.array(np.ma.empty((*self.flow_dir.shape, max_n_waterbodies_per_cell, 7, 3), dtype=int), mask=True)
        # # Loop through cell inflows to create this array
        # for j, xyw in enumerate(inflows_arr):
        #     for i, cell_inflows in enumerate(xyw):
        #         if not self.flow_dir.mask[j,i]:         # Only if we're in the model domain
        #             if not cell_inflows.mask.all():     # Only if the cell has inflows
        #                 for k, cell_inflow in enumerate(cell_inflows):
        #                     if not cell_inflow.mask.any():
        #                         j_in, i_in = cell_inflow[0] - 1, cell_inflow[1] - 1
        #                         n_waterbodes_inflow = n_waterbodies_arr[j_in, i_in]
        #                         # Each cell inflow is to one waterbody, and this waterbody receives all the inflows from that cell
        #                         for w in range(1, n_waterbodes_inflow + 1):
        #                             waterbody_inflows_arr[j,i,k,w-1] = [*cell_inflow, w]

        # NEXT, set waterbody outflow array



    def outflow_from_flow_dir(self, x, y, flow_dir):
        """Get the outflow cell reference given the current cell
        reference and a flow direction."""
        xy_out = {
            1: [x+1, y],
            2: [x+1, y+1],
            4: [x, y+1],
            8: [x-1, y+1],
            16: [x-1, y],
            32: [x-1, y-1],
            64: [x, y-1],
            128: [x+1, y-1]
        }
        return xy_out[flow_dir]

    def inflows_from_flow_dir(self, x, y):
        """Get the inflow cell references given the current cell
        reference and a flow direction."""
        j, i = y - 1, x - 1

        # Flow direction that means the cell with indices inflows[j_n-j, i_n-i]
        # flows into this cell
        inflow_flow_dir = {
            (-1, -1): 2,
            (0, -1): 1,
            (1, -1): 128,
            (-1, 0): 4,
            (0, 0): 0,
            (1, 0): 64,
            (-1, 1): 8,
            (0, 1): 16,
            (1, 1): 32
        }
        inflow_cells = []
        # Loop through the neighbours and check if they're inflows to this cell
        for j_n in range(j-1,j+2):
            for i_n in range(i-1,i+2):
                if self.in_model_domain(i_n, j_n):
                    if self.flow_dir[j_n, i_n] == inflow_flow_dir[(j_n-j, i_n-i)]:
                        inflow_cells.append([j_n+1, i_n+1])

        # Create masked array from the inflow_cells list
        inflow_cells_ma = np.ma.array(np.ma.empty((7,2), dtype=int), mask=True)
        if len(inflow_cells) > 0:       # Only fill if there are inflows
            inflow_cells_ma[0:len(inflow_cells)] = inflow_cells
        return inflow_cells_ma

    def in_model_domain(self, i, j):
        """Check if index [j,i] is in model domain."""
        i_in_domain = i >= 0 and i < self.grid_mask.shape[1]
        j_in_domain = j >= 0 and j < self.grid_mask.shape[0]
        not_masked = self.flow_dir[j, i] is not np.ma.masked if (i_in_domain and j_in_domain) else False
        return i_in_domain and j_in_domain and not_masked