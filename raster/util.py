from netCDF4 import Dataset

def setup_netcdf_dataset(config, grid, crs):
    """Create NetCDF file, add required dimensions and coordinate variables"""
    nc = Dataset(config['output_file'], 'w', format='NETCDF4')
    nc.description = "Input data for NanoFASE model"
    nc.Conventions = 'CF-1.0'
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
    # Return the NetCDF file
    return nc

def var_lookup():
    return {
        'runoff': {
            'standard_name': 'runoff_flux',
            'units': 'mm day-1'
        },
        'soil_bulk_density': {
            'standard_name': 'soil_bulk_density',
            'units': 'T m-3'
        }
    }