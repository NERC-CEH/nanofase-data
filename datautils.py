#!/usr/bin/env python
import pandas as pd
from datetime import date  
from datetime import timedelta
import rasterio
import numpy as np
import os
import json
from vendor.osgb36_to_wgs84 import OSGB36toWGS84

def outflow_from_flow_dir(x, y, flow_dir):
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


def impose_distribution(val):
    """Impose a size distribution on a total value."""
    return [0.5*val, 0.3*val, 0.1*val, 0.07*val, 0.03*val]


def generate_grid_from_flow_dir(rs):
    """Generate data dict containing the model grid from the flow
    direction raster provided."""
    data_dict = {}
    flow_dir_array = rs.read(1)      # Read the first band of the Thames raster
    # First, just create the grid cells and set their outflow property to
    # contain the flow direction
    for y, xy in enumerate(flow_dir_array):
        for x, flow_dir in enumerate(xy):
            # Get the spatial coord at the centre and lower left of this pixel
            # Rastio .xy(row,col) takes row, col (y,x)
            xy_spatial = rs.xy(y,x)
            xy_spatial_ll = rs.xy(y,x,offset='ll')
            if flow_dir > 0:
                cell_ref = "GridCell_{0}_{1}".format(x+1,y+1)
                data_dict[cell_ref] = {
                    "x": x+1,
                    "y": y+1,
                    "x_coord_c": xy_spatial[0],
                    "y_coord_c": xy_spatial[1],
                    "x_coord_ll": xy_spatial_ll[0],
                    "y_coord_ll": xy_spatial_ll[1],
                    "size[d]": list(rs.res),
                    "outflow[d]": outflow_from_flow_dir(x+1, y+1, flow_dir),
                    "flow_dir": int(flow_dir)
                }
                
    # Set the the inflow for each cell, based on the cell outflows. At the
    # end of this, each cell will have the number of outflow equal to the
    # number of river reaches that should be in that cell
    for ref, cell in data_dict.items():
        x = cell['x']
        y = cell['y']
        outflow_ref = "GridCell_{0}_{1}".format(*cell['outflow[d]'])
        # Set this cell's outflow as the receiving cell's inflow.
        if outflow_ref in data_dict:
            if 'inflows' in data_dict[outflow_ref]:
                data_dict[outflow_ref]['inflows'].append([x,y])
            else:
                data_dict[outflow_ref]['inflows'] = [[x,y]]
        else:
            data_dict[ref]['domain_outflow[d]'] = cell['outflow[d]']
            
    # Now create the number of river reaches, based on the number of inflows.
    # If there are no inflows, there must just be one river reach
    for ref, cell in data_dict.items():
        x = cell['x']
        y = cell['y']
        if 'inflows' in cell:
            for i in range(1, len(cell['inflows']) + 1):
                data_dict[ref]['RiverReach_{0}_{1}_{2}'.format(x,y,i)] = {}
            data_dict[ref]['n_river_reaches'] = i
        else:
            data_dict[ref]['RiverReach_{0}_{1}_1'.format(x,y)] = {}
            data_dict[ref]['n_river_reaches'] = 1

    # Now that we've got the correct number of river reaches, we need to
    # set their inflows. We couldn't do this until they were all created
    # as we wouldn't know how many inflows the grid cell inflow reach would
    # have from the upstream cell. Remember each cell can only have one
    # outflow, so any river reach inflow will come from *all* reaches in
    # the upstream grid cell
    for ref, cell in data_dict.items():
        x = cell['x']
        y = cell['y']
        data_dict[ref]['SoilProfile_{0}_{1}_1'.format(x, y)] = {}       # Empty soil profile group
        # Loop through the number of reaches, which will be the same number
        # as the number of cell inflows
        for i in range(1,cell['n_river_reaches'] + 1):
            reach_ref = "RiverReach_{0}_{1}_{2}".format(x,y,i)
            # Only carry on if this cell has inflows
            if 'inflows' in cell:
                x_in, y_in = cell['inflows'][i-1]
                data_dict[ref][reach_ref]['inflows[in][river_reach_ref]'] = []
                # *All* of the upstream cell's reaches must drain into this river
                for j in range(1,data_dict["GridCell_{0}_{1}".format(x_in, y_in)]['n_river_reaches'] + 1):
                    data_dict[ref][reach_ref]['inflows[in][river_reach_ref]'].append([x_in, y_in, j])
            # If there's a domain outflow to the grid cell, this setup means that every reach
            # will also be a domain outflow, so we must set them as so, using the grid cell's
            # domain outflow property
            if 'domain_outflow[d]' in cell:
                data_dict[ref][reach_ref]['domain_outflow[d]'] = cell['domain_outflow[d]']

    grid_dimensions = flow_dir_array.shape    # Returns (y,x), not (x,y)!
    data_dict["grid_dimensions[d]"] = [grid_dimensions[1], grid_dimensions[0]]

    return data_dict


def parse_runoff_data(runoff_data_path, data_dict):
    """Parse the runoff data given the path to the runoff data."""
    runoff_file_last_modified = os.path.getmtime(runoff_data_path)
    cache_file = 'cache/{0}_runoff.json'.format(runoff_file_last_modified)

    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as cache:
            data_dict = json.load(cache)
        print("\t...retrieving from cache ({0})".format(cache_file))
    else:
        df = pd.read_csv(runoff_data_path, header=0)
        df['date'] = pd.to_datetime({
            'year': df['yr'],
            'month': df['month'],
            'day': df['day']
        })

        for ref, cell in data_dict.items():
            if ref != 'grid_dimensions[d]':
                print("\t...for cell {0}".format(ref))
                df_xy = df[(df['easts'] == int(cell['x_coord_c'])) & (df['norths'] == int(cell['y_coord_c']))]
                data_dict[ref]['runoff[t]'] = []
                data_dict[ref]['quickflow[t]'] = []
                for t in range(0,365):
                    current_date = date(2015,1,1) + timedelta(days=t)
                    df_xy_t = df_xy[df_xy['date'] == current_date.strftime('%Y-%m-%d')]
                    if len(df_xy_t) > 0:
                        # If negative (!), set to 0
                        qf = df_xy_t['QF'].values[0] if df_xy_t['QF'].values[0] > 0 else 0
                        sf = df_xy_t['SF'].values[0] if df_xy_t['SF'].values[0] > 0 else 0
                        # Convert from mm/day to m/s
                        data_dict[ref]['runoff[t]'].append((qf + sf) / (1000 * 86400))
                        data_dict[ref]['quickflow[t]'].append(qf / (1000 * 86400))
                    else:
                        data_dict[ref]['runoff[t]'].append(0)
                        data_dict[ref]['quickflow[t]'].append(0)

        with open(cache_file, 'w') as cache:
            json.dump(data_dict, cache)

    return data_dict


def parse_atmospheric_data(dry_depo_dir, wet_depo_dir, data_dict, timesteps):
    # Only check the first of the drydepo files to see if it's been updated
    dry_depo_last_modified = os.path.getmtime(dry_depo_dir + "LE-Sofia-UK_drydepo_dayval_1.tif")
    cache_file = 'cache/{0}_atmospheric.json'.format(dry_depo_last_modified)

    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as cache:
            data_dict = json.load(cache)
        print("\t...retrieving from cache ({0})".format(cache_file))
    else:
        for t in range(1,timesteps+1):
            rs_dry = rasterio.open(dry_depo_dir + "LE-Sofia-UK_drydepo_dayval_" + str(t) + ".tif")
            rs_wet = rasterio.open(wet_depo_dir + "LE-Sofia-UK_wetdepo_dayval_" + str(t) + ".tif")
            # Loop through the dict and append this timestep's rainfall to the correct cell
            for grid_cell_ref, grid_cell in data_dict.items():
                if grid_cell_ref != 'grid_dimensions[d]':
                    # Create the empty timeseries if this is the first timestep
                    if t == 1:
                        data_dict[grid_cell_ref]['DiffuseSource_1'] = {}
                        data_dict[grid_cell_ref]['DiffuseSource_2'] = {}
                        data_dict[grid_cell_ref]['DiffuseSource_1']['input_mass_atmospheric[n][t]'] = np.zeros((5, 365)).tolist()
                        data_dict[grid_cell_ref]['DiffuseSource_2']['input_mass_atmospheric[n][t]'] = np.zeros((5, 365)).tolist()
                        data_dict[grid_cell_ref]['n_diffuse_sources'] = 2
                    x_grid = grid_cell['x_coord_c']
                    y_grid = grid_cell['y_coord_c']
                    dry_row, dry_col = rs_dry.index(x_grid, y_grid)
                    wet_row, wet_col = rs_wet.index(x_grid, y_grid)
                    dry_arr = rs_dry.read(1)
                    wet_arr = rs_wet.read(1)
                    dry_dep = impose_distribution(dry_arr[dry_row, dry_col].item())
                    wet_dep = impose_distribution(wet_arr[wet_row, wet_col].item())
                    # Add deposited mass to free, core
                    # TODO check units
                    for i in range(0,5):
                        data_dict[grid_cell_ref]['DiffuseSource_1']['input_mass_atmospheric[n][t]'][i][t-1] = dry_dep[i]
                        data_dict[grid_cell_ref]['DiffuseSource_2']['input_mass_atmospheric[n][t]'][i][t-1] = wet_dep[i]

        with open(cache_file, 'w') as cache:
            json.dump(data_dict, cache)

    return data_dict


def parse_source_data(sources, data_dict):

    # Only check the first of the drydepo files to see if it's been updated
    sources_last_modified = os.path.getmtime(sources)
    cache_file = 'cache/{0}_sources.json'.format(sources_last_modified)

    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as cache:
            data_dict = json.load(cache)
        print("\t...retrieving from cache ({0})".format(cache_file))
    else:
        df = pd.read_csv(sources, header=0, sep=';')
        # For the moment, just point sources to surface waters of NP (not matrix or product embedded)
        df = df.loc[(df['ISO3'] == 'GBR') & (df['Form'] == 'Nanoparticulate') & (df['Compartment'] == 'Surface water')]
        df.drop(['ISO3', 'Nanomaterial', 'Compartment', 'Form', 'SourceType'], axis=1)
        print('\t...there are {0} sources...'.format(df.shape[0]))
        s = 0
        for i, source in df.iterrows():
            s = s+1
            print('\t\t...source {0} ({1})'.format(s, source['SourceType']))
            for grid_cell_ref, grid_cell in data_dict.items():
                if grid_cell_ref != 'grid_dimensions[d]':
                    x = grid_cell['x']
                    y = grid_cell['y']
                    if grid_cell_ref != 'grid_dimensions[d]':
                        x_grid_ll = grid_cell['x_coord_ll']
                        y_grid_ll = grid_cell['y_coord_ll']
                        cell_lat_ll, cell_lon_ll = OSGB36toWGS84(x_grid_ll, y_grid_ll)                  # OSGB lower left coords to lat/lon
                        cell_lat_ur, cell_lon_ur = OSGB36toWGS84(x_grid_ll + 5000, y_grid_ll + 5000)    # OSGB upper right coords to lat/lon
                        source_lat, source_lon = source['Latitude'], source['Longitude']

                        if source_lat > cell_lat_ll and source_lon > cell_lon_ll:
                            if source_lat < cell_lat_ur and source_lon < cell_lon_ur:
                                # If it's a point source
                                if source['SourceType'] == 'P':
                                    rr = 'RiverReach_{0}_{1}_1'.format(x, y) 
                                    if rr in data_dict[grid_cell_ref]:
                                        if 'n_point_sources' in data_dict[grid_cell_ref][rr]:
                                            n_point_sources = data_dict[grid_cell_ref][rr]['n_point_sources'] + 1
                                        else:
                                            n_point_sources = 1
                                        # Units of point source emissions are ton/source/year, convert to kg/source/day
                                        np_in = impose_distribution(source['SumOfLocal_emission_ton']*1e3/365)
                                        data_dict[grid_cell_ref][rr]['n_point_sources'] = n_point_sources
                                        ps = 'PointSource_{0}'.format(n_point_sources)
                                        data_dict[grid_cell_ref][rr][ps] = {}
                                        data_dict[grid_cell_ref][rr][ps]['fixed_mass[state][form][n]'] = np.zeros((7, 4, 5)).tolist()
                                        data_dict[grid_cell_ref][rr][ps]['fixed_mass[state][form][n]'][0][0][:] = np_in
                                        # data_dict[grid_cell_ref][rr][ps]['fixed_mass_frequency'] = 20
                                # If it's a diffuse source
                                elif source['SourceType'] == 'A':
                                    # There will already be 2 diffuse sources from atmospheric, so set these as DiffuseSource_3+
                                    if 'n_diffuse_sources' in data_dict[grid_cell_ref]:
                                        n_diffuse_sources = data_dict[grid_cell_ref]['n_diffuse_sources'] + 3
                                    else:
                                        n_diffuse_sources = 3
                                    data_dict[grid_cell_ref]['n_diffuse_sources'] = n_diffuse_sources
                                    # Units of areal source emissions are ton/source/year, convert to kg/m2/s
                                    # HACK this is approximately /m2, based on 7x7km grid cells, which they won't all be
                                    np_in = impose_distribution(source['SumOfLocal_emission_ton']*1e3/(86400*365*7000*7000))
                                    ds = 'DiffuseSource_{0}'.format(n_diffuse_sources)
                                    data_dict[grid_cell_ref][ds] = {}
                                    # Use atmospheric input field for the moment as no form/state info provided
                                    data_dict[grid_cell_ref][ds]['input_mass_atmospheric[n][t]'] = [[np_in_n] * 365 for np_in_n in np_in]

        with open(cache_file, 'w') as cache:
            json.dump(data_dict, cache)

    return data_dict


def parse_soil_texture_data(soil_texture_dir, data_dict):
    # Loop through the cells and fill texture properties using the raster
    for texture in ['sand', 'silt', 'clay', 'coarse-frag']:
        # Open the appropriate texture content raster
        texture_rs = rasterio.open(soil_texture_dir + "{0}-content_osgb_5000m.tif".format(texture))
        texture = texture.replace('-', '_')   # Change coarse-frag to coarse_frag
        for grid_cell_ref, grid_cell in data_dict.items():
            if grid_cell_ref != 'grid_dimensions[d]':
                x_grid = grid_cell['x_coord_c']
                y_grid = grid_cell['y_coord_c']
                row, col = texture_rs.index(x_grid, y_grid)
                arr = texture_rs.read(1)
                if arr[row, col].item() > 0:
                    data_dict[grid_cell_ref]['SoilProfile_{0}_{1}_1'.format(grid_cell['x'], grid_cell['y'])]['{0}_content'.format(texture)] = int(round(arr[row, col].item()))
                else:
                    data_dict[grid_cell_ref]['SoilProfile_{0}_{1}_1'.format(grid_cell['x'], grid_cell['y'])]['{0}_content'.format(texture)] = 0 

    return data_dict


def parse_bulk_density_data(bulk_density, data_dict):
    bd_rs = rasterio.open(bulk_density)
    for grid_cell_ref, grid_cell in data_dict.items():
        if grid_cell_ref != 'grid_dimensions[d]':
            x_grid = grid_cell['x_coord_c']
            y_grid = grid_cell['y_coord_c']
            row, col = bd_rs.index(x_grid, y_grid)
            arr = bd_rs.read(1)
            data_dict[grid_cell_ref]['SoilProfile_{0}_{1}_1'.format(grid_cell['x'], grid_cell['y'])]['bulk_density'] = arr[row, col].item()

    return data_dict


def parse_cfactor_data(cfactor, data_dict):
    c_rs = rasterio.open(cfactor)
    for grid_cell_ref, grid_cell in data_dict.items():
        if grid_cell_ref != 'grid_dimensions[d]':
            x_grid = grid_cell['x_coord_c']
            y_grid = grid_cell['y_coord_c']
            row, col = c_rs.index(x_grid, y_grid)
            arr = c_rs.read(1)
            if arr[row, col].item() > 0:
                data_dict[grid_cell_ref]['SoilProfile_{0}_{1}_1'.format(grid_cell['x'], grid_cell['y'])]['usle_C[t]'] = arr[row, col].item()
            else:
                # If there's no data, pick a really low value to represent this cell is mainly urban
                data_dict[grid_cell_ref]['SoilProfile_{0}_{1}_1'.format(grid_cell['x'], grid_cell['y'])]['usle_C[t]'] = 0.00055095

    return data_dict


def parse_lsfactor_data(lsfactor, data_dict):
    ls_rs = rasterio.open(lsfactor)
    for grid_cell_ref, grid_cell in data_dict.items():
        if grid_cell_ref != 'grid_dimensions[d]':
            x_grid = grid_cell['x_coord_c']
            y_grid = grid_cell['y_coord_c']
            row, col = ls_rs.index(x_grid, y_grid)
            arr = ls_rs.read(1)
            if arr[row, col].item() > 0:
                data_dict[grid_cell_ref]['SoilProfile_{0}_{1}_1'.format(grid_cell['x'], grid_cell['y'])]['usle_LS'] = arr[row, col].item()
            else:
                # Fudge LS factor for now
                data_dict[grid_cell_ref]['SoilProfile_{0}_{1}_1'.format(grid_cell['x'], grid_cell['y'])]['usle_LS'] = 0.3

    return data_dict


def parse_pfactor_data(pfactor, data_dict):
    p_rs = rasterio.open(pfactor)
    for grid_cell_ref, grid_cell in data_dict.items():
        if grid_cell_ref != 'grid_dimensions[d]':
            x_grid = grid_cell['x_coord_c']
            y_grid = grid_cell['y_coord_c']
            row, col = p_rs.index(x_grid, y_grid)
            arr = p_rs.read(1)
            if arr[row, col].item() > 0:
                data_dict[grid_cell_ref]['SoilProfile_{0}_{1}_1'.format(grid_cell['x'], grid_cell['y'])]['usle_P'] = arr[row, col].item()
            else:
                # If there's no data, go for no support practices (1)
                data_dict[grid_cell_ref]['SoilProfile_{0}_{1}_1'.format(grid_cell['x'], grid_cell['y'])]['usle_P'] = 1

    return data_dict


def parse_precipitation_data(precipitation_dir, data_dict, timesteps):

    # Only check the first of the precip files to see if it's been updated
    precip_last_modified = os.path.getmtime(precipitation_dir + "rainfall_5km_2015-1.tif")
    cache_file = 'cache/{0}_precip.json'.format(precip_last_modified)

    if os.path.isfile(cache_file):
        with open(cache_file, 'r') as cache:
            data_dict = json.load(cache)
        print("\t...retrieving from cache ({0})".format(cache_file))
    else:
        for t in range(1,timesteps+1):
            # Open this timestep's file
            rs = rasterio.open(precipitation_dir + "rainfall_5km_2015-{0}.tif".format(t))
            # Loop through the dict and append this timestep's rainfall to the correct cell
            for grid_cell_ref, grid_cell in data_dict.items():
                if grid_cell_ref != 'grid_dimensions[d]':
                    # Create the empty timeseries if this is the first timestep
                    if t == 1:
                        data_dict[grid_cell_ref]['precip[t]'] = []
                    x_grid = grid_cell['x_coord_c']
                    y_grid = grid_cell['y_coord_c']
                    row, col = rs.index(x_grid, y_grid)
                    arr = rs.read(1)
                    data_dict[grid_cell_ref]['precip[t]'].append(arr[row, col].item() / (1000 * 86400))  # Converted to m/s from mm/day

        with open(cache_file, 'w') as cache:
            json.dump(data_dict, cache)

    return data_dict


def parse_tidal_bounds(tidal_bounds, data_dict):
    # Open the raster file that defines the extent of the estuary. Any grid cell that has a
    # value in this raster will be defined as having an EstuaryReach
    rs = rasterio.open(tidal_bounds)
    # Loop through the grid cells and check if they should be estuarine
    for grid_cell_ref, grid_cell in data_dict.items():
        if grid_cell_ref != 'grid_dimensions[d]':
            x_grid = grid_cell['x_coord_c']
            y_grid = grid_cell['y_coord_c']
            row, col = rs.index(x_grid, y_grid)
            arr = rs.read(1)
            # If this is an estuary
            if arr[row, col].item() > 0:
                temp_grid_cell_dict = data_dict[grid_cell_ref]
                for k, v in temp_grid_cell_dict.items():
                    if 'RiverReach' in k:
                        # Create new dict for the estuary, and fill with the existing river dict's values
                        temp_grid_cell_dict[k.replace('River', 'Estuary')] = v
                        # Remove the river dict
                        del temp_grid_cell_dict[k]
                # Change n_river_reaches to n_estuary_reaches
                temp_grid_cell_dict['n_estuary_reaches'] = temp_grid_cell_dict.pop('n_river_reaches')
                # Apply these changes to the master dict
                data_dict[grid_cell_ref] = temp_grid_cell_dict

    return data_dict

