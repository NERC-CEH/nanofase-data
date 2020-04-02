#!/usr/bin/env python
"""Compiling different data sources into JSON file (for conversion to NetCDF)
for NanoFASE model."""
import rasterio
import yaml
from legacy import datautils as du
import json
from deepmerge import Merger

# Get the config options from the YAML file
with open("config.yaml", 'r') as stream:
    try:
        config = yaml.load(stream, Loader=yaml.BaseLoader)
    except yaml.YAMLError as exc:
        print(exc)

print('Specified config options:')
for opt, val in config.items():
    print('\t{0}: {1}'.format(opt, val))

print('\nLoading flow direction...')
# Load flow direction raster and save the min/max dimensions
rs = rasterio.open(config['flow_dir_raster'])
xmin, xmax = int(rs.bounds.left), int(rs.bounds.right)
ymin, ymax = int(rs.bounds.bottom), int(rs.bounds.top)

# TODO
#   Get (x,y) size of model domain from Rasterio vector
#   Pass this to methods like parse_runoff_data. These then construct (x,y,val) array of data
#   Write something to compile all of these at the end of this script

print('Generating grid structure and linking grid cells/river reaches...')
data_dict = du.generate_grid_from_flow_dir(rs)

# Datasets to merge at the end
datasets = []

print('Parsing runoff data...')
runoff_data_dict = du.parse_runoff_data(config['runoff'], data_dict)
datasets.append(runoff_data_dict)

print('Parsing atmospheric data...')
atmospheric_data_dict = du.parse_atmospheric_data(config['atmospheric_dry_depo_dir'],
                                                  config['atmospheric_wet_depo_dir'],
                                                  data_dict,
                                                  int(config['timesteps']),
                                                  config['material'])
datasets.append(atmospheric_data_dict)

# Source data parsing is horrendously inefficient at the moment!
print('Parsing source data...')
source_data_dict = du.parse_source_data_v3(config['sources'], config['sources_temp'], config['sources_areal'], config['material'], data_dict)
# source_data_dict = du.parse_source_data_v2(config['sources'], config['sources_temp'], config['material'], data_dict)
# data_dict = du.parse_source_data(config['sources'], data_dict)
datasets.append(source_data_dict)

# Soil texture. LUCAS interpolated by (https://esdac.jrc.ec.europa.eu/content/topsoil-physical-properties-europe-based-lucas-topsoil-data)
# and resampled to OSGB
print('Parsing soil texture data...')
soil_texture_data_dict = du.parse_soil_texture_data(config['soil_texture_dir'], data_dict)
datasets.append(soil_texture_data_dict)

# Bulk density [g/cm3 = T/m3]
print('Parsing bulk density data...')
bulk_density_data_dict = du.parse_bulk_density_data(config['bulk_density'], data_dict)
datasets.append(bulk_density_data_dict)

# USLE C-factor. This resampled raster (from ESDAC dataset) isn't that realistic - there are
# too many data gaps for urban areas, which is an artefact of the urban areas being amplified
# upon conversion. In the future, it might be worth proportioning this relative to different
# land use classes in each cell (see the C-factor paper)
print('Parsing C-factor data...')
cfactor_data_dict = du.parse_cfactor_data(config['c-factor'], data_dict)
datasets.append(cfactor_data_dict)

# USLE LS-factor. Resampled raster (from ESDAC dataset) from 100m to 5x5 km.
print('Parsing LS-factor data...')
lsfactor_data_dict = du.parse_lsfactor_data(config['ls-factor'], data_dict)
datasets.append(lsfactor_data_dict)

# USLE P-factor (support practice). Resampled raster (from ESDAC dataset) from 1 km to 5 km.
# Same missing urban data issues as C-factor. Set these to 1 (no support practice) from the moment.
print('Parsing P-factor data...')
pfactor_data_dict = du.parse_pfactor_data(config['p-factor'], data_dict)
datasets.append(pfactor_data_dict)

print('Parsing precipitation data...')
precip_data_dict = du.parse_precipitation_data(config['precipitation_dir'], data_dict, int(config['timesteps']))
datasets.append(precip_data_dict)

# Merge the datasets
print('Merging datasets...')
my_merger = Merger([(list, ["override"]), (dict, ["merge"])], ["override"], ["override"])
for dataset in datasets:
    data_dict = my_merger.merge(data_dict, dataset)

# Converting RiverReaches to EstuaryReaches, where applicable
print('Setting tidal bounds...')
data_dict = du.parse_tidal_bounds(config['tidal_bounds'], data_dict)

# Filling in the gaps
print('Filling the gaps...')
data_dict["dimensions"].update({ "d": 2, "state": 7, "form": 4, "n": 5 })
for grid_cell_ref, grid_cell in data_dict.items():
    if grid_cell_ref not in ["dimensions", "grid_dimensions[d]", "routed_reaches[branches][seeds][river_reach_ref]"]:
        x = grid_cell['x']
        y = grid_cell['y']
        data_dict[grid_cell_ref]["dimensions"] = { "r": 1 }
        data_dict[grid_cell_ref]["type"] = 1
        data_dict[grid_cell_ref]["slope"] = 0.0005
        data_dict[grid_cell_ref]["SoilProfile_{0}_{1}_1".format(x, y)].update({
            "n_soil_layers" : 4,
            "WC_sat" : 0.8,
            "WC_FC" : 0.5,
            "K_s" : 1e-6,
            "usle_K" : 0.032,
            "usle_area_hru" : 0.1,
            "usle_L_ch" : 500,
            "distribution_sediment[s]" : [ 50, 30, 10, 7, 3 ],
            "SoilLayer_1" : {
                "depth" : 0.1
            },
            "SoilLayer_2" : {
                "depth" : 0.1
            },
            "SoilLayer_3" : {
                "depth" : 0.1
            },
            "SoilLayer_4" : {
                "depth" : 0.1
            }
        })
        data_dict[grid_cell_ref]["demands"] = "file::demands-example.json"
        data_dict[grid_cell_ref]["reach_types[r]"] = [1]
        # RiverReach sediment
        if "n_river_reaches" in data_dict[grid_cell_ref]:
            for i in range(1,data_dict[grid_cell_ref]["n_river_reaches"] + 1):
                reach_ref = "RiverReach_{0}_{1}_{2}".format(x,y,i)
                data_dict[grid_cell_ref][reach_ref]["BedSediment"] = "file::bed-sediment-compact.json"
                data_dict[grid_cell_ref][reach_ref]["beta_res"] = 0.000001
                data_dict[grid_cell_ref][reach_ref]["alpha_res"] = 0.001
                data_dict[grid_cell_ref][reach_ref]["slope"] = 0.0005
                if "inflows[in][river_reach_ref]" in data_dict[grid_cell_ref][reach_ref]:
                    data_dict[grid_cell_ref][reach_ref]["dimensions"] = { "in": len(grid_cell[reach_ref]['inflows[in][river_reach_ref]']) }
        # EstuaryReach sediment
        if "n_estuary_reaches" in data_dict[grid_cell_ref]:
            for i in range(1,data_dict[grid_cell_ref]["n_estuary_reaches"] + 1):
                reach_ref = "EstuaryReach_{0}_{1}_{2}".format(x,y,i)
                data_dict[grid_cell_ref][reach_ref]["BedSediment"] = "file::bed-sediment-compact.json"
                data_dict[grid_cell_ref][reach_ref]["beta_res"] = 0.000001
                data_dict[grid_cell_ref][reach_ref]["alpha_res"] = 0.001
                data_dict[grid_cell_ref][reach_ref]["slope"] = 0.0005
                if "inflows[in][river_reach_ref]" in data_dict[grid_cell_ref][reach_ref]:
                    data_dict[grid_cell_ref][reach_ref]["dimensions"] = { "in": len(grid_cell[reach_ref]['inflows[in][river_reach_ref]']) }
        # Remove some stuff
        if 'inflows' in data_dict[grid_cell_ref]: 
            del data_dict[grid_cell_ref]['inflows']

# Put all this in to a file    
with open(config['output_file'], "w") as f:
    json.dump(data_dict, f, indent=4)

print('Compiled data saved to {0}'.format(config['output_file']))