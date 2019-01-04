# NanoFASE Data Tools

Collection of scripts for parsing data related to the [NanoFASE model](https://github.com/nerc-ceh/nanofase).

## `input-data-compilation.py`
Compile input data into one big JSON file, for subsequent conversion to NetCDF.

Config options:
- `flow_dir_raster`: Path to the flow direction raster for the desired catchment. Must be in ESPG:27700 projection (OSGB). The bounds of this raster will be used to clip all of the other data. File must be a valid raster file, openable by [Rasterio](https://github.com/mapbox/rasterio).
- `runoff`: CSV file of runoff with column headers `yr,month,day,easts,norths,QF,SF`. Easts and norths are OSGB grid references for the *centre* of the grid cell and QF and SF are quickflow and slowflow, as output from HMF run with CERF.
- `atmospheric_dry_depo_dir`: Directory to the atmospheric dry deposition data. Data should be one file per time step and have the file name `LE-Sofia-UK_drydepo_dayval_{t}.tif`, where `{t}` is the time step number. Must be in ESPG:27700 projection (OSGB).
- `atmospheric_wet_depo_dir`: Directory to the atmospheric wet deposition data. Data should be one file per time step and have the file name `LE-Sofia-UK_wetdepo_dayval_{t}.tif`, where `{t}` is the time step number. Must be in ESPG:27700 projection (OSGB).
- `sources`: Path to the sources data .txt file with column headers `ISO3;Nanomaterial;Compartment;Form;SourceType;Longitude;Latitude;SumOfLocal_emission_ton`. Values must be semicolon separated. Longitude and latitude are in degrees.
- `soil_texture_dir`: Directory to the sand, silt, clay and coarse-frag raster files. File names must follow format `{0}-content_osgb_5000m.tif`, where `{0}` is iether sand, silt, clay or coarse-frag. Must be in ESPG:27700 projection (OSGB).
- `bulk_density`: Path to bulk density raster. Must be in ESPG:27700 projection (OSGB) and be a valid raster file, openable by Rasterio.
- `c-factor`, `ls-factor` and `p-factor`: Path to the RUSLE C-factor, LS-factor and P-factor rasters. Must be in ESPG:27700 projection (OSGB) and be a valid raster file, openable by Rasterio.
- `precipitation_dir`: Directory to the precipitation data. Data should be one file per time step and have the file name `Lrainfall_5km_2015-{t}.tif`, where `{t}` is the time step number. Must be in ESPG:27700 projection (OSGB).
- `tidal_bounds`: Path to a raster file that defines the bounds of the estuary. Every raster pixel (grid cell) that has a value will be assumed to have only estuarine waterbodies.
- `output_file`: Where should the final JSON file be placed?

### Cache
`input-data-compilation.py` has a primitive caching system to speed up data compilation for the runoff, atmospheric, sources and precipitation datasets. Cached JSON files are stored in `cache/`. The cache works by comparing the timestamp of the original data file with a timestamp encoded in the cached file's name; if they are equal, the cached file is used, if not, the original data is used and a new cache file is created (the old cached file currently isn't removed, this must be done manually). **Note:** For data sources that have multiple input files (e.g. atmospheric data with one file per timestep), only the first file's timestamp is checked.
