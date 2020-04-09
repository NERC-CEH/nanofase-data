# Config files

A config file must be provided when running the `nanofase_data.py` script. Examples for creation and editing are given:
- [`config.create.example.yaml`](../config.create.example.yaml)
- [`config.edit.example.yaml`](../config.edit.example.yaml)

The examples are annotated and should be self-explanatory. However, there are a few areas that need further documentation:

## General layout

The `create` and `edit` config files follow a similar layout. A variety of setup data is required and in the examples is placed at the top of the file. This includes file paths to input and output path, and model config info (e.g. timestep info). This is then followed by a list of variables, each of which must have at least a `path` and `type` property. If `units` are included, the module will convert them to the correct units on compilation. `source` and `references` can be used to add these attributes to the NetCDF file, but are not used by the model.

```yaml
soil_bulk_density:
  type: raster
  units: t/m**3
  path: <root_dir>soil_bulk_density.tif
```

See the [`config.create.example.yaml`](../config.create.example.yaml) for more examples and the full list of variables required for the NanoFASE model.

## Constants file

The NanoFASE data module generates two files, a NetCDF dataset and a Fortran namelist constants file. The NetCDF dataset holds spatial and/or temporal data, encompassing *most* of the data required by the NanoFASE model. The constants file holds data for variables which are constant in space and time. The main reason for including this as a separate text-based file is to provide an easier way to edit constant variables, using a text editor rather than having to write a script to do so.

The data module simply converts the YAML constants file provided into a Fortran namelist file. The location of this YAML file should be given in the config file:

```YAML
constants_file: /path/to/constants.yaml
```

Note this conversion only happens in *creation* mode and there is no utility to edit the Fortran namelist file via the data module. Instead, if you wish to edit the file, you can just use a text editor to do so.

## `<root_dir>`

The `root_dir` variable can be used to specify a directory which can be used in the `path` property of each variable, for example to point to a directory in which all the data are stored. If `<root_dir>` is included in a `path` property, the value of `root_dir` will be substituted. For example:

```yaml
...
root_dir: /path/to/data
...
flow_dir:
  type: raster
  path: <root_dir>flow_dir.tif      # Evaluates to /path/to/data/flow_dir.tif
runoff: 
  type: csv
  path: <root_dir>runoff.csv        # Evaluates to /path/to/data/runoff.csv
```  

## Land use

The module maps between common land use classes (e.g. those provided by [CORINE](https://land.copernicus.eu/pan-european/corine-land-cover)) and the simpler, grouped land use classes used within the NanoFASE model by way of a land use config file. If `land_use_file` is not provided in the config file, `land_use.default.yaml` is used instead - we recommend you use the CORINE land cover map, resampled to the correct CRS (e.g. ESPG:27700, British National Grid, in case of the Thames scenario) and stick with this default.

### Point source emissions and temporal profiles

Unlike areal source emissions, which are (currently) constant throughout the model run\*, point sources can have a temporal profile applied and this makes their input a little more complicated than most variables (though I am working on making it simpler than it currently is).

Point source emissions are provided by a shapefile. Each point within the shapefile should have a number of variables:
  - Source type: A string to categorise this source. This is used to apply different temporal profiles to different sources (currently a maximum of one temporal profile is supported). Named `profile` in [example data](../data.example/thames_tio2_2015).
  - Value variable: The value for this point source. Named `emission` in [example data](../data.example/thames_tio2_2015).

The names of these variables (columns) are specified in the model config:

```yaml
emissions_point_water_pristine:
  type: shapefile
  value_var: emission                 # The name of the value variable in the shapefile
  path: ...
  source_type_col: profile            # The name of the source type variable in the shapefile
```

**Temporal profiles** for a shapefile can be specified by the `temporal_profile` property. This should point to a CSV file (example given [here](../data.example/thames_tio2_2015/emissions_temporal-profile_2015.csv)), with `ISO3`, source type and factor columns. The name of the source type and factor columns can be specified in the config file:

```yaml
emissions_point_water_pristine:
  ...
  temporal_profile:
    path: /path/to/temporal_profile.csv
    source_type_col: Emission_source_type       # The name of the column giving the source type
    for_source_type: P2                         # The value of source_type_col in the shapefile for which this temporal profile should apply
    factor_col: Factor                          # Which column gives the temporal factor?
``` 

The source type column is cross-referenced with the `source_type_col` column for the shapefile and only those points with matching source types have this temporal profile applied to them. In the example data, profiles with source type `P2` are given this temporal profile.

Note that for the moment, only daily temporal factors are allowed and temporal profiles are for each year, and thus when the temporal profile CSV file is filtered by ISO3 and source type, it should contain 365-366 rows (depending on whether it is a leap year or not).

\* *This is kind of true. Individual model runs have constant emissions for the whole run. However, individual model runs can be chained together in a multi-year model run, each year having different areal emissions. This is how multi-year simulations are currently performed.* 