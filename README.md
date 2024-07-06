# NanoFASE Data

The NanoFASE Data module is a set of scripts to compile and edit input data for the [NanoFASE model](https://github.com/nerc-ceh/nanofase).

[See the NanoFASE documentation for full documentation](https://nerc-ceh.github.io/nanofase/users/nanofase-data.html).

## Getting started

The easiest way to use the library is to [install it from PyPI](https://pypi.org/project/nfdata/). For example, using pip:

```bash
$ pip install nfdata
```

## Usage

```
usage: nfdata [-h] [--output OUTPUT] {create,edit,constants} file

Compile or edit data for the NanoFASE model.

positional arguments:
  {create,edit,constants}
                        do you wish to create from scratch, edit the data or
                        create a constants file?
  file                  path to the config file (create/edit tasks) or
                        constants file (constants task)

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT, -o OUTPUT
                        where to create the new constants file (for constants
                        task)
```

### Creating a new dataset

Specifying the "create" option compiles a new NetCDF dataset and Fortran namelist constant file:

```shell script
$ nfdata create /path/to/config.create.yaml
```

An annotated example config file is given: [`config.create.example.yaml`](config.create.example.yaml). The file is quite self-explanatory, but a few further explanations are provided in [this document](docs/config.md).

The two files will be output to the paths specified in the config file.

### Editing an existing dataset

To edit an existing NetCDF dataset, specify the "edit" option:

```shell script
$ nfdata edit /path/to/config.edit.yaml
```

An annotated example config file is given: [`config.edit.example.yaml`](config.edit.example.yaml). This is similar (but not identical) in format to the creation config file, except only those variables you with to edit should be specified (all other variables are left as-is).

Certain variables can't be edited: `flow_dir`, `is_estuary`. Create a new dataset instead if you wish to change these variables.

The Fortran namelist file cannot be edited using this method and you should instead edit the file directly.

### Only creating a new constants file

To simply convert a constants YAML file to a Fortran namelist file, you can use the `constants` option:

```shell script
$ nfdata constants /path/to/constants.yaml -o /path/to/constants.nml
```

No config file is required. The location of the newly created constants file is given by the `-o` or `--output` argument.

### Tips
- For the moment, all rasters must be the same CRS as the `flow_dir` raster, and this must be a projected raster. We recommend ESPG:27700 (British National Grid). In addition, all rasters except for `land_use` must be the same resolution as `flow_dir`. They can cover a large geographical region and the module will automatically clip them to the correct size.
- Support for different file types is a bit sporadic at the moment. I suggest sticking the raster files for spatial variables, raster or CSV files for spatiotemporal variables (with 1 file per timestep for raster files) and shapefiles for point sources. You will trigger errors if you use an unsupported file.
- Example input data files are given in `data.example/thames_tio2_2015/`. Running the model using the example config files uses these data. 