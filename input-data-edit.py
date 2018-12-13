from netCDF4 import Dataset
import yaml
import sys

# Get the config options from the YAML file
with open("config.yaml", 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

nc = Dataset(config['dataset'], 'w')
var = sys.argv[1]
val = sys.argv[2]
# Get the variable asked for
var = nc[var]
var[:] = val
