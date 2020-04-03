#!/usr/bin/env python
"""NanoFASE data is responsible for compiling and editing data for use in NanoFASE model."""
import sys
import os
import argparse
from compiler import Compiler

parser = argparse.ArgumentParser(description='Compile or edit data for the NanoFASE model.')
parser.add_argument('task', help='do you wish to create from scratch or edit the data?', choices=['create', 'edit'])
parser.add_argument('config_file', help='path to the config file')
args = parser.parse_args()

# Create the compiler. We presume the model_vars.yaml file is in the same directory as this script
compiler = Compiler(args.task, args.config_file, os.path.join(sys.path[0], 'model_vars.yaml'))

# Do we want to compile from scratch or edit a pre-exiting file?
if args.task == 'create':
    # Run the data compilation
    compiler.create()
elif args.task == 'edit':
    # Edit the variables given in the config file
    compiler.edit()

