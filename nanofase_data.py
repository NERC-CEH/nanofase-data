#!/usr/bin/env python
"""NanoFASE data is responsible for compiling and editing data for use in NanoFASE model."""
import sys
import os
import argparse
from compiler import Compiler

parser = argparse.ArgumentParser(description='Compile or edit data for the NanoFASE model.')
parser.add_argument('task',
                    help='do you wish to create from scratch, edit the data or create a constants file?',
                    choices=['create', 'edit', 'constants'])
parser.add_argument('file', help='path to the config file (create/edit tasks) or constants file (constants task)')
parser.add_argument('--output', '-o', help='where to create the new constants file (for constants task)')
args = parser.parse_args()

if args.task == 'constants':
    # If we're just compiling the constants file, just call the static create_constants method
    Compiler.create_constants(args.file, args.output)
else:
    # Create the compiler. We presume the model_vars.yaml file is in the same directory as this script
    compiler = Compiler(args.task, args.file, os.path.join(sys.path[0], 'model_vars.yaml'))
    # Do we want to compile from scratch or edit a pre-exiting file?
    if args.task == 'create':
        # Run the data compilation
        compiler.create()
    elif args.task == 'edit':
        # Edit the variables given in the config file
        compiler.edit()
