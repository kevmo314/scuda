#!/bin/bash

set -e

cd codegen

# Run the Python script in the codegen directory
python3 ./codegen.py

# Move up to the parent directory
cd ..

# Run CMake to configure and build the project
cmake .
cmake --build .