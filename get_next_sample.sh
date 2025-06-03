#!/bin/bash

# Check number of arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: ./run_next_sample.sh <walls_path.csv> <data_path.csv> <step_size>"
    exit 1
fi

# Assign arguments to variables
WALLS_PATH=$1
DATA_PATH=$2
STEP=$3

# Run the Julia script with arguments
julia run_next_sample.jl "$WALLS_PATH" "$DATA_PATH" "$STEP"