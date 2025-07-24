#!/bin/bash

# Directory to search; default is current directory
DIR=${1:-.}

# Find all .py files in the directory and run each with python3
for file in "$DIR"/*.py; do
  if [ -f "$file" ]; then
    echo "Running $file"
    python3 "$file"
    echo "----------------------"
  fi
done
