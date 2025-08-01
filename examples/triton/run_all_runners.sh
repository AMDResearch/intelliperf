#!/bin/bash

# This script runs all Python files in the autogen_10 and autogen_science_10 directories
# that end with '_runner.py', passing the --validate flag to each script.

echo "Running runners in autogen_10..."
for f in autogen_10/*_runner.py; do
    echo "Running $f with validation..."
    python3 "$f" --validate
    echo "------------------------"
done

echo "Running runners in autogen_science_10..."
for f in autogen_science_10/*_runner.py; do
    echo "Running $f with validation..."
    python3 "$f" --validate
    echo "------------------------"
done 