#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

debug=0
output_path="apptainer/intelliperf.sif"

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            output_path="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [-o|--output OUTPUT_PATH]"
            echo "  -o, --output    Specify output path for the .sif file (default: apptainer/intelliperf.sif)"
            exit 1
            ;;
    esac
done

definition="apptainer/intelliperf.def"

echo "Building Apptainer image..."
echo "Definition file: $definition"
echo "Output path: $output_path"

apptainer build $output_path $definition