#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

debug=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            debug=1
            shift
            ;;
        *)
            echo "Usage: $0 [-d|--debug]"
            exit 1
            ;;
    esac
done


definition="apptainer/intelliperf.def"

image="apptainer/intelliperf.sif"
apptainer build \
     $image $definition