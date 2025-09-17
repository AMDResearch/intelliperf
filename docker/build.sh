#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

# Parse command line arguments
dev_mode=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev|-d)
            dev_mode=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Container name
name="intelliperf"

# Script directories
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"
cur_dir=$(pwd)
pushd "$script_dir"

docker build \
    -t "$name" \
    -f "$script_dir/intelliperf.Dockerfile" \
    .

popd