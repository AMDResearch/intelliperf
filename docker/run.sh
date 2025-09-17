#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

name="intelliperf"

docker run -it --rm \
    --name "$name" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -e LLM_GATEWAY_KEY="$LLM_GATEWAY_KEY" \
    "$name"