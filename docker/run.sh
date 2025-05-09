#!/bin/bash

name="maestro"

docker run -it --rm \
    --name "$name" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    "$name"
