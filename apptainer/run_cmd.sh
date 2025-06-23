#!/bin/bash
################################################################################
# MIT License

# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

cd $parent_dir

size=2048
cmd=""
debug=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -s)
            size=$2
            shift 2
            ;;
        --cmd)
            cmd=$2
            shift 2
            ;;
        -d|--debug)
            debug=1
            shift
            ;;
        *)
            echo "Usage: $0 [-s size] --cmd '<command>' [-d|--debug]"
            exit 1
            ;;
    esac
done

workload=$(date +"%Y%m%d%H%M%S")

# Create filesystem image overlay, if it doesn't exist
overlay="/tmp/intelliperf_overlay_$(whoami)_$workload.img"
if [ ! -f $overlay ]; then
    echo "[Log] Overlay image ${overlay} does not exist. Creating overlay of ${size} MiB..."
    apptainer overlay create --size ${size} --create-dir /var/cache/intelliperf ${overlay}
else
    echo "[Log] Overlay image ${overlay} already exists. Using this one."
fi
echo "[Log] Utilize the directory /var/cache/intelliperf as a sandbox to store data you'd like to persist between container runs."

# Run the container
if [[ $debug -eq 1 ]]; then
    image="apptainer/intelliperf_debug.sif"
else
    image="apptainer/intelliperf.sif"
fi
echo "cmd: $cmd"
apptainer exec --overlay "${overlay}"\
            --cleanenv --env OPENAI_API_KEY="$OPENAI_API_KEY"\
            "$image" bash --rcfile /etc/bash.bashrc\
            -c "cd src && eval \"$cmd\""