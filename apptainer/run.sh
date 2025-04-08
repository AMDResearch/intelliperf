#!/bin/bash

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"
working_dir=$(pwd)

cd $parent_dir

size=2048
debug=0

while [[ $# -gt 0 ]]; do
    case $1 in
        -s)
            size=$2
            shift 2
            ;;
        -d|--debug)
            debug=1
            shift
            ;;
        *)
            echo "Usage: $0 [-s size] [-d|--debug]"
            exit 1
            ;;
    esac
done
workload=$(date +"%Y%m%d%H%M%S")
overlay="/tmp/maestro_overlay_$(whoami)_$workload.img"
if [ ! -f $overlay ]; then
    echo "[Log] Overlay image ${overlay} does not exist. Creating overlay of ${size} MiB..."
    apptainer overlay create --size ${size} --create-dir /var/cache/maestro ${overlay}
else
    echo "[Log] Overlay image ${overlay} already exists. Using this one."
fi
echo "[Log] Utilize the directory /var/cache/maestro as a sandbox to store data you'd like to persist between container runs."

# Run the container
if [[ $debug -eq 1 ]]; then
    image="apptainer/maestro_debug.sif"
else
    image="apptainer/maestro.sif"
fi
apptainer exec --overlay ${overlay} --pwd "$working_dir" --cleanenv --env OPENAI_API_KEY=$OPENAI_API_KEY $image bash --rcfile /etc/bash.bashrc