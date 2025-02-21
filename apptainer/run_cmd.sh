#!/bin/bash


script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

cd $parent_dir

size=2048
cmd=""

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
        *)
            echo "Usage: $0 [-s size] --cmd '<command>'"
            exit 1
            ;;
    esac
done

workload=$(date +"%Y%m%d%H%M%S")

# Create filesystem image overlay, if it doesn't exist
overlay="/tmp/maestro_overlay_$(whoami)_$workload.img"
if [ ! -f $overlay ]; then
    echo "[Log] Overlay image ${overlay} does not exist. Creating overlay of ${size} MiB..."
    apptainer overlay create --size ${size} --create-dir /var/cache/maestro ${overlay}
else
    echo "[Log] Overlay image ${overlay} already exists. Using this one."
fi
echo "[Log] Utilize the directory /var/cache/maestro as a sandbox to store data you'd like to persist between container runs."

# Run the container
image="apptainer/maestro.sif"
echo "cmd: $cmd"
apptainer exec --overlay "${overlay}" --cleanenv --env OPENAI_API_KEY="$OPENAI_API_KEY" "$image" bash --rcfile /etc/bash.bashrc -c "cd src && eval \"$cmd\""