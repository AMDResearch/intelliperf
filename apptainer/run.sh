#!/bin/bash

size=1024
while getopts "s:" opt; do
    case $opt in
        s)
            size=$OPTARG
            ;;
        *)
            echo "Usage: $0 [-s size]"
            exit 1
            ;;
    esac
done

# Create filesystem image overlay, if it doesn't exist
if [ ! -f /tmp/maestro_overlay.img ]; then
    echo "[Log] Overlay image /tmp/maestro_overlay.img does not exist. Creating overlay of ${size} MiB..."
    apptainer overlay create --size ${size} --create-dir /var/cache/maestro /tmp/maestro_overlay.img
else
    echo "[Log] Overlay image /tmp/maestro_overlay.img already exists. Using this one."
fi
echo "[Log] Utilize the directory /var/cache/maestro as a sandbox to store data you'd like to persist between container runs."

# Run the container
image="apptainer/maestro.sif"
apptainer exec --overlay /tmp/maestro_overlay.img --cleanenv --env OPENAI_API_KEY=$OPENAI_API_KEY $image bash --rcfile /etc/bash.bashrc