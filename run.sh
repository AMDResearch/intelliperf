#!/bin/bash

timestamp=$(date +"%Y%m%d_%H%M%S")

ssh_auth_sock_path=$(readlink -f "$SSH_AUTH_SOCK")
# Build the Singularity container
#   --build-arg SSH_AUTH_SOCK=$SSH_AUTH_SOCK is used to pass the SSH agent socket to the container
#   (advantage of this method is that the key is at no point copied to the container image.)
#   If your SSH_AUTH_SOCK will not already bound to the container, and is available at /run/..., add `--bind /run` to the build command
apptainer build \
    --build-arg SSH_AUTH_SOCK=${ssh_auth_sock_path} \
    "maestro_${timestamp}.sif" maestro.def