#!/bin/bash

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


# Auto config SSH agent
if [ ! -S ~/.ssh/ssh_auth_sock ]; then
    eval `ssh-agent` > /dev/null
    ln -sf "$SSH_AUTH_SOCK" ~/.ssh/ssh_auth_sock
fi
export SSH_AUTH_SOCK=~/.ssh/ssh_auth_sock
[ -f ~/.ssh/id_rsa ] && ssh-add ~/.ssh/id_rsa
[ -f ~/.ssh/id_ed25519 ] && ssh-add ~/.ssh/id_ed25519

ssh_auth_sock_path=$(readlink -f "$SSH_AUTH_SOCK")
# Build the Singularity container
#   --build-arg SSH_AUTH_SOCK=$SSH_AUTH_SOCK is used to pass the SSH agent socket to the container
#   (advantage of this method is that the key is at no point copied to the container image.)
#   If your SSH_AUTH_SOCK will not already bound to the container, and is available at /run/..., add `--bind /run` to the build command
definition="apptainer/maestro.def"

if [[ $debug -eq 1 ]]; then
    image="apptainer/maestro_debug.sif"
    cmake_build_type="Debug"
else
    image="apptainer/maestro.sif"
    cmake_build_type="Release"
fi

apptainer build \
    --build-arg SSH_AUTH_SOCK=${ssh_auth_sock_path} \
    --build-arg CMAKE_BUILD_TYPE=${cmake_build_type}\
     $image $definition