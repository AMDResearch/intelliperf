#!/bin/bash

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
image="apptainer/maestro.sif"
apptainer build \
    --build-arg SSH_AUTH_SOCK=${ssh_auth_sock_path} \
     $image $definition