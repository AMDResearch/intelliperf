#!/bin/bash

# Container name
name="maestro"

# Script directories
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"
cur_dir=$(pwd)

pushd "$script_dir"

# Auto-configure SSH agent
if [ ! -S ~/.ssh/ssh_auth_sock ]; then
    eval "$(ssh-agent)" > /dev/null
    ln -sf "$SSH_AUTH_SOCK" ~/.ssh/ssh_auth_sock
fi
export SSH_AUTH_SOCK=~/.ssh/ssh_auth_sock

# Add default keys if they exist
[ -f ~/.ssh/id_rsa ] && ssh-add ~/.ssh/id_rsa
[ -f ~/.ssh/id_ed25519 ] && ssh-add ~/.ssh/id_ed25519
[ -f ~/.ssh/id_github ] && ssh-add ~/.ssh/id_github

# Enable BuildKit and build the Docker image
export DOCKER_BUILDKIT=1
docker build \
    --ssh default \
    -t "$name" \
    -f "$script_dir/maestro.Dockerfile" \
    .

popd
