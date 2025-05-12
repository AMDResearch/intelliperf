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
