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
definition="apptainer/intelliperf.def"

if [[ $debug -eq 1 ]]; then
    image="apptainer/intelliperf_debug.sif"
    cmake_build_type="Debug"
else
    image="apptainer/intelliperf.sif"
    cmake_build_type="Release"
fi

apptainer build \
    --build-arg SSH_AUTH_SOCK=${ssh_auth_sock_path} \
    --build-arg CMAKE_BUILD_TYPE=${cmake_build_type}\
     $image $definition