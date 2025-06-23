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

name="intelliperf"

docker run -it --rm \
    --name "$name" \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    -v $HOME/.ssh:/tmp/ssh:ro \
    -v $(pwd):$(pwd) \
    -w $(pwd) \
    -e LLM_GATEWAY_KEY="$LLM_GATEWAY_KEY" \
    -e SSH_AUTH_SOCK="$SSH_AUTH_SOCK" \
    -v $SSH_AUTH_SOCK:$SSH_AUTH_SOCK \
    "$name" \
    bash -c "cp -r /tmp/ssh/* /root/.ssh/ 2>/dev/null || true && chown -R root:root /root/.ssh && chmod 700 /root/.ssh && chmod 600 /root/.ssh/config /root/.ssh/id_* /root/.ssh/known_hosts 2>/dev/null || true; exec bash"
