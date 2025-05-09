# syntax=docker/dockerfile:1.4

FROM ubuntu:22.04

ARG MAESTRO_HOME=/maestro

# Set environment variables
ENV LANG=en_US.UTF-8 \
    PATH="/opt/rocm/bin:$PATH" \
    LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH" \
    ROCM_PATH=/opt/rocm \
    GT_COLOR=0

# Install dependencies
RUN apt-get -y update && \
    apt-get install -y locales && \
    locale-gen en_US.UTF-8 && \
    apt-get install -y software-properties-common && \
    apt-get upgrade -y && \
    apt-get install -y build-essential python3 python3-pip python3-setuptools python3-wheel git wget clang lld libzstd-dev libomp-dev vim libdwarf-dev && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install 'cmake==3.22' && \
    \
    # Add GitHub trusted host
    mkdir -p ~/.ssh && \
    touch ~/.ssh/known_hosts && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    chmod 700 ~/.ssh && \
    chmod 644 ~/.ssh/known_hosts

# Install ROCm
RUN apt-get -y update && \
    wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/jammy/amdgpu-install_6.3.60303-1_all.deb && \
    apt-get -y install ./amdgpu-install_6.3.60303-1_all.deb && \
    apt-get -y update && \
    apt-get install -y rocm-dev rocm-llvm-dev rocm-hip-runtime-dev rocm-smi-lib rocminfo rocthrust-dev rocprofiler-compute rocblas rocm-gdb gdb tmux


# Set the working directory
WORKDIR $MAESTRO_HOME

# Install Maestro
RUN  --mount=type=ssh \
    git clone git@github.com:AMDResearch/maestro.git . && \
    pip install -e . && \
    python3 scripts/install_tool.py --all && \
    ./scripts/build_examples.sh

# Update PATH
ENV PATH="$MAESTRO_HOME/external/logduration/omniprobe:$PATH:/$MAESTRO_HOME/external/rocprofiler-compute/src:$PATH:/$MAESTRO_HOME/src:$PATH"
ENV LD_LIBRARY_PATH="/$MAESTRO_HOME/external/logduration/build:$LD_LIBRARY_PATH"
