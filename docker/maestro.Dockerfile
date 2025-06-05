# syntax=docker/dockerfile:1.4

FROM rocm/vllm-dev:nightly_aiter_integration_final_20250325



ENV LANG=en_US.UTF-8
ARG MAESTRO_HOME=/maestro
ENV GT_COLOR=0
ENV PATH=/opt/rocm/bin:$PATH
ENV PATH=$MAESTRO_HOME/external/logduration/install/bin/logDuration:$MAESTRO_HOME/external/rocprofiler-compute/src:$MAESTRO_HOME/src:$PATH

# Install dependencies
RUN apt-get update && apt-get install -y \
    libzstd-dev \
    python3-setuptools \
    python3-wheel \
    libdwarf-dev \
    rocm-llvm-dev\
    locales \
    gdb \
    && locale-gen en_US.UTF-8

# Temporary until we merge the logduration PR
RUN apt-get install -y nlohmann-json3-dev

# Add GitHub trusted host
RUN mkdir -p ~/.ssh && \
    touch ~/.ssh/known_hosts && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    chmod 700 ~/.ssh && \
    chmod 644 ~/.ssh/known_hosts

# Set the working directory
WORKDIR $MAESTRO_HOME

# Clone Maestro
RUN  --mount=type=ssh \
    git clone git@github.com:AMDResearch/maestro.git .
