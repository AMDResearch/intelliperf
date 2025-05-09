# syntax=docker/dockerfile:1.4

FROM maawad/therock-gfx90a-release_debug

ARG MAESTRO_HOME=/maestro

# Set environment variables
ENV LANG=en_US.UTF-8 \
    ROCM_PATH=/opt/rocm \
    GT_COLOR=0
ENV GT_PATH=/opt/logduration/bin/logDuration
ENV ROCM_PATH=/opt/rocm
ENV PATH=/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=""
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
ENV HIP_DEVICE_LIB_PATH=/opt/rocm/lib/llvm/amdgcn/bitcode
ENV CXX=$ROCM_PATH/bin/hipcc
ENV CC=$ROCM_PATH/bin/hipcc

# Install dependencies
RUN apt-get update && apt-get install -y \
    libzstd-dev \
    python3-setuptools \
    python3-wheel \
    libdwarf-dev \
    locales \
    gdb \
    && locale-gen en_US.UTF-8

# Add GitHub trusted host
RUN mkdir -p ~/.ssh && \
    touch ~/.ssh/known_hosts && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts && \
    chmod 700 ~/.ssh && \
    chmod 644 ~/.ssh/known_hosts

# Set the working directory
WORKDIR $MAESTRO_HOME

# Install Maestro
RUN  --mount=type=ssh \
    . /app/.venv/bin/activate && \
    git clone git@github.com:AMDResearch/maestro.git --branch muhaawad/cleanup-examples . && \
    ./scripts/build_examples.sh

#    pip install -e . && \
#    python3 scripts/install_tool.py --all && \
#    ./scripts/build_examples.sh

# Update PATH
ENV PATH="$MAESTRO_HOME/external/logduration/omniprobe:$PATH:/$MAESTRO_HOME/external/rocprofiler-compute/src:$PATH:/$MAESTRO_HOME/src:$PATH"
ENV LD_LIBRARY_PATH="/$MAESTRO_HOME/external/logduration/build:$LD_LIBRARY_PATH"

#CMD ["/bin/bash"]
#ENTRYPOINT [".", "/app/.venv/bin/activate"]
ENTRYPOINT ["/bin/bash", "-c", ". /app/.venv/bin/activate && exec bash"]
