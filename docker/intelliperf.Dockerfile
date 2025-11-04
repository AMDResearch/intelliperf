FROM rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0

ARG INTELLIPERF_HOME=/intelliperf

ENV LANG=en_US.UTF-8
ENV PATH=/opt/conda/envs/py_3.10/bin:/opt/rocm/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
ENV ROCM_PATH=/opt/rocm
ENV OMPI_MCA_mtl="^ofi"
ENV OMPI_MCA_pml="ob1"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    wget \
    clang \
    lld \
    libzstd-dev \
    libomp-dev \
    vim \
    libdwarf-dev \
    gdb \
    tmux \
    locales \
    && locale-gen en_US.UTF-8

# Upgrade pip
RUN pip install --upgrade pip

# Install additional Python packages for intelliperf
RUN pip install --no-cache-dir \
    astunparse==1.6.2 \
    colorlover \
    dash-bootstrap-components \
    dash-svg \
    dash>=3.0.0 \
    kaleido==0.2.1 \
    matplotlib \
    numpy>=1.17.5 \
    pandas>=1.4.3 \
    plotext \
    plotille \
    pymongo \
    pyyaml \
    setuptools \
    tabulate \
    textual \
    textual_plotext \
    textual-fspicker \
    tqdm \
    tomli \
    ml_dtypes \
    dspy==2.6.27 \
    duckdb \
    rich \
    pytest \
    litellm[proxy]

# Set the working directory
WORKDIR $INTELLIPERF_HOME