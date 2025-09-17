FROM rocm/vllm-dev:nightly_aiter_integration_final_20250325

ARG INTELLIPERF_HOME=/intelliperf

ENV LANG=en_US.UTF-8
ENV PATH=/opt/rocm/bin:$PATH

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

# Set the working directory
WORKDIR $INTELLIPERF_HOME