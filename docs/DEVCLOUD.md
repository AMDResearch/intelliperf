# AMD Developer Cloud Setup Guide

This guide provides step-by-step instructions for setting up IntelliPerf on the AMD Developer Cloud environment.

## Prerequisites

Before starting, ensure you have access to an AMD Developer Cloud and create a GPU Droplet.

## Environment Setup

### 1. Set ROCm Environment Variables

First, set up the ROCm environment variables:

```bash
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib
```

**Note**: You may want to add these to your shell profile (`.bashrc`, `.zshrc`, etc.) for persistence across sessions.

### 2. Install System Dependencies

Install the required system packages:

```bash
# Update package list
sudo apt-get update && sudo apt-get install -y python3-venv cmake  rocm-llvm-dev libzstd-dev libdwarf-dev
```

### 3. Create and Activate Virtual Environment

Create a Python virtual environment to isolate IntelliPerf dependencies:

```bash
# Create virtual environment
python3 -m venv intelliperf_env

# Activate virtual environment
source intelliperf_env/bin/activate
```

**Note**: Always activate the virtual environment before working with IntelliPerf.

## IntelliPerf Installation

### 1. Clone the Repository

```bash
git clone git@github.com:AMDResearch/intelliperf.git
cd intelliperf
```

### 2. Install IntelliPerf

Install IntelliPerf in development mode:

```bash
pip install -e .
```

### 3. Install Tool Dependencies

Install all required tool dependencies:

```bash
python3 scripts/install_tool.py --all
```


Next, you can run the examples! See the [Examples README](../examples/README.md) for detailed information about available examples and how to run them.
