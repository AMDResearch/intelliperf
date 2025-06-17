<!--
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Maestro: Orchestrating the Omniverse

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![Maestro](./images/maestro.png)

## Overview

**Maestro** is a tool that reports and optimizes performance bottlenecks in an automated workflow. Given a target application, our tool generates kernel report cards containing performance metrics and their source code object alongside suggestions for code improvements. Maestro orchestrates existing Omni-tools such as [rocprofiler-compute](https://github.com/ROCm/rocprofiler-compute), and [guided-tuning](https://github.com/AMDResearch/guided-tuning) in addition to new ones like _Accordo_ for correctness validation and [nexus](https://github.com/AMDResearch/nexus) for code object back to source code mapping.

### Key Features

* **AI-Powered Optimization**: Automatically identifies and optimizes common performance bottlenecks using Large Language Models
* **Precise Analysis**: Pinpoints performance issues down to specific source code lines
* **Comprehensive Coverage**: Supports multiple bottleneck types:
  - Bank conflicts
  - Memory access patterns
  - Atomic contention
  - And more to come...
* **Diagnostic Mode**: Run in diagnose-only mode to analyze performance without making code changes

## Installation

### Quick Start with Containers

We provide both Apptainer and Docker images for easy setup:

#### Using Apptainer
```bash
./apptainer/build.sh
./apptainer/run.sh
```
#### Using Docker
```bash
./docker/build.sh
./docker/run.sh
```
#### For baremetal installation


1. **Install Additional Dependencies**:
   ```bash
   # ROCm dependencies
   apt-get install -y rocm-llvm-dev libzstd-dev

   # KernelDB dependencies
   apt-get install -y libdwarf-dev

   # Omniperf dependencies
   apt-get install -y locales
   locale-gen en_US.UTF-8
   ```

### Installation from Source

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:AMDResearch/maestro.git
   cd maestro
   ```

2. **Install Maestro**:
   ```bash
   pip install -e .
   ```

3. **Install Dependencies**:
   ```bash
   python3 scripts/install_tool.py --all
   ```

## Environment Variables

Set the following environment variable for AI-powered optimization:

```bash
export LLM_GATEWAY_KEY="your_api_key_here"
```

Required for bank conflicts, memory access patterns, and atomic contention optimization.

## Supported GPUs

Maestro currently supports:

- **MI300X**

> **Note**: Maestro may work on other AMD GPUs with ROCm compatibility, but has only been tested on MI300X.

## Usage

Maestro can be used to analyze and optimize your GPU applications:

```bash
maestro [options] -- <profile_cmd>
```

### Examples

```bash
# Optimize bank conflicts in a HIP application
maestro -b ~/rocBLAS/build.sh -f bankConflict -- ~/rocBLAS/build/bin/rocblas_gemm

# Diagnose a Triton application
maestro -- python3 gemm.py
```

### Command Line Options

| Option                           | Description          |
|----------------------------------|----------------------|
| `-h, --help` | Show help message and exit |
| `-v, --verbose` | Increase verbosity level (e.g., -v, -vv, -vvv) |
| `-b, --build_command` | Command to build your application |
| `-i, --instrument_command` | Command to build your application with instrument |
| `-p, --project_directory` | Directory containing your codebase |
| `-f, --formula` | Optimization formula to use (bankConflict, memoryAccess, atomicContention, diagnoseOnly) |
| `--top_n` | Control top-n kernels in diagnoseOnly mode (default: 10) |
| `--num_attempts` | Control optimization attempts (default: 10) |
| `-o, --output_file` | Path to output file |
| `-t, --accordo_absolute_tolerance` | Validation tolerance |

## Documentation

- [Running Examples](examples/README.md)

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on how to set up your development environment and contribute to the project.

## Support

For support, please:
1. Open an [issue](https://github.com/AMDResearch/maestro/issues)
2. Contact the development team

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

