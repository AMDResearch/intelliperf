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

# IntelliPerf: LLM-Powered Autonomous GPU Performance Engineer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/AMDResearch/intelliperf/actions/workflows/lint.yml) [![IntelliPerf CI](https://github.com/AMDResearch/intelliperf/actions/workflows/ci.yml/badge.svg)](https://github.com/AMDResearch/intelliperf/actions/workflows/ci.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15845118.svg)](https://doi.org/10.5281/zenodo.15845118)


> [!IMPORTANT]  
> This project is intended for research purposes only and is provided by AMD Research and Advanced Development team. 
This is not a product. Use it at your own risk and discretion.

![IntelliPerf](./images/intelliperf.png)

## Overview

**IntelliPerf** is an automated performance engineering framework that addresses the complex challenge of GPU kernel optimization. Manual optimization requires deep domain expertise and is time-consuming, error-prone, and resource-intensive. IntelliPerf systematizes this workflow by orchestrating a comprehensive toolchain that automatically profiles applications using [rocprofiler-compute](https://github.com/ROCm/rocprofiler-compute), identifies high-level bottlenecks with [Guided Tuning](./external/guided-tuning/), pinpoints specific source code lines using [Omniprobe](https://github.com/AARInternal/omniprobe), generates optimized code through Large Language Models (LLMs), and validates results using [Accordo](src/accordo/) for correctness and performance. Built on a modular "formula-driven" architecture, it targets specific bottlenecks like bank conflicts, memory access patterns, and atomic contention through a sophisticated multi-stage optimization loop that includes profiling, analysis, code generation, and automated validation.

### Key Features

* **AI-Powered Optimization**: Generates optimized code using LLMs with iterative feedback for performance improvements
* **Precise Analysis**: Pinpoints performance issues down to specific source code lines using compiler-based instrumentation
* **Automated Validation**: Validates both correctness and performance improvements through runtime comparison
* **Comprehensive Coverage**: Supports multiple bottleneck types (bank conflicts, memory access, atomic contention)
* **CI/CD Integration**: Seamless workflow integration with automated pull request generation
* **Extensible Architecture**: Formula-driven design for easy addition of new optimization targets

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

> [!NOTE]
> Due to the complex dependency chain, IntelliPerf currently supports development mode installation only. Future versions will support standard pip installation.

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:AMDResearch/intelliperf.git
   cd intelliperf
   ```

2. **Install IntelliPerf**:
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

Required for bank conflicts, memory access patterns, and atomic contention optimization. The AI-powered optimization supports various language models and providers through the `--provider` and `--model` command line arguments. The key should be the backend key for the specified provider.

## Supported GPUs

IntelliPerf currently supports:

- **MI300X**

> [!NOTE]
> IntelliPerf may work on other AMD GPUs with ROCm compatibility, but has only been tested on MI300X.

## Usage

IntelliPerf can be used to analyze and optimize your GPU applications:

```bash
intelliperf [options] -- <profile_cmd>
```

### Examples

```bash
# Optimize bank conflicts in a HIP application
intelliperf -b ~/rocBLAS/build.sh -f bankConflict -- ~/rocBLAS/build/bin/rocblas_gemm

# Diagnose a Triton application
intelliperf -- python3 gemm.py
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
| `-m, --model` | Specify the model to use for optimization (default: gpt-4o) |
| `-r, --provider` | Specify the provider to use for optimization (default: openai) |
| `-l, --in_place` | Modify source files in place during optimization (default: creates backups) |

> [!NOTE]
> IntelliPerf copies the entire project directory to a temporary location. Make sure your project doesn't include any temporary CMake files if you pass the `project_directory` flag.


## Documentation

- [IntelliPerf Technical Paper](docs/IntelliPerf.md) - Detailed technical overview of the IntelliPerf framework
- [Running Examples](examples/README.md)
- [AMD Developer Cloud Setup Guide](docs/DEVCLOUD.md) - Step-by-step instructions for setting up IntelliPerf on AMD Developer Cloud GPU droplets


## Citation

If you use IntelliPerf or discuss our work in your research, please always cite our work:

```bibtex
@software{   Awad:2025:ILP,
  author        = {Muhammad Awad and Cole Ramos and Keith Lowery},
  title         = {IntelliPerf: {LLM}-Powered Autonomous {GPU} Performance Engineer},
  year          = 2025,
  month         = jul,
  doi           = {10.5281/zenodo.15845118},
  url           = {https://github.com/AMDResearch/intelliperf},
  code          = {https://github.com/AMDResearch/intelliperf}
}
```

You can also use the [CITATION.cff](CITATION.cff) file in the repository root for automatic citation generation.


## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details on how to set up your development environment and contribute to the project.

## Support

For support, please:
1. Open an [issue](https://github.com/AMDResearch/intelliperf/issues/new/choose)
2. Contact the development team

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

