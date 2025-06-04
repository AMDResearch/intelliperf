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

![Maestro](./images/maestro.png)


**Maestro** is a tool that reports and optimizes performance bottlenecks in an automated workflow. Given a target application, our tool generates kernel report cards containing performance metrics and their source code object alongside suggestions for code improvements. Maestro orchestrates existing Omni-tools such as [rocprofiler-compute](https://github.com/ROCm/rocprofiler-compute), and [guided-tuning](https://github.com/AARInternal/guided-tuning) in addition to new ones like _Accordo_ for correctness validation and [nexus](https://github.com/AARInternal/nexus) for code object back to source code mapping.


## Quick start

We provide both Apptainer and Docker images containing all the dependencies. To get started, run:
```
./apptainer/build.sh
```
or,
```
./docker/build.sh
```


To start the container, run:

```
./apptainer/run.sh
```
or,
```
./docker/run.sh
```


## Install from source

1. Clone:

```shell
git clone git@github.com:AMDResearch/maestro.git
cd maestro
```

2. Install Maestro:
```shell
pip install -e .
```

3. Install the dependencies:
```shell
python3 scripts/install_tool.py --all
```

4. Additional dependencies:
```shell
# For ROCm
apt-get install -y rocm-llvm-dev libzstd-dev

# For KernelDB
apt-get install -y libdwarf-dev

# For Omniperf
apt-get install -y locales
locale-gen en_US.UTF-8 
```

5. Add Maestro and dependencies to your path:

```shell
export PATH=$(pwd)/external/logduration/install/bin/logDuration:$PATH
export PATH=$(pwd)/external/rocprofiler-compute/src:$PATH
export PATH=$(pwd)/$maestro/src:$PATH
```


6. Build the examples (optional):

```shell
./scripts/build_examples.sh
```


## Usage:

```console
usage: 
        maestro [options] -- <profile_cmd>

        Example:
        # Run maestro to optimize bank conflicts in a HIP app
        maestro -s ~/rocBLAS/build.sh -f bankConflict -- ~/rocBLAS/build/bin/rocblas_gemm
        # Run maestro to diagnose a Triton app
        maestro -- python3 gemm.py
        

Optimize and analyze the given application based on available Maestro formulas.

options:
  -h, --help                  show this help message and exit
  -v, --verbose               Increase verbosity level (e.g., -v, -vv, -vvv).

required arguments:
  -- [ ...]                   Provide the command to launch the application.

optional arguments:
  -b , --build_command        A command to build your application. When project_directory is provided,
                              the command must be relative to the project directory.
  -i , --instrument_command   A command to instrument your application (required when formula is not diagnoseOnly). When project_directory is provided,
                              the command must be relative to the project directory.
  -p , --project_directory    The directory containing your entire codebase (required when formula is not diagnoseOnly)
  -f , --formula              Specify the formula to use for optimization.
                              Available options: bankConflict, memoryAccess, atomicContention, diagnoseOnly (default: diagnoseOnly)
  --top_n                     Control the top-n kernels collected in diagnoseOnly mode (default: 10)
  --num_attempts              Control the number of attempts in optimize mode (default: 10)
  -o , --output_file          Path to the output file
```
