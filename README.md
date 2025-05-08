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

We provide an Apptainer image containing all the dependencies. To get started, run:
```
./apptainer/build.sh
```

To start the Apptainer container, run:

```
./apptainer/run.sh
```

## Install from source

1. Clone:

```shell
git clone git@github.com:AARInternal/maestro.git
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
export PATH=$(pwd)/external/logduration/omniprobe:$PATH
export PATH=$(pwd)/external/rocprofiler-compute/src:$PATH
export PATH=$(pwd)/$maestro/src:$PATH
```



## Usage:

```console
$ maestro --help
usage: 
        maestro [options] -- <profile_cmd>

        Example:
        # Run maestro to optimize bank conflicts in a HIP app
        maestro -s ~/rocBLAS/build.sh -f bankConflict -- ~/rocBLAS/build/bin/rocblas_gemm
        # Run maestro to diagnose a Triton app
        maestro -- python3 gemm.py
        

Optimize and analyze the given application based on available Maestro formulas.

optional arguments:
  -h, --help                  show this help message and exit
  -v, --verbose               Increase verbosity level (e.g., -v, -vv, -vvv).

required arguments:
  -- [ ...]                   Provide the command to launch the application.

optional arguments:
  -s SCRIPT, --script SCRIPT  A script to build your application.
  -f {bankConflict,diagnoseOnly}, --formula {bankConflict,diagnoseOnly}
                              Specify the formula to use for optimization (default: diagnoseOnly).
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                              Path to the output file (optional)
  -c CI_OUTPUT_FILE, --ci_output_file CI_OUTPUT_FILE
                              Path to the output file for CI integration (optional)
```
