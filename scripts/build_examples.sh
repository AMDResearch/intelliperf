#!/bin/bash
################################################################################
# MIT License

# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

set -eo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

clean=false
parallel=8
build_dir=build
verbose=false
build_type=RelWithDebInfo

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c, --clean         Clean build directory"
    echo "  -v, --verbose       Print verbose output"
    echo "  -b, --build <type>  CMake build type (Release, Debug, RelWithDebInfo, MinSizeRel)"
    echo "  -j, --jobs <num>    Set number of parallel jobs"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--clean)
            clean=true
            shift
            ;;
        -v|--verbose)
            verbose=true
            shift
            ;;
        -b|--build)
            if [[ -n "$2" && "$2" =~ ^(Release|Debug|RelWithDebInfo|MinSizeRel)$ ]]; then
                build_type="$2"
                shift 2
            else
                echo "Error: --build requires a valid build type (Release, Debug, RelWithDebInfo, MinSizeRel)"
                print_usage
                exit 1
            fi
            ;;
        -j|--jobs)
            if [[ -n "$2" && "$2" =~ ^[0-9]+$ ]]; then
                parallel="$2"
                shift 2
            else
                echo "Error: --jobs requires a numeric argument"
                print_usage
                exit 1
            fi
            ;;
        *)
            echo "Invalid option: $1"
            print_usage
            exit 1
            ;;
    esac
done

pushd "$script_dir/../examples" > /dev/null

if [ "$clean" = true ]; then
    echo "Cleaning build directory..."
    rm -rf "$build_dir"
fi

cmake -B "$build_dir" -DCMAKE_BUILD_TYPE="$build_type"

cmake_args=()
if [ "$verbose" = true ]; then
    cmake_args+=(--verbose)
fi

cmake --build "$build_dir" --parallel "$parallel" "${cmake_args[@]}"

popd > /dev/null
