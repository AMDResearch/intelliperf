#!/bin/bash
set -eo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

clean=false
parallel=8
build_dir=build
verbose=false

print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -c, --clean       Clean build directory"
    echo "  -v, --verbose     Print verbose output"
    echo "  -j, --jobs <num>  Set number of parallel jobs"
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

cmake -B "$build_dir"

cmake_args=()
if [ "$verbose" = true ]; then
    cmake_args+=(--verbose)
fi

cmake --build "$build_dir" --parallel "$parallel" "${cmake_args[@]}"

popd > /dev/null
