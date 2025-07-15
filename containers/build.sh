#!/bin/bash

# Container name
name="intelliperf"

# Script directories
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"
cur_dir=$(pwd)

# Parse arguments
build_docker=false
build_apptainer=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --docker|-d)
      use_docker=true
      shift
      ;;
    --apptainer|-a)
      use_apptainer=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--docker|-d] [--apptainer|-a] -- Exactly one option is required."
      exit 1
      ;;
  esac
done

if [ "$build_docker" = false ] && [ "$build_apptainer" = false ]; then
    echo "Error: At least one of the options --docker or --apptainer is required."
    echo "Usage: $0 [--docker] [--apptainer]"
    echo "  --docker      Build Docker container"
    echo "  --apptainer   Build Apptainer container"
    exit 1
fi

pushd "$parent_dir"

if [ "$build_docker" = true ]; then
    echo "Building Docker container..."

    # Enable BuildKit and build the Docker image
    export DOCKER_BUILDKIT=1
    docker build \
        -t "$name:$(cat "$parent_dir/VERSION")" \
        -f "$script_dir/intelliperf.Dockerfile" \
        .

    echo "Docker build complete!"
fi

if [ "$build_apptainer" = true ]; then
    echo "Building Apptainer container..."

    # Check if apptainer is installed
    if ! command -v apptainer &> /dev/null; then
        echo "Error: Apptainer is not installed or not in PATH"
        echo "Please install Apptainer first: https://apptainer.org/docs/admin/main/installation.html"
        exit 1
    fi

    # Build the Apptainer container with ROCm version
    export ROCM_VERSION="$rocm_version"
    apptainer build \
      "${script_dir}/${name}_$(cat "$parent_dir/VERSION").sif" "$script_dir/intelliperf.def"

    echo "Apptainer build complete!"
fi
  
popd