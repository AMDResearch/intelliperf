#!/bin/bash

# Supports both Docker and Apptainer with automatic building

# Container name
name="intelliperf"

# Script directories
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
parent_dir="$(dirname "$script_dir")"

# Parse arguments
use_docker=false
use_apptainer=false
overlay_size=2048

while [[ $# -gt 0 ]]; do
  case $1 in
    --docker)
      use_docker=true
      shift
      ;;
    --apptainer)
      use_apptainer=true
      shift
      ;;
    -s|--overlay-size)
      overlay_size=$2
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--docker] [--apptainer] [-s|--overlay-size SIZE] -- Exactly one option is required."
      echo "  -s, --overlay-size SIZE   Size of overlay filesystem in MiB (default: 2048, Apptainer only)"
      exit 1
      ;;
  esac
done

# Validate arguments
if [ "$use_docker" = true ] && [ "$use_apptainer" = true ]; then
    echo "Error: Cannot use both --docker and --apptainer simultaneously."
    echo "Usage: $0 [--docker] [--apptainer]"
    exit 1
elif [ "$use_docker" = false ] && [ "$use_apptainer" = false ]; then
    echo "Error: Must specify either --docker or --apptainer."
    echo "Usage: $0 [--docker] [--apptainer]"
    echo "  --docker      Run using Docker container"
    echo "  --apptainer   Run using Apptainer container"
    exit 1
fi

echo "Starting intelliperf container..."
echo "Project directory will be mounted at $(pwd)"
echo "Any files you create/modify will persist after the container closes."
echo ""

if [ "$use_docker" = true ]; then
    echo "Using Docker containerization..."
    
    # Check if the Docker image exists
    if ! docker image inspect "$name:$(cat "$parent_dir/VERSION")" > /dev/null 2>&1; then
        echo "Docker image $name:$(cat "$parent_dir/VERSION") not found."
        echo "Building Docker image..."
        echo ""
        
        if ! "$script_dir/build.sh" --docker; then
            echo "Error: Failed to build Docker image."
            exit 1
        fi
        
        echo ""
        echo "Docker image built successfully!"
    else
        echo "Docker image found."
    fi
    
    # Run the Docker container
    echo "Running Docker container with project directory mounted..."
    docker run -it --rm \
        --device=/dev/kfd \
        --device=/dev/dri \
        --group-add video \
        --cap-add=SYS_PTRACE \
        --security-opt seccomp=unconfined \
        -e LLM_GATEWAY_KEY="$LLM_GATEWAY_KEY" \
        -v $(pwd):$(pwd) \
        -w $(pwd) \
        "$name:$(cat "$parent_dir/VERSION")"

elif [ "$use_apptainer" = true ]; then
    echo "Using Apptainer containerization..."
    
    # Check if apptainer is installed
    if ! command -v apptainer &> /dev/null; then
        echo "Error: Apptainer is not installed or not in PATH"
        echo "Please install Apptainer first: https://apptainer.org/docs/admin/main/installation.html"
        exit 1
    fi
    
    # Apptainer image filename
    apptainer_image="$script_dir/${name}_$(cat "$parent_dir/VERSION").sif"
    
    # Check if the Apptainer image exists
    if [ ! -f "$apptainer_image" ]; then
        echo "Apptainer image $apptainer_image not found."
        echo "Building Apptainer image automatically..."
        echo ""
        
        if ! "$script_dir/build.sh" --apptainer; then
            echo "Error: Failed to build Apptainer image."
            exit 1
        fi
        
        echo ""
        echo "Apptainer image built successfully!"
    else
        echo "Apptainer image found."
    fi

    # Create overlay filesystem for writable areas
    workload=$(date +"%Y%m%d%H%M%S")
    overlay="/tmp/intelliperf_overlay_$(whoami)_$workload.img"
    if [ ! -f "$overlay" ]; then
        echo "[Log] Overlay image ${overlay} does not exist. Creating overlay of ${overlay_size} MiB..."
        apptainer overlay create --size ${overlay_size} --create-dir /var/cache/intelliperf ${overlay}
    else
        echo "[Log] Overlay image ${overlay} already exists. Using this one."
    fi
    echo "[Log] Overlay filesystem provides writable areas for profiling tools."
    echo "[Log] Use /var/cache/intelliperf as a sandbox for persistent data between container runs."
    
    # Run the Apptainer container
    echo "Running Apptainer container with project directory mounted..."
    cd "$parent_dir"
    apptainer exec \
        --cleanenv \
        --pwd $(pwd) \
        --overlay ${overlay} \
        --env LLM_GATEWAY_KEY=$LLM_GATEWAY_KEY \
        "$apptainer_image" \
        /bin/bash \
        --rcfile /etc/bashrc
fi

echo "Container session ended." 