#!/bin/bash
# SPDX-License-Identifier: MIT
# Generate IntelliPerf definition files from different base images

set -e

# Default values
BASE_IMAGE=""
OUTPUT_NAME=""
DEF_TEMPLATE="intelliperf.def"
SCRIPT_DIR=$(dirname $(realpath $0))

# Preset base images
declare -A PRESETS=(
    ["composable_kernel"]="rocm/composable_kernel:ck_ub24.04_rocm7.1.1_amd-staging"
    ["gemm"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["matrix_transpose1"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["vllm"]="rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210"
    ["counting_sketch"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["global_reduction"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["silu"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["Dataset_hip"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["gpu-kernel-agent"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["point_to_voxel"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["diff_gaussian_rasterization"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["gsplat"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["histogram"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["reduction"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["causal_conv1d_fwd"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["element_multiply"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["matrix_transpose"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["rocm7.0-pytorch"]="rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["rocm7.1-pytorch"]="rocm/pytorch:rocm7.1_ubuntu22.04_py3.10_pytorch_release_2.8.0"
    ["composable-kernel"]="rocm/composable_kernel:ck_pytorch"
    ["rocm6.2-pytorch"]="rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.4.0"
    ["rocm7.0-ubuntu22"]="rocm/rocm-terminal:rocm7.0-ubuntu22.04"
    ["rocm7.1-ubuntu22"]="rocm/rocm-terminal:rocm7.1-ubuntu22.04"
    ["rocm7.0-vllm"]="rocm/vllm:rocm7.0.0_vllm_0.11.2_20251210"
    
)

# Help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Generate IntelliPerf Apptainer definition files from different base images.

OPTIONS:
    -b, --base IMAGE        Base Docker image (required)
    -o, --output NAME       Output .def filename (default: intelliperf-<preset>.def)
    -t, --template FILE     Template .def file (default: intelliperf.def)
    -p, --preset NAME       Use preset base image (see list below)
    -l, --list              List available presets
    -h, --help              Show this help message

PRESETS:
EOF
    for preset in "${!PRESETS[@]}"; do
        echo "    $preset: ${PRESETS[$preset]}"
    done | sort
    echo ""
    echo "EXAMPLES:"
    echo "    # Generate def file from rocm7.1-pytorch preset"
    echo "    $0 --preset rocm7.1-pytorch"
    echo ""
    echo "    # Generate def file from custom base image"
    echo "    $0 --base rocm/pytorch:rocm7.2_ubuntu22.04_py3.10_pytorch_release_2.9.0 --output intelliperf-rocm72.def"
    echo ""
    echo "    # Generate using intelliperf_ck.def template with different base"
    echo "    $0 --preset rocm7.1-pytorch --template intelliperf_ck.def --output intelliperf-ck-rocm71.def"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--base)
            BASE_IMAGE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        -t|--template)
            DEF_TEMPLATE="$2"
            shift 2
            ;;
        -p|--preset)
            PRESET_NAME="$2"
            if [[ -z "${PRESETS[$PRESET_NAME]}" ]]; then
                echo "Error: Unknown preset '$PRESET_NAME'"
                echo "Use --list to see available presets"
                exit 1
            fi
            BASE_IMAGE="${PRESETS[$PRESET_NAME]}"
            # Set default output name based on preset if not specified
            if [[ -z "$OUTPUT_NAME" ]]; then
                OUTPUT_NAME="intelliperf-${PRESET_NAME}.def"
            fi
            shift 2
            ;;
        -l|--list)
            echo "Available presets:"
            for preset in "${!PRESETS[@]}"; do
                echo "  $preset: ${PRESETS[$preset]}"
            done | sort
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$BASE_IMAGE" ]]; then
    echo "Error: Base image is required (use --base or --preset)"
    show_help
    exit 1
fi

# Set default output name if not specified
if [[ -z "$OUTPUT_NAME" ]]; then
    # Extract a sanitized name from the base image
    SANITIZED=$(echo "$BASE_IMAGE" | sed 's/[\/:]/-/g')
    OUTPUT_NAME="intelliperf-${SANITIZED}.def"
fi

# Check if template exists
if [[ ! -f "$SCRIPT_DIR/$DEF_TEMPLATE" ]]; then
    echo "Error: Template file '$DEF_TEMPLATE' not found in $SCRIPT_DIR"
    exit 1
fi

echo "=========================================="
echo "IntelliPerf Definition File Generator"
echo "=========================================="
echo "Base Image:    $BASE_IMAGE"
echo "Template:      $DEF_TEMPLATE"
echo "Output:        $OUTPUT_NAME"
echo "=========================================="

# Replace the 'From:' line in the template and save to output
sed "s|^From:.*|From: ${BASE_IMAGE}|" "$SCRIPT_DIR/$DEF_TEMPLATE" > "$SCRIPT_DIR/$OUTPUT_NAME"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "=========================================="
    echo "Definition file generated successfully!"
    echo "Saved to: $SCRIPT_DIR/$OUTPUT_NAME"
    echo "=========================================="
    echo ""
    echo "To build the container:"
    echo "  apptainer build ${OUTPUT_NAME%.def}.sif $SCRIPT_DIR/$OUTPUT_NAME"
else
    echo ""
    echo "=========================================="
    echo "Generation failed!"
    echo "=========================================="
    exit 1
fi
