#!/bin/bash

# New script: Run swizzling_test only for selected kernels across multiple models
# Kernels: gemm, layer_norm (ML); fdtd_2d, smith_waterman (Science)
# Models: gpt-4o, gpt-oss-120b, o3-mini

SKIP_DUPLICATES=false
if [[ "$1" == "--skip-duplicates" ]]; then
    SKIP_DUPLICATES=true
    echo "Skipping existing log files."
fi

# Directories
AUTOGEN_DIR="examples/triton/autogen_10"
AUTOGEN_SCIENCE_DIR="examples/triton/autogen_science_10"

# Selected kernels
AUTOGEN_KERNELS=(
    "gemm.py"
    "layer_norm.py"
)

AUTOGEN_SCIENCE_KERNELS=(
    "fdtd_2d.py"
    "smith_waterman.py"
)

# Models to run
MODELS=(
    #"gpt-4o"
    "gpt-4.1-mini"
    "o3-mini"
)

# Create output directories that avoid overwriting prior runs
# We will organize outputs per model
for MODEL in "${MODELS[@]}"; do
    mkdir -p "$AUTOGEN_DIR/swizzling_aware_models/$MODEL/logs"
    mkdir -p "$AUTOGEN_SCIENCE_DIR/swizzling_aware_models/$MODEL/logs"
done

# Helper to run a single kernel
run_kernel() {
    local base_dir="$1"
    local kernel_py="$2"
    local runner_kernel="${kernel_py%.py}_runner.py"
    local kernel_path="$base_dir/$runner_kernel"

    echo "Adding execute permission to $kernel_path"
    chmod +x "$kernel_path"

    for MODEL in "${MODELS[@]}"; do
        local out_dir="$base_dir/swizzling_aware_models/$MODEL"
        local output_file="$out_dir/${kernel_py%.py}.txt"
        local log_file="$out_dir/logs/${kernel_py%.py}_log.txt"

        # Ensure directories exist even if the pre-creation step was skipped
        mkdir -p "$out_dir/logs"

        if [ "$SKIP_DUPLICATES" = true ] && [ -f "$output_file" ]; then
            echo "Log file $output_file already exists, skipping for model $MODEL."
            continue
        fi

        echo "Running $runner_kernel with swizzling_test (model=$MODEL), output to $output_file"
        timeout 20m intelliperf -vvv \
            --top_n 1 \
            --project_directory=./examples \
            --formula=swizzling_test \
            --model="$MODEL" \
            --output_kernel_file="$output_file" \
            --unittest_command="\"./triton/$(basename "$base_dir")/$runner_kernel --validate\"" \
            -- "./triton/$(basename "$base_dir")/$runner_kernel" > "$log_file" 2>&1
    done
}

# Run selected ML kernels (autogen_10)
for kernel in "${AUTOGEN_KERNELS[@]}"; do
    run_kernel "$AUTOGEN_DIR" "$kernel"
done

# Run selected Science kernels (autogen_science_10)
for kernel in "${AUTOGEN_SCIENCE_KERNELS[@]}"; do
    run_kernel "$AUTOGEN_SCIENCE_DIR" "$kernel"
done

echo "Selected kernels processed for models: ${MODELS[*]}" 