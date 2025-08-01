#!/bin/bash

SKIP_DUPLICATES=false
if [[ "$1" == "--skip-duplicates" ]]; then
    SKIP_DUPLICATES=true
    echo "Skipping existing log files."
fi

# Create swizzling directories if they don't exist
mkdir -p examples/triton/autogen_10/swizzling_aware
mkdir -p examples/triton/autogen_10/swizzling_unaware
mkdir -p examples/triton/autogen_science_10/swizzling_aware
mkdir -p examples/triton/autogen_science_10/swizzling_unaware

# autogen_10 kernels
AUTOGEN_DIR="examples/triton/autogen_10"
AUTOGEN_LOG_DIR_AWARE="$AUTOGEN_DIR/swizzling_aware"
AUTOGEN_LOG_DIR_UNAWARE="$AUTOGEN_DIR/swizzling_unaware"
AUTOGEN_KERNELS=(
    "spmv.py"
    "fused_elementwise.py"
    "stencil_2d.py"
    "layer_norm.py"
    "softmax.py"
    "conv2d.py"
    "gemm.py"
    "transpose.py"
)

for kernel in "${AUTOGEN_KERNELS[@]}"; do
    runner_kernel="${kernel%.py}_runner.py"
    kernel_path="$AUTOGEN_DIR/$runner_kernel"
    echo "Adding execute permission to $kernel_path"
    chmod +x "$kernel_path"

    # Swizzling aware
    output_file_aware="$AUTOGEN_LOG_DIR_AWARE/${kernel%.py}.txt"
    if [ "$SKIP_DUPLICATES" = true ] && [ -f "$output_file_aware" ]; then
        echo "Log file $output_file_aware already exists, skipping."
    else
        echo "Running $runner_kernel with swizzling_test, output to $output_file_aware"
        echo timeout 20m intelliperf -vvv --top_n 1 --project_directory=./examples --formula=swizzling_test --output_kernel_file="$output_file_aware" --unittest_command="\"./triton/autogen_10/$runner_kernel --validate\"" -- "./triton/autogen_10/$runner_kernel" > "${output_file_aware%.txt}_log.txt" 2>&1
    fi

    # Swizzling unaware
    output_file_unaware="$AUTOGEN_LOG_DIR_UNAWARE/${kernel%.py}.txt"
    if [ "$SKIP_DUPLICATES" = true ] && [ -f "$output_file_unaware" ]; then
        echo "Log file $output_file_unaware already exists, skipping."
    else
        echo "Running $runner_kernel with swizzling_baseline, output to $output_file_unaware"
        echo timeout 20m intelliperf -vvv --top_n 1 --project_directory=./examples --formula=swizzling_baseline --output_kernel_file="$output_file_unaware" --unittest_command="\"./triton/autogen_10/$runner_kernel --validate\"" -- "./triton/autogen_10/$runner_kernel" > "${output_file_unaware%.txt}_log.txt" 2>&1
    fi
done

# autogen_science_10 kernels
AUTOGEN_SCIENCE_DIR="examples/triton/autogen_science_10"
AUTOGEN_SCIENCE_LOG_DIR_AWARE="$AUTOGEN_SCIENCE_DIR/swizzling_aware"
AUTOGEN_SCIENCE_LOG_DIR_UNAWARE="$AUTOGEN_SCIENCE_DIR/swizzling_unaware"
AUTOGEN_SCIENCE_KERNELS=(
    "gravity_potential.py"
    "ising_model.py"
    "black_scholes.py"
    "smith_waterman.py"
    "fdtd_2d.py"
    "n_body.py"
)

for kernel in "${AUTOGEN_SCIENCE_KERNELS[@]}"; do
    runner_kernel="${kernel%.py}_runner.py"
    kernel_path="$AUTOGEN_SCIENCE_DIR/$runner_kernel"
    echo "Adding execute permission to $kernel_path"
    chmod +x "$kernel_path"

    # Swizzling aware
    output_file_aware="$AUTOGEN_SCIENCE_LOG_DIR_AWARE/${kernel%.py}.txt"
    if [ "$SKIP_DUPLICATES" = true ] && [ -f "$output_file_aware" ]; then
        echo "Log file $output_file_aware already exists, skipping."
    else
        echo "Running $runner_kernel with swizzling_test, output to $output_file_aware"
        echo timeout 20m intelliperf -vvv --top_n 1 --project_directory=./examples --formula=swizzling_test --output_kernel_file="$output_file_aware" --unittest_command="\"./triton/autogen_science_10/$runner_kernel --validate\"" -- "./triton/autogen_science_10/$runner_kernel" > "${output_file_aware%.txt}_log.txt" 2>&1
    fi

    # Swizzling unaware
    output_file_unaware="$AUTOGEN_SCIENCE_LOG_DIR_UNAWARE/${kernel%.py}.txt"
    if [ "$SKIP_DUPLICATES" = true ] && [ -f "$output_file_unaware" ]; then
        echo "Log file $output_file_unaware already exists, skipping."
    else
        echo "Running $runner_kernel with swizzling_baseline, output to $output_file_unaware"
        echo timeout 20m intelliperf -vvv --top_n 1 --project_directory=./examples --formula=swizzling_baseline --output_kernel_file="$output_file_unaware" --unittest_command="\"./triton/autogen_science_10/$runner_kernel --validate\"" -- "./triton/autogen_science_10/$runner_kernel" > "${output_file_unaware%.txt}_log.txt" 2>&1
    fi
done

echo "All kernels processed."