#!/bin/bash

SKIP_DUPLICATES=false
if [[ "$1" == "--skip-duplicates" ]]; then
    SKIP_DUPLICATES=true
    echo "Skipping existing log files."
fi

# Create swizzling_logs directories if they don't exist
mkdir -p examples/triton/autogen_10/swizzling_logs_2
mkdir -p examples/triton/autogen_science_10/swizzling_logs_2

# autogen_10 kernels
AUTOGEN_DIR="examples/triton/autogen_10"
AUTOGEN_LOG_DIR="$AUTOGEN_DIR/swizzling_logs_2"
AUTOGEN_KERNELS=(
    "fft.py"
    "spmv.py"
    "fused_elementwise.py"
    "stencil_2d.py"
    "layer_norm.py"
    "softmax.py"
    "fused_attention.py"
    "conv2d.py"
    "gemm.py"
    "transpose.py"
)

for kernel in "${AUTOGEN_KERNELS[@]}"; do
    output_file="$AUTOGEN_LOG_DIR/${kernel%.py}_log.txt"

    if [ "$SKIP_DUPLICATES" = true ] && [ -f "$output_file" ]; then
        echo "Log file $output_file already exists, skipping $kernel."
        continue
    fi

    kernel_path="$AUTOGEN_DIR/$kernel"
    echo "Adding execute permission to $kernel_path"
    chmod +x "$kernel_path"

    echo "Running $kernel, output to $output_file"
    intelliperf -vvv --top_n 1 --project_directory=./examples --formula=swizzling_test --output_kernel_file="$output_file" -- "./triton/autogen_10/$kernel"
done

# autogen_science_10 kernels
AUTOGEN_SCIENCE_DIR="examples/triton/autogen_science_10"
AUTOGEN_SCIENCE_LOG_DIR="$AUTOGEN_SCIENCE_DIR/swizzling_logs_2"
AUTOGEN_SCIENCE_KERNELS=(
    "gravity_potential.py"
    "ising_model.py"
    "black_scholes.py"
    "smith_waterman.py"
    "pic_1d.py"
    "lbm_d2q9.py"
    "fdtd_2d.py"
    "jacobi_3d.py"
    "molecular_dynamics.py"
    "n_body.py"
)

for kernel in "${AUTOGEN_SCIENCE_KERNELS[@]}"; do
    output_file="$AUTOGEN_SCIENCE_LOG_DIR/${kernel%.py}_log.txt"

    if [ "$SKIP_DUPLICATES" = true ] && [ -f "$output_file" ]; then
        echo "Log file $output_file already exists, skipping $kernel."
        continue
    fi

    kernel_path="$AUTOGEN_SCIENCE_DIR/$kernel"
    echo "Adding execute permission to $kernel_path"
    chmod +x "$kernel_path"

    echo "Running $kernel, output to $output_file"
    intelliperf -vvv --top_n 1 --project_directory=./examples --formula=swizzling_test --output_kernel_file="$output_file" -- "./triton/autogen_science_10/$kernel"
done

echo "All kernels processed."

# Create cumulative CSV file
CSV_FILE="swizzling_results.csv"
echo "Kernel Name,Best Swizzling Pattern,L2 Hit Rate Improvement %,Speedup" > "$CSV_FILE"

# Process logs from autogen_10
for kernel in "${AUTOGEN_KERNELS[@]}"; do
    log_file="$AUTOGEN_LOG_DIR/${kernel%.py}_log.txt"
    if [ -f "$log_file" ]; then
        # Use awk to extract the best result section
        awk '
        BEGIN { record=0; pattern=""; }
        /--- BEST RESULT ---/ { record=1; next; }
        /--- END BEST RESULT ---/ { record=0; print kernel_name "," "\"" pattern "\"", l2_improvement "," speedup; }
        record {
            if ($1 == "Kernel" && $2 == "Name:") { kernel_name = $3; }
            if ($1 == "Best" && $2 == "Swizzling") { getline; pattern = $0; while (getline > 0 && !/L2 Hit Rate Improvement/) { pattern = pattern "\\n" $0; } }
            if ($1 == "L2" && $2 == "Hit") { l2_improvement = $6; }
            if ($1 == "Speedup:") { speedup = $2; }
        }
        ' "$log_file" >> "$CSV_FILE"
    fi
done

# Process logs from autogen_science_10
for kernel in "${AUTOGEN_SCIENCE_KERNELS[@]}"; do
    log_file="$AUTOGEN_SCIENCE_LOG_DIR/${kernel%.py}_log.txt"
    if [ -f "$log_file" ]; then
        # Use awk to extract the best result section
        awk '
        BEGIN { record=0; pattern=""; }
        /--- BEST RESULT ---/ { record=1; next; }
        /--- END BEST RESULT ---/ { record=0; print kernel_name "," "\"" pattern "\"", l2_improvement "," speedup; }
        record {
            if ($1 == "Kernel" && $2 == "Name:") { kernel_name = $3; }
            if ($1 == "Best" && $2 == "Swizzling") { getline; pattern = $0; while (getline > 0 && !/L2 Hit Rate Improvement/) { pattern = pattern "\\n" $0; } }
            if ($1 == "L2" && $2 == "Hit") { l2_improvement = $6; }
            if ($1 == "Speedup:") { speedup = $2; }
        }
        ' "$log_file" >> "$CSV_FILE"
    fi
done

echo "CSV file created: $CSV_FILE" 