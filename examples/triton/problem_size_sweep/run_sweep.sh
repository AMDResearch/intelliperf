#!/bin/bash

# Define kernels and their problem size arguments
declare -A KERNELS
KERNELS=(
    [gemm]="--m 512 --n 512 --k 512;--m 1024 --n 1024 --k 1024;--m 2048 --n 2048 --k 2048;--m 4096 --n 4096 --k 4096;--m 8192 --n 8192 --k 8192;--m 1024 --n 2048 --k 512;--m 2048 --n 1024 --k 512;--m 512 --n 4096 --k 1024;--m 4096 --n 512 --k 1024;--m 8192 --n 1024 --k 2048"
    [layer_norm]="--batch_size 32 --seq_len 512 --hidden_size 768;--batch_size 64 --seq_len 1024 --hidden_size 1024;--batch_size 128 --seq_len 2048 --hidden_size 2048;--batch_size 8 --seq_len 4096 --hidden_size 8192;--batch_size 1 --seq_len 8192 --hidden_size 16384;--batch_size 128 --seq_len 512 --hidden_size 768;--batch_size 64 --seq_len 2048 --hidden_size 1024;--batch_size 32 --seq_len 4096 --hidden_size 2048;--batch_size 16 --seq_len 1024 --hidden_size 8192;--batch_size 4 --seq_len 8192 --hidden_size 1024"
    [softmax]="--batch_size 32 --seq_len 512 --vocab_size 50257;--batch_size 64 --seq_len 1024 --vocab_size 50257;--batch_size 128 --seq_len 2048 --vocab_size 50257;--batch_size 8 --seq_len 4096 --vocab_size 50257;--batch_size 1 --seq_len 8192 --vocab_size 50257;--batch_size 128 --seq_len 512 --vocab_size 10000;--batch_size 64 --seq_len 2048 --vocab_size 25000;--batch_size 32 --seq_len 4096 --vocab_size 75000;--batch_size 16 --seq_len 1024 --vocab_size 100000;--batch_size 4 --seq_len 8192 --vocab_size 128000"
    [smith_waterman]="--seq1_len 512 --seq2_len 512;--seq1_len 1024 --seq2_len 1024;--seq1_len 2048 --seq2_len 2048;--seq1_len 4096 --seq2_len 4096;--seq1_len 8192 --seq2_len 8192;--seq1_len 1024 --seq2_len 512;--seq1_len 512 --seq2_len 1024;--seq1_len 2048 --seq2_len 4096;--seq1_len 4096 --seq2_len 2048;--seq1_len 1024 --seq2_len 8192"
    [stencil_2d]="--grid_size 512;--grid_size 1024;--grid_size 2048;--grid_size 4096;--grid_size 8192;--grid_size 1536;--grid_size 2560;--grid_size 3072;--grid_size 6144;--grid_size 10240"
)

for kernel_name in "${!KERNELS[@]}"; do
    runner_script="${kernel_name}_runner.py"
    problem_sizes_str="${KERNELS[$kernel_name]}"
    
    IFS=';' read -r -a problem_sizes <<< "$problem_sizes_str"

    for version in "swizzle" "unswizzle"; do
        for i in "${!problem_sizes[@]}"; do
            problem_args="${problem_sizes[$i]}"
            echo "Running $kernel_name ($version) with problem size: $problem_args"
            
            # Construct the full command
            full_command="./profile.sh $runner_script --version $version --iteration $i $problem_args"
            
            echo "Executing: $full_command"
            $full_command
            
            if [ $? -ne 0 ]; then
                echo "Error running profiling for $kernel_name ($version) with size $problem_args. Aborting."
                exit 1
            fi
        done
    done
done

echo "All profiling runs completed." 