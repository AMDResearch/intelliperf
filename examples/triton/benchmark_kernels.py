import os
import re
import subprocess
import csv
import tempfile
import argparse

# Define the directories to scan for kernels and logs
BASE_DIRS = [
    "autogen_10",
    "autogen_science_10"
]

# Define 10 problem sizes for each kernel type.
# To add a new kernel, add its name and a list of 10 parameter dictionaries.
PROBLEM_SIZES = {
    'conv2d': [
        {'N': 32, 'C': 32, 'H': 32, 'W': 32, 'F': 64, 'R': 3, 'S': 3},
        {'N': 64, 'C': 32, 'H': 32, 'W': 32, 'F': 64, 'R': 3, 'S': 3},
        {'N': 128, 'C': 32, 'H': 32, 'W': 32, 'F': 64, 'R': 3, 'S': 3},
        {'N': 32, 'C': 32, 'H': 64, 'W': 64, 'F': 64, 'R': 3, 'S': 3},
        {'N': 32, 'C': 32, 'H': 128, 'W': 128, 'F': 64, 'R': 3, 'S': 3},
        {'N': 32, 'C': 64, 'H': 32, 'W': 32, 'F': 128, 'R': 3, 'S': 3},
        {'N': 32, 'C': 128, 'H': 32, 'W': 32, 'F': 256, 'R': 3, 'S': 3},
        {'N': 64, 'C': 64, 'H': 64, 'W': 64, 'F': 128, 'R': 3, 'S': 3},
        {'N': 128, 'C': 128, 'H': 64, 'W': 64, 'F': 256, 'R': 3, 'S': 3},
        {'N': 256, 'C': 64, 'H': 32, 'W': 32, 'F': 512, 'R': 3, 'S': 3},
    ],
    'spmv': [
        {'M': 8192, 'N': 8192, 'density': 0.01},
        {'M': 4096, 'N': 4096, 'density': 0.01},
        {'M': 16384, 'N': 16384, 'density': 0.01},
        {'M': 32768, 'N': 32768, 'density': 0.01},
        {'M': 8192, 'N': 8192, 'density': 0.001},
        {'M': 8192, 'N': 8192, 'density': 0.05},
        {'M': 16384, 'N': 16384, 'density': 0.005},
        {'M': 16384, 'N': 16384, 'density': 0.02},
        {'M': 4096, 'N': 16384, 'density': 0.01},
        {'M': 16384, 'N': 4096, 'density': 0.01},
    ],
    'fft': [
        {'N': 2**12}, {'N': 2**13}, {'N': 2**14}, {'N': 2**15},
        {'N': 2**16}, {'N': 2**17}, {'N': 2**18}, {'N': 2**19},
        {'N': 2**20}, {'N': 2**21}
    ],
    'fused_attention': [
        {'Z': 16, 'H': 24, 'N_CTX': 2048, 'D_HEAD': 64},
        {'Z': 32, 'H': 48, 'N_CTX': 4096, 'D_HEAD': 64},
        {'Z': 64, 'H': 96, 'N_CTX': 1024, 'D_HEAD': 64},
        {'Z': 32, 'H': 48, 'N_CTX': 4096, 'D_HEAD': 128},
        {'Z': 16, 'H': 24, 'N_CTX': 8192, 'D_HEAD': 64},
        {'Z': 32, 'H': 48, 'N_CTX': 2048, 'D_HEAD': 128},
        {'Z': 8, 'H': 12, 'N_CTX': 16384, 'D_HEAD': 64},
        {'Z': 16, 'H': 48, 'N_CTX': 4096, 'D_HEAD': 64},
        {'Z': 32, 'H': 24, 'N_CTX': 4096, 'D_HEAD': 128},
        {'Z': 4, 'H': 96, 'N_CTX': 8192, 'D_HEAD': 32},
    ],
    'fused_elementwise': [
        {'size': 2**16}, {'size': 2**17}, {'size': 2**18}, {'size': 2**19},
        {'size': 2**20}, {'size': 2**21}, {'size': 2**22}, {'size': 2**23},
        {'size': 2**24}, {'size': 2**25}
    ],
    'gemm': [
        {'M': 4096, 'N': 4096, 'K': 4096},
        {'M': 8192, 'N': 8192, 'K': 8192},
        {'M': 16384, 'N': 16384, 'K': 16384},
        {'M': 8192, 'N': 4096, 'K': 8192},
        {'M': 4096, 'N': 8192, 'K': 4096},
        {'M': 8192, 'N': 8192, 'K': 4096},
        {'M': 16384, 'N': 8192, 'K': 16384},
        {'M': 4096, 'N': 16384, 'K': 8192},
        {'M': 2048, 'N': 2048, 'K': 2048},
        {'M': 1024, 'N': 1024, 'K': 1024},
    ],
    'layer_norm': [
        {'M': 4096, 'N': 4096}, {'M': 8192, 'N': 8192},
        {'M': 16384, 'N': 16384}, {'M': 32768, 'N': 32768},
        {'M': 8192, 'N': 4096}, {'M': 4096, 'N': 8192},
        {'M': 16384, 'N': 8192}, {'M': 8192, 'N': 16384},
        {'M': 2048, 'N': 2048}, {'M': 1024, 'N': 1024},
    ],
    'softmax': [
        {'n_rows': 4096, 'n_cols': 4096}, {'n_rows': 8192, 'n_cols': 8192},
        {'n_rows': 16384, 'n_cols': 16384}, {'n_rows': 32768, 'n_cols': 32768},
        {'n_rows': 8192, 'n_cols': 4096}, {'n_rows': 4096, 'n_cols': 8192},
        {'n_rows': 16384, 'n_cols': 8192}, {'n_rows': 8192, 'n_cols': 16384},
        {'n_rows': 2048, 'n_cols': 2048}, {'n_rows': 1024, 'n_cols': 1024},
    ],
    'stencil_2d': [
        {'M': 1024, 'N': 1024}, {'M': 2048, 'N': 2048},
        {'M': 4096, 'N': 4096}, {'M': 8192, 'N': 8192},
        {'M': 16384, 'N': 16384}, {'M': 8192, 'N': 4096},
        {'M': 4096, 'N': 8192}, {'M': 16384, 'N': 8192},
        {'M': 8192, 'N': 16384}, {'M': 512, 'N': 512},
    ],
    'transpose': [
        {'M': 4096, 'N': 4096}, {'M': 8192, 'N': 8192},
        {'M': 16384, 'N': 16384}, {'M': 32768, 'N': 32768},
        {'M': 8192, 'N': 4096}, {'M': 4096, 'N': 8192},
        {'M': 16384, 'N': 8192}, {'M': 8192, 'N': 16384},
        {'M': 2048, 'N': 2048}, {'M': 1024, 'N': 1024},
    ]
}

def parse_optimized_code(log_file_path):
    """Parses the final log file to extract the full kernel code."""
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        # The code is in the second '[[[' block, following "Full Kernel Code:"
        parts = content.split('[[[')
        if len(parts) > 2:
            code = parts[2].split(']]]')[0]
            return code.strip()
        else:
            print(f"Warning: Could not find second '[[[' block in {log_file_path}.")
            return None
    except FileNotFoundError:
        print(f"Error: Log file not found at {log_file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while parsing {log_file_path}: {e}")
        return None

def run_benchmark(script_path, params):
    """Runs a kernel script with given parameters and returns the execution time in ms."""
    if not os.path.exists(script_path):
        print(f"Warning: Script not found at {script_path}, skipping benchmark.")
        return "not_found", False
        
    cmd = ['python', script_path]
    for key, value in params.items():
        cmd.extend([f'--{key}', str(value)])
    cmd.append('--validate')
    try:
        # Using a timeout for safety, in case a kernel hangs
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)
        output = result.stdout
        success = result.returncode == 0
        
        # Example output: "Triton conv2d time: 1.2345 ms"
        match = re.search(r"time: ([\d\.]+) ms", output)
        if match:
            return float(match.group(1)), success
        else:
            print(f"Warning: Could not parse runtime from output of {script_path} with params {params}")
            print("STDOUT:", output)
            return "parse_error", success
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path} with params {params}:")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return "runtime_error", success
    except subprocess.TimeoutExpired:
        print(f"Timeout running {script_path} with params {params}")
        return "timeout", success

def main():
    parser = argparse.ArgumentParser(description="Benchmark optimized and unoptimized Triton kernels.")
    parser.add_argument("--kernels", nargs='+', help="Specific kernels to run (e.g., conv2d spmv). Runs all if not specified.")
    args = parser.parse_args()

    for base_dir in BASE_DIRS:
        swizzling_logs_dir = os.path.join(base_dir, 'swizzling_logs_2')
        if not os.path.isdir(swizzling_logs_dir):
            continue

        for log_filename in sorted(os.listdir(swizzling_logs_dir)):
            if log_filename.endswith('_log_final.txt'):
                kernel_name = log_filename.replace('_log_final.txt', '')

                if args.kernels and kernel_name not in args.kernels:
                    continue

                print(f"--- Processing kernel: {kernel_name} in {base_dir} ---")

                if kernel_name not in PROBLEM_SIZES:
                    print(f"Warning: No problem sizes defined for kernel '{kernel_name}'. Skipping.")
                    continue

                log_file_path = os.path.join(swizzling_logs_dir, log_filename)
                
                # 1. Parse optimized code and write to a permanent file
                optimized_code = parse_optimized_code(log_file_path)
                if not optimized_code:
                    continue

                optimized_script_path = os.path.join(swizzling_logs_dir, f"{kernel_name}_optimized.py")
                with open(optimized_script_path, 'w') as f:
                    f.write(optimized_code)

                # 2. Paths for unoptimized script and output CSV
                unoptimized_script_path = os.path.join(base_dir, f"{kernel_name}.py")
                csv_path = os.path.join(swizzling_logs_dir, f"{kernel_name}_performance.csv")
                
                problem_params = PROBLEM_SIZES[kernel_name]
                param_names = list(problem_params[0].keys())
                
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(param_names + ['unoptimized_ms', 'optimized_ms', 'unoptimized_success', 'optimized_success'])

                    # 3. Run benchmarks for all problem sizes
                    for i, params in enumerate(problem_params):
                        print(f"  Running problem size {i+1}/{len(problem_params)}...")
                        
                        unoptimized_time, unoptimized_success = run_benchmark(unoptimized_script_path, params)
                        optimized_time, optimized_success = run_benchmark(optimized_script_path, params)
                        
                        row = [params[k] for k in param_names] + [unoptimized_time, optimized_time, unoptimized_success, optimized_success]
                        writer.writerow(row)
                
                print(f"Results for {kernel_name} written to {csv_path}")
                print(f"Optimized kernel code saved to {optimized_script_path}")
                
                # The optimized script is now saved permanently.
                print(f"--- Finished {kernel_name} ---")

if __name__ == "__main__":
    main() 