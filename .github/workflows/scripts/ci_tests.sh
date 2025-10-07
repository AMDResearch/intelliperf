#!/bin/bash
set -e

# Run examples and store outputs
echo 'Running IntelliPerf examples...'

results_dir=~/intelliperf_results
rm -rf $results_dir
mkdir -p $results_dir


provider="openrouter"
model="openai/gpt-4o"

# Formulas
intelliperf -vvv --project_directory=./examples --provider $provider --model $model --build_command="./scripts/build_examples.sh -c" --formula=memoryAccess -o $results_dir/memory_access_output.json -- ./build/access_pattern/uncoalesced || true
intelliperf -vvv --project_directory=./examples --provider $provider --model $model --build_command="./scripts/build_examples.sh -c" --formula=bankConflict -o $results_dir/bank_conflict_output.json -- ./build/bank_conflict/matrix_transpose 1024 1024 || true
intelliperf -vvv --project_directory=./examples --provider $provider --model $model --build_command="./scripts/build_examples.sh -c" --instrument_command="./scripts/build_examples.sh -i -c" --formula=atomicContention -o $results_dir/atomic_contention_output.json -- ./build/contention/reduction || true
intelliperf -vvv --project_directory=./examples --provider $provider --model $model --formula=swizzling --project_directory="./examples" --unittest_command="triton/gemm_runner.py --validate" -o $results_dir/swizzling_output.json -- ./triton/gemm_runner.py || true
# Diagnose Only
intelliperf -vvv --formula=diagnoseOnly -o $results_dir/diagnose_only_hip_uncoalesced.json -- ./examples/build/access_pattern/uncoalesced
intelliperf -vvv --formula=diagnoseOnly -o $results_dir/diagnose_only_torch_add.json -- ./examples/torch/add.py
TRITON_DISABLE_LINE_INFO=0 intelliperf -vvv --formula=diagnoseOnly -o $results_dir/diagnose_only_triton_reduce.json -- ./examples/triton/reduce.py

# Display output files
echo 'Memory Access Output:'
cat $results_dir/memory_access_output.json || echo "File not found"
echo 'Bank Conflict Output:'
cat $results_dir/bank_conflict_output.json || echo "File not found"
echo 'Atomic Contention Output:'
cat $results_dir/atomic_contention_output.json || echo "File not found"
echo 'Diagnose Only Output:'
cat $results_dir/diagnose_only_hip_uncoalesced.json || echo "File not found"
cat $results_dir/diagnose_only_torch_add.json || echo "File not found"
cat $results_dir/diagnose_only_triton_reduce.json || echo "File not found" 
