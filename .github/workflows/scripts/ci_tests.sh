#!/bin/bash
set -e

# Run examples and store outputs
echo 'Running IntelliPerf examples...'

mkdir -p /intelliperf_results

provider="openrouter"
model="openai/gpt-4o"

# Formulas
intelliperf -vvv --project_directory=./examples --provider $provider --model $model --build_command="./scripts/build_examples.sh -c" --formula=memoryAccess -o /intelliperf_results/memory_access_output.json -- ./build/access_pattern/uncoalesced || true
intelliperf -vvv --project_directory=./examples --provider $provider --model $model --build_command="./scripts/build_examples.sh -c" --formula=bankConflict -o /intelliperf_results/bank_conflict_output.json -- ./build/bank_conflict/matrix_transpose 1024 1024 || true
intelliperf -vvv --project_directory=./examples --provider $provider --model $model --build_command="./scripts/build_examples.sh -c" --instrument_command="./scripts/build_examples.sh -i -c" --formula=atomicContention -o /intelliperf_results/atomic_contention_output.json -- ./build/contention/reduction || true

# Diagnose Only
intelliperf -vvv --provider $provider --model $model --formula=diagnoseOnly -o /intelliperf_results/diagnose_only_hip_uncoalesced.json -- ./examples/build/access_pattern/uncoalesced
intelliperf -vvv --provider $provider --model $model --formula=diagnoseOnly -o /intelliperf_results/diagnose_only_torch_add.json -- python ./examples/torch/add.py
TRITON_DISABLE_LINE_INFO=0 intelliperf -vvv --provider $provider --model $model --formula=diagnoseOnly -o /intelliperf_results/diagnose_only_triton_reduce.json -- python ./examples/triton/reduce.py

# Display output files
echo 'Memory Access Output:'
cat /intelliperf_results/memory_access_output.json || echo "File not found"
echo 'Bank Conflict Output:'
cat /intelliperf_results/bank_conflict_output.json || echo "File not found"
echo 'Atomic Contention Output:'
cat /intelliperf_results/atomic_contention_output.json || echo "File not found"
echo 'Diagnose Only Output:'
cat /intelliperf_results/diagnose_only_hip_uncoalesced.json || echo "File not found"
cat /intelliperf_results/diagnose_only_torch_add.json || echo "File not found"
cat /intelliperf_results/diagnose_only_triton_reduce.json || echo "File not found" 
