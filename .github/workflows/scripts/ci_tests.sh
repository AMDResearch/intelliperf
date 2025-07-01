#!/bin/bash
set -e

# Run examples and store outputs
echo 'Running IntelliPerf examples...'

mkdir -p /intelliperf_results
# Formulas
intelliperf -vvv --project_directory=./examples --build_command="./scripts/build_examples.sh -c" --formula=memoryAccess -o /intelliperf_results/memory_access_output.json -- ./build/access_pattern/uncoalesced
intelliperf -vvv --project_directory=./examples --build_command="./scripts/build_examples.sh -c" --formula=bankConflict -o /intelliperf_results/bank_conflict_output.json -- ./build/bank_conflict/matrix_transpose 1024 1024
intelliperf -vvv --project_directory=./examples --build_command="./scripts/build_examples.sh -c" --instrument_command="./scripts/build_examples.sh -i -c" --formula=atomicContention -o /intelliperf_results/atomic_contention_output.json -- ./build/contention/reduction


# Diagnose Only
intelliperf -vvv --formula=diagnoseOnly -o /intelliperf_results/diagnose_only_hip_uncoalesced.json -- ./examples/build/access_pattern/uncoalesced
intelliperf -vvv --formula=diagnoseOnly -o /intelliperf_results/diagnose_only_torch_add.json -- python ./examples/torch/add.py
intelliperf -vvv --formula=diagnoseOnly -o /intelliperf_results/diagnose_only_triton_reduce.json -- python ./examples/triton/reduce.py

# Display output files
echo 'Memory Access Output:'
cat /intelliperf_results/memory_access_output.json
echo 'Bank Conflict Output:'
cat /intelliperf_results/bank_conflict_output.json
echo 'Atomic Contention Output:'
cat /intelliperf_results/atomic_contention_output.json
echo 'Diagnose Only Output:'
cat /intelliperf_results/diagnose_only_hip_uncoalesced.json
cat /intelliperf_results/diagnose_only_torch_add.json
cat /intelliperf_results/diagnose_only_triton_reduce.json

# Success checking disabled for now
echo 'Success checking disabled - just displaying outputs' 
