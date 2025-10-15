#!/bin/bash
set -e

#===============================================================================
# IntelliPerf CI Test Suite
# This script runs various IntelliPerf examples and logs their output
#===============================================================================

echo '================================'
echo 'Running IntelliPerf examples...'
echo '================================'

# Setup results directory
results_dir=~/intelliperf_results
rm -rf "$results_dir"
mkdir -p "$results_dir"

# Configuration
provider="openrouter"
model="openai/gpt-4o"

#===============================================================================
# Formula-based Tests
#===============================================================================

echo ""
echo "[1/7] Running Memory Access formula test..."
intelliperf -vvv \
    --project_directory=./examples \
    --provider "$provider" \
    --model "$model" \
    --build_command="./scripts/build_examples.sh -c" \
    --formula=memoryAccess \
    -o "$results_dir/memory_access_output.json" \
    --trace_path "$results_dir/memory_access" \
    -- ./build/access_pattern/uncoalesced 2>&1 | tee "$results_dir/memory_access_output.log" || true

echo ""
echo "[2/7] Running Bank Conflict formula test..."
intelliperf -vvv \
    --project_directory=./examples \
    --provider "$provider" \
    --model "$model" \
    --build_command="./scripts/build_examples.sh -c" \
    --formula=bankConflict \
    -o "$results_dir/bank_conflict_output.json" \
    --trace_path "$results_dir/bank_conflict" \
    -- ./build/bank_conflict/matrix_transpose 1024 1024 2>&1 | tee "$results_dir/bank_conflict_output.log" || true

echo ""
echo "[3/7] Running Atomic Contention formula test..."
intelliperf -vvv \
    --project_directory=./examples \
    --provider "$provider" \
    --model "$model" \
    --build_command="./scripts/build_examples.sh -c" \
    --instrument_command="./scripts/build_examples.sh -i -c" \
    --formula=atomicContention \
    -o "$results_dir/atomic_contention_output.json" \
    --trace_path "$results_dir/atomic_contention" \
    -- ./build/contention/reduction 2>&1 | tee "$results_dir/atomic_contention_output.log" || true

echo ""
echo "[4/7] Running Swizzling formula test..."
intelliperf -vvv \
    --project_directory=./examples \
    --provider "$provider" \
    --model "$model" \
    --formula=swizzling \
    --unittest_command="triton/gemm_runner.py --validate" \
    -o "$results_dir/swizzling_output.json" \
    -- ./triton/gemm_runner.py 2>&1 | tee "$results_dir/swizzling_output.log" || true

#===============================================================================
# Diagnose-Only Tests
#===============================================================================

echo ""
echo "[5/7] Running Diagnose Only test (HIP uncoalesced)..."
intelliperf -vvv \
    --formula=diagnoseOnly \
    -o "$results_dir/diagnose_only_hip_uncoalesced.json" \
    --trace_path "$results_dir/diagnose_only_hip_uncoalesced" \
    -- ./examples/build/access_pattern/uncoalesced 2>&1 | tee "$results_dir/diagnose_only_hip_uncoalesced.log"

echo ""
echo "[6/7] Running Diagnose Only test (Torch add)..."
intelliperf -vvv \
    --formula=diagnoseOnly \
    -o "$results_dir/diagnose_only_torch_add.json" \
    --trace_path "$results_dir/diagnose_only_torch_add" \
    -- ./examples/torch/add.py 2>&1 | tee "$results_dir/diagnose_only_torch_add.log"

echo ""
echo "[7/7] Running Diagnose Only test (Triton reduce)..."
TRITON_DISABLE_LINE_INFO=0 intelliperf -vvv \
    --formula=diagnoseOnly \
    -o "$results_dir/diagnose_only_triton_reduce.json" \
    --trace_path "$results_dir/diagnose_only_triton_reduce" \
    -- ./examples/triton/reduce.py 2>&1 | tee "$results_dir/diagnose_only_triton_reduce.log"

#===============================================================================
# Display Results Summary
#===============================================================================

echo ""
echo '======================================='
echo 'Test Results Summary'
echo '======================================='

echo ""
echo '[Memory Access Output]'
cat "$results_dir/memory_access_output.json" 2>/dev/null || echo "  ⚠ File not found"

echo ""
echo '[Bank Conflict Output]'
cat "$results_dir/bank_conflict_output.json" 2>/dev/null || echo "  ⚠ File not found"

echo ""
echo '[Atomic Contention Output]'
cat "$results_dir/atomic_contention_output.json" 2>/dev/null || echo "  ⚠ File not found"

echo ""
echo '[Swizzling Output]'
cat "$results_dir/swizzling_output.json" 2>/dev/null || echo "  ⚠ File not found"

echo ""
echo '[Diagnose Only - HIP Uncoalesced]'
cat "$results_dir/diagnose_only_hip_uncoalesced.json" 2>/dev/null || echo "  ⚠ File not found"

echo ""
echo '[Diagnose Only - Torch Add]'
cat "$results_dir/diagnose_only_torch_add.json" 2>/dev/null || echo "  ⚠ File not found"

echo ""
echo '[Diagnose Only - Triton Reduce]'
cat "$results_dir/diagnose_only_triton_reduce.json" 2>/dev/null || echo "  ⚠ File not found"

echo ""
echo '======================================='
echo 'All logs saved to:' "$results_dir/*.log"
echo '======================================='
