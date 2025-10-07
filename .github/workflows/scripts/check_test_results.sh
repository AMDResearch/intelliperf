#!/bin/bash
set -e

# Check test results for success
echo 'Checking test results...'
results_dir=~/intelliperf_results

# Function to check test result
check_test_result() {
  local file="$1"
  local test_name="$2"
  if [ -f "$file" ]; then
    if jq -e '.success == true' "$file" >/dev/null 2>&1; then
      echo "✅ $test_name: PASSED"
      return 0
    else
      echo "❌ $test_name: FAILED"
      return 1
    fi
  else
    echo "❌ $test_name: FAILED (file not found)"
    return 1
  fi
}

# Track overall success
overall_success=true

check_test_result "$results_dir/memory_access_output.json" "Memory Access" || overall_success=false
check_test_result "$results_dir/bank_conflict_output.json" "Bank Conflict" || overall_success=false
check_test_result "$results_dir/atomic_contention_output.json" "Atomic Contention" || overall_success=false
check_test_result "$results_dir/swizzling_output.json" "Swizzling" || overall_success=false
check_test_result "$results_dir/diagnose_only_hip_uncoalesced.json" "Diagnose Only (HIP)" || overall_success=false
check_test_result "$results_dir/diagnose_only_torch_add.json" "Diagnose Only (Torch)" || overall_success=false
check_test_result "$results_dir/diagnose_only_triton_reduce.json" "Diagnose Only (Triton)" || overall_success=false

echo ""
if [ "$overall_success" = true ]; then
  echo "🎯 All IntelliPerf tests PASSED! ✅"
else
  echo "⚠️ Some IntelliPerf tests FAILED! ❌"
fi
