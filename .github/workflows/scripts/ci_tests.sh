#!/bin/bash
set -e

# Run examples and store outputs
echo 'Running IntelliPerf examples...'

# Memory Access example
intelliperf -vvv --project_directory=./examples --build_command="./scripts/build_examples.sh -c" --formula=memoryAccess -o /tmp/memory_access_output.json -- ./build/access_pattern/uncoalesced

return 0
# Bank Conflict example
intelliperf -vvv --project_directory=./examples --build_command="./scripts/build_examples.sh -c" --formula=bankConflict -o /tmp/bank_conflict_output.json -- ./build/bank_conflict/matrix_transpose 1024 1024

# Diagnose Only example
intelliperf -vvv --formula=diagnoseOnly -o /tmp/diagnose_only_output.json -- ./build/access_pattern/uncoalesced

# Display output files
echo 'Memory Access Output:'
cat /tmp/memory_access_output.json
echo 'Bank Conflict Output:'
cat /tmp/bank_conflict_output.json
echo 'Diagnose Only Output:'
cat /tmp/diagnose_only_output.json

# Success checking disabled for now
echo 'Success checking disabled - just displaying outputs' 
