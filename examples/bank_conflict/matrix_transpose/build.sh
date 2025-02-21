#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1


INST_BASE=/opt/logduration
INCLUDE="-I ${INST_BASE}/include/kerneldb -I ${INST_BASE}/include/dh_comms -I /opt/rocm/include/hsa"
NO_WARN=-Wno-unused-result


INST_LIB=${INST_BASE}/lib/libAMDGCNMemTraceHip.so

hipcc ${NO_WARN} ${INCLUDE} -fgpu-rdc -fpass-plugin=${INST_LIB} -O3 -g -o matrix_transpose matrix_transpose.hip