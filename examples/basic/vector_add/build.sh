#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


INST_BASE=/opt/logduration
INCLUDE="-I ${INST_BASE}/include/kerneldb -I ${INST_BASE}/include/dh_comms -I /opt/rocm/include/hsa"
NO_WARN=-Wno-unused-result


INST_LIB=${INST_BASE}/lib/libAMDGCNSubmitAddressMessages-rocm.so

pushd $SCRIPT_DIR

hipcc ${NO_WARN} ${INCLUDE} -fgpu-rdc -fpass-plugin=${INST_LIB} -O3 -g -o vector_add vector_add.hip

popd