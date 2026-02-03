#!/bin/bash

intelliperf "Optimize the grouped_gemm_kernel kernel" \
  --target-speedup 1.4 \
  --verbose 2>&1 | tee intelliperf_output.txt
