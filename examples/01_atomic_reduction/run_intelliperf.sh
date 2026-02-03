#!/bin/bash

intelliperf "Optimize the atomic_reduce_kernel kernel" \
  --target-speedup 1.4 \
  --verbose 2>&1 | tee intelliperf_output.txt
