# Atomic Reduction

Simple atomic reduction benchmark for array summation.

## Build

```bash
cmake -B build && cmake --build build
```

## Run

```bash
./atomic_reduction_bench
```

## Optimize with IntelliPerf

```bash
intelliperf "Optimize the atomic reduction kernel" --target-speedup 2.0
```
