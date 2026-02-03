# Grouped GEMM

Grouped matrix multiplication benchmark.

## Build

```bash
make
```

## Run

```bash
./grouped_gemm_bench
```

## Optimize with IntelliPerf

```bash
intelliperf "Optimize the grouped GEMM kernel" --target-speedup 1.5
```
