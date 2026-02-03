# Parallel Scan

Parallel prefix sum (scan) benchmark.

## Build

```bash
make
```

## Run

```bash
./scan_bench
```

## Optimize with IntelliPerf

```bash
intelliperf "Optimize the parallel scan kernel" --target-speedup 1.5
```
