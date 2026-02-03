# 1D Convolution

Causal 1D convolution kernel with tiled shared memory.

## Build

```bash
make
```

## Run

```bash
./conv1d_bench
```

## Optimize with IntelliPerf

```bash
intelliperf "Optimize the 1D convolution kernel" --target-speedup 1.5
```
