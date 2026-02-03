# IntelliPerf

AI-powered GPU kernel optimizer for AMD GPUs.

## Installation

```bash
pip install .
```

This uses build isolation by default.
For development (editable):

```bash
python -m pip install -e . --no-build-isolation
```

## Quick Start

```bash
# Navigate to an example
cd examples/01_atomic_reduction

# Run optimization
export LLM_GATEWAY_KEY="your-key-here"
intelliperf "Optimize the atomic reduction kernel" --target-speedup 2.0
```

## Examples

- `01_atomic_reduction/` - Atomic reduction kernel
- `02_conv1d/` - 1D convolution kernel
- `03_grouped_gemm/` - Grouped GEMM kernel
- `04_scan/` - Parallel scan kernel

## Requirements

- Python 3.9+
- ROCm 7.0+
- AMD GPU (gfx90a, gfx942, gfx1201)
- IntelliKit (installed automatically via pip)
