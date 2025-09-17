# IntelliPerf: AI-Powered GPU Performance Engineering Framework

IntelliPerf is a Python-based framework that uses Large Language Models (LLMs) to automatically analyze and optimize GPU kernel performance. It supports HIP/ROCm, Triton, and PyTorch applications, targeting bottlenecks like bank conflicts, memory access patterns, and atomic contention.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Quick Start (Container Recommended)
Use containers for full functionality including GPU-dependent features:
```bash
# Using Docker (recommended)
./docker/build.sh
./docker/run.sh

# Using Apptainer
./apptainer/build.sh
./apptainer/run.sh
```

### Development Installation (Basic Python Functionality)
For Python-only development without GPU dependencies:
```bash
# Install the main package (takes ~90 seconds)
pip install -e .

# Verify installation
intelliperf --help
```

### Full Dependencies Installation (Network-Intensive)
**WARNING**: This step frequently fails due to network timeouts. NEVER CANCEL builds - they may take 45+ minutes.
```bash
# Install external tools - NEVER CANCEL: Can take 45+ minutes. Set timeout to 60+ minutes.
python3 scripts/install_tool.py --all

# If network timeouts occur, this is expected - document as "may fail due to network limitations"
```

### Examples Build (Requires ROCm/HIP)
```bash
# Build examples - requires ROCm/HIP environment
cd examples
./scripts/build_examples.sh -c

# Clean build if needed
./scripts/build_examples.sh -c --clean

# Verbose build for debugging
./scripts/build_examples.sh -c --verbose
```

## Core Development Commands

### Code Quality (Always Run Before Committing)
```bash
# Install linting tools
pip install ruff==0.3.0

# Check code style (fast, <1 second)
ruff check .

# Fix auto-fixable issues
ruff check . --fix

# Format code
ruff format .
```

### Pre-commit Hooks (May Fail Due to Network Issues)
```bash
pip install pre-commit==3.6.0
pre-commit install

# Run all hooks - NEVER CANCEL: Takes 2-5 minutes. Set timeout to 10+ minutes.
# NOTE: May fail due to network timeouts - this is expected in some environments
pre-commit run --all-files
```

### Testing
```bash
# Note: Most tests require GPU hardware and ROCm environment
# Basic test check (will fail without GPU libraries but shows test structure)
python -m pytest tests/ -v

# Shell-based integration tests (require built examples)
./tests/test_matrix_transpose.sh
```

## IntelliPerf Usage Patterns

### Diagnose Only (Works Without GPU Optimization)
```bash
# Diagnose HIP application
intelliperf --formula=diagnoseOnly -- ./examples/build/access_pattern/uncoalesced

# Diagnose PyTorch application  
intelliperf --formula=diagnoseOnly -- python ./examples/torch/add.py

# Diagnose Triton application
TRITON_DISABLE_LINE_INFO=0 intelliperf --formula=diagnoseOnly -- python ./examples/triton/reduce.py
```

### Full Optimization (Requires LLM API Key and GPU)
```bash
# Set required environment variable
export LLM_GATEWAY_KEY="your_api_key_here"

# Memory access optimization
intelliperf --project_directory=./examples \
  --build_command="./scripts/build_examples.sh -c" \
  --formula=memoryAccess -- ./build/access_pattern/uncoalesced

# Bank conflict optimization  
intelliperf --project_directory=./examples \
  --build_command="./scripts/build_examples.sh -c" \
  --formula=bankConflict -- ./build/bank_conflict/matrix_transpose 1024 1024

# Atomic contention optimization
intelliperf --project_directory=./examples \
  --build_command="./scripts/build_examples.sh -c" \
  --instrument_command="./scripts/build_examples.sh -i -c" \
  --formula=atomicContention -- ./build/contention/reduction
```

## Manual Validation Requirements

**CRITICAL**: After making any changes to IntelliPerf, ALWAYS run through these complete validation scenarios:

### 1. Memory Access Pattern Validation
```bash
# Test uncoalesced memory access detection and optimization
intelliperf --formula=memoryAccess --project_directory=./examples \
  --build_command="./scripts/build_examples.sh -c" \
  -- ./build/access_pattern/uncoalesced

# Verify: Should show memory coalescing improvements and performance gains
```

### 2. Bank Conflict Validation  
```bash
# Test shared memory bank conflict detection and optimization
intelliperf --formula=bankConflict --project_directory=./examples \
  --build_command="./scripts/build_examples.sh -c" \
  -- ./build/bank_conflict/matrix_transpose 1024 1024

# Verify: Should show bank conflict reduction and speedup
```

### 3. Atomic Contention Validation
```bash
# Test atomic operation contention detection and optimization  
intelliperf --formula=atomicContention --project_directory=./examples \
  --build_command="./scripts/build_examples.sh -c" \
  --instrument_command="./scripts/build_examples.sh -i -c" \
  -- ./build/contention/reduction

# Verify: Should show atomic contention reduction and performance improvement
```

### 4. Multi-Backend Diagnose Validation
```bash
# Test HIP application analysis
intelliperf --formula=diagnoseOnly -- ./examples/build/access_pattern/uncoalesced

# Test PyTorch application analysis  
intelliperf --formula=diagnoseOnly -- python ./examples/torch/add.py

# Test Triton application analysis
TRITON_DISABLE_LINE_INFO=0 intelliperf --formula=diagnoseOnly -- python ./examples/triton/reduce.py

# Verify: All should generate valid performance analysis JSON output
```

## Critical Timing and Timeout Information

### Build Commands - NEVER CANCEL
- **Python package install**: 90 seconds normal, set timeout to 3+ minutes
- **External tools install**: 45+ minutes normal, set timeout to 60+ minutes  
- **Examples build**: 5-10 minutes normal, set timeout to 15+ minutes
- **Pre-commit setup**: 2-5 minutes normal, set timeout to 10+ minutes
- **IntelliPerf optimization runs**: 10-30 minutes normal, set timeout to 45+ minutes

### Network Issues (Expected)
- External dependency installation frequently fails due to network timeouts
- Pre-commit hooks may fail to install due to PyPI timeouts
- Document these as "may fail due to network limitations" rather than fixing
- Use containers for reliable development environment

## Repository Structure

### Key Directories
```
src/intelliperf/     # Main Python package
src/accordo/         # Validation and correctness checking
examples/            # Test applications in HIP, Triton, PyTorch
  scripts/build_examples.sh  # Example build system
external/            # External dependencies (rocprofiler-compute, omniprobe, nexus)
tests/               # Integration tests (require GPU hardware)
.github/workflows/   # CI that runs on AMD GPU droplets
```

### Configuration Files
- `pyproject.toml` - Python dependencies and tool configuration
- `.pre-commit-config.yaml` - Code quality hooks
- `.github/workflows/ci.yml` - Full GPU-based testing pipeline
- `docker/` and `apptainer/` - Container definitions

## Environment Requirements

### Minimal (Python Development)
- Python 3.8+
- pip

### Full Functionality
- ROCm/HIP environment
- AMD GPU hardware (tested on MI300X)
- Network access for dependency installation
- LLM API key for optimization features

## Common Issues and Solutions

### "ROCm not found" Error
- Expected in non-GPU environments
- Use containers for full GPU functionality
- Python-only features still work (CLI, some validation)

### Network Timeout Errors
- Very common with `python3 scripts/install_tool.py --all`
- Expected with pre-commit installation
- Document as limitation rather than trying to fix
- Use containers which have dependencies pre-installed

### Test Failures Without GPU
- Expected - most tests require GPU hardware
- CI runs on actual AMD GPU droplets
- Focus on code quality checks for local development

### Performance Validation
- Always test at least one complete optimization scenario after changes
- Verify JSON output contains expected performance metrics
- Check that both correctness and performance validation pass

## CI Integration

The CI system (.github/workflows/ci.yml) runs comprehensive tests on AMD GPU hardware:
- Spins up GPU droplets with MI300X hardware  
- Installs full dependency chain
- Tests all optimization formulas
- Validates correctness and performance improvements
- NEVER CANCEL: CI can take 45+ minutes including droplet provisioning

Always ensure your changes pass both local code quality checks and will work in the GPU CI environment.

Always ensure your changes pass both local code quality checks and will work in the GPU CI environment.