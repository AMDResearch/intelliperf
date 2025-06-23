<!--
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
-->

# Contributing to IntelliPerf

Thank you for your interest in contributing to IntelliPerf! This document provides guidelines and instructions for contributing to the project.

## Development Setup

### Prerequisites

We provide both Apptainer and Docker images containing all the dependencies. To get started, run:

#### Using Apptainer

```bash
./apptainer/build.sh
./apptainer/run.sh
```

#### Using Docker
```bash
./docker/build.sh
./docker/run.sh
```

### Setting Up Development Environment

1. **Clone the Repository**:
   ```bash
   git clone git@github.com:AMDResearch/intelliperf.git
   cd intelliperf
   ```

2. **Install Development Dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Install Pre-commit Hooks**:
   ```bash
   pre-commit install
   ```

4. **Run Code Quality Checks**:
   ```bash
   pre-commit run --all-files
   ```

## Development Workflow

1. **Create a New Branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Commit Your Changes**:
   ```bash
   git commit -m "Description of your changes"
   ```

3. **Push Your Changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Create a Pull Request**:
   - Go to the GitHub repository
   - Click "New Pull Request"
   - Select your branch
   - Fill in the PR template
   - Submit the PR

## Code Style Guidelines

Code style guidelines are enforced via pre-commit hooks using Ruff. To manually check your code style:

```bash
pre-commit run --all-files
```

## License

By contributing to IntelliPerf, you agree that your contributions will be licensed under the project's MIT License. 