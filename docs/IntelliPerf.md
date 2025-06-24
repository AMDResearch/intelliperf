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

# IntelliPerf: An Automated Framework for GPU Performance Engineering

## Abstract

*Manual optimization of GPU kernels is a complex, time-consuming, and error-prone process that requires deep domain expertise, which remains a scarce resource. This article introduces IntelliPerf, an automated performance engineering framework developed by AMD Research and Advanced Development (RAD) that systematizes and automates this workflow. IntelliPerf orchestrates a suite of profiling, instrumentation, and analysis tools (Figure 1) to automatically pinpoint performance bottlenecks, generate optimized code using Large Language Models (LLMs), and validate the results for both correctness and performance. By integrating seamlessly into CI/CD pipelines, IntelliPerf enables continuous, automated performance improvement, effectively acting as an AI performance engineer. This work demonstrates a novel approach to combining program analysis with generative AI to address the challenges of GPU software optimization.*

<p align="center">
  <img src="assets/intelliperf_tools.png" alt="IntelliPerf Tool Ecosystem"><br>
  <i>Figure 1: The IntelliPerf toolchain, orchestrating a suite of existing and novel AMD tools for automated performance engineering.</i>
</p>

## Introduction

The performance of High-Performance Computing (HPC) and Machine Learning (ML) applications is increasingly dominated by the efficiency of their GPU kernels. However, optimizing these kernels presents a significant challenge. The process requires a deep understanding of the underlying hardware architecture, mastery of various low-level profiling tools, and the ability to manually rewrite kernel source to resolve subtle bottlenecks such as memory access inefficiencies or contention issues. This expertise is rare, and the manual tuning cycle represents a major bottleneck in software development, consuming significant engineering resources and delaying time-to-solution.

Existing tools often address only isolated parts of this problem—a profiler may reveal a bottleneck but offers no path to resolution, while a static analyzer may flag potential issues without contextual performance data. This leaves a critical gap: the "last mile" of interpreting the data, forming a hypothesis, rewriting the code, and validating the change remains a purely manual effort.

This article presents IntelliPerf, the _maestro_ that automates this entire end-to-end workflow. Inspired by the process of expert human engineers, it profiles the code, diagnoses the bottleneck, rewrites the kernel, and validates the fix. By orchestrating a toolchain of advanced program analysis and generative AI technologies, IntelliPerf significantly reduces the time and expertise required to optimize GPU code.

## How It Works: The IntelliPerf Workflow

IntelliPerf is designed as a modular, extensible system that can be configured to target specific, well-known performance bottlenecks.

### The Formula-Driven Architecture

At its core, IntelliPerf is built on a "formula-driven" architecture. Each common GPU performance issue is abstracted into a **formula**. Users can select a specific bottleneck to target, such as `bankConflict`, `atomicContention`, or `memoryAccess`, through a configuration option.

This modular design is implemented through an object-oriented approach where a **base formula** defines a universal, multi-stage optimization workflow. **Specialized formulas** then inherit from this base and implement the specific logic for a particular bottleneck, such as the exact performance counters to query and the precise prompts to send to the LLM.

### The Multi-Stage Optimization Loop

IntelliPerf executes a closed-loop process that systematically moves from high-level profiling to a validated code change, as illustrated in Figure 2.

<p align="center">
  <img src="assets/intelliperf_loop.png" alt="IntelliPerf Loop"><br>
  <i>Figure 2: The multi-stage optimization loop executed by IntelliPerf.</i>
</p>

1.  **Profiling (`rocprofv3`)**: The process begins with a timing run to identify the most time-consuming kernel in the application (ranked by wallclock time). A second, more detailed run then collects a rich set of hardware performance counters relevant to the chosen formula. IntelliPerf considers the top N kernels but optimizes the first one that exhibits the specific bottleneck being targeted.

2.  **Analysis (`Guided Tuning` & `Omniprobe`)**: The performance counters are first analyzed by **`Guided Tuning`**, a tool that summarizes the data to identify the likely issue. **`Omniprobe`** then uses compiler-based instrumentation to pinpoint the specific source code line responsible for the bottleneck.

3.  **Code Generation (`Omniwiser`)**: Armed with this data, **`Omniwiser`**, a novel component built for IntelliPerf, crafts a detailed, context-aware prompt for existing public LLMs (such as OpenAI's GPTs, xAI, Claude, or others) to generate an optimized version of the code.

4.  **Validation (`Accordo`)**: The LLM-generated code is then validated for both correctness and performance by **`Accordo`**, another novel tool developed for the IntelliPerf project. Accordo operates as an Heterogenous System Architecture (HSA) Tools Library that intercepts kernel execution, captures output buffers through HIP IPC mechanisms, and performs automated side-by-side comparison with the reference implementation using user-defined tolerances.

## Key Technologies

IntelliPerf's success relies on the tight integration of several key technologies.

### Iterative AI-Powered Optimization

The interaction with the LLM is not a simple one-shot request. IntelliPerf employs a sophisticated, iterative feedback loop managed by a wrapper around the **DSPy** library. If an LLM-generated optimization is incorrect or not performant, IntelliPerf analyzes the failure and re-prompts the LLM with corrective feedback (e.g., "The previous attempt failed a correctness check..."). This cycle continues until a validated solution is found.

### Accordo: Automated Runtime Validation

Correctness validation is handled by Accordo, a specialized HSA Tools Library that performs automated side-by-side comparison of kernel outputs without requiring any application code changes. The validation process works through a sophisticated inter-process communication (IPC) mechanism:

**Setup and Execution**: IntelliPerf launches the unoptimized application with Accordo configured as the HSA Tools Library via environment variables. Accordo then intercepts all memory allocations, kernel dispatches, and other necessary ROCm runtime APIs. IntelliPerf communicates the target kernel identifier, argument layout (order and types), and communication pipes to Accordo.

**Kernel Interception**: When Accordo detects the target kernel dispatch, it executes the kernel and waits for completion. After kernel execution, Accordo exports IPC handles using HIP's memory export mechanism to the parent IntelliPerf process, enabling cross-process memory access.

**Memory Comparison**: The IntelliPerf process copies memory from kernel argument pointers for both the reference and optimized implementations. Accordo performs a side-by-side comparison of all non-const pointer arguments (output buffers) using a user-defined tolerance to handle floating-point arithmetic variations. This targeted approach ensures validation focuses only on the kernel's actual outputs while maintaining minimal performance overhead.

**Error Handling**: The entire validation process operates within IntelliPerf's optimization loop with timeout mechanisms. Any runtime errors from LLM-generated code are treated as optimization failures, triggering the iterative feedback loop to generate improved solutions.

**Performance Considerations**: Accordo is only enabled during correctness validation phases, not during profiling runs, and specifically targets only non-const pointer arguments to minimize overhead while ensuring comprehensive validation coverage.

<p align="center">
  <img src="assets/accordo.png" alt="Accordo Validation"><br>
  <i>Figure 3: The Accordo validation workflow, which intercepts runtime calls to compare memory outputs.</i>
</p>

## Automated CI/CD Workflow

IntelliPerf operates as a command-line tool that can be integrated into CI/CD pipelines through the **IntelliPerf Action**, a GitHub Action wrapper that handles the CI/CD integration and automated pull request creation. When a successful optimization is found, the IntelliPerf Action automatically generates a pull request with the validated fix, including a summary of the bottleneck and the measured performance improvement. The final pull request is the culmination of the entire automated workflow, presenting the developer with a ready-to-merge solution (Figure 6).

<p align="center">
  <img src="assets/intelliperf_ci.png" alt="IntelliPerf CI/CD Integration"><br>
  <i>Figure 4: IntelliPerf's integration into CI/CD pipelines, showing the automated workflow from code analysis to pull request generation.</i>
</p>

The following YAML snippet demonstrates how to integrate IntelliPerf into a GitHub Actions workflow. Key configuration parameters include:

- **`formula`**: Specifies the performance bottleneck to target (e.g., `bankConflict`, `atomicContention`, `memoryAccess`)
- **`docker_image`**: The container image containing the IntelliPerf toolchain
- **`top_n`**: Number of top kernels to consider for analysis (ranked by wallclock time), with optimization applied to the first kernel that exhibits the specific bottleneck being targeted
- **`create_pr`**: Whether to automatically create pull requests with optimizations
- **`build_command`**: Command to build the target application
- **`instrument_command`**: Command to instrument the application for profiling
- **`applications`**: JSON array specifying the commands to run and output file locations


```yaml
      - name: Checkout intelliperf-action action
        uses: actions/checkout@v3
        with:
          repository: AMDResearch/intelliperf-action
          token: ${{ secrets.INTELLIPERF_ACTIONS_TOKEN }}
          path: .github/actions/intelliperf-action

      - name: Run IntelliPerf Action for ${{ matrix.apps.name }}
        uses: ./.github/actions/intelliperf-action
        with:
          formula: "${{ matrix.apps.formula }}"
          docker_image: "intelliperf"
          top_n: "40"
          create_pr: "true"
          intelliperf_actions_token: ${{ secrets.INTELLIPERF_ACTIONS_TOKEN }}
          llm_gateway_key: ${{ secrets.LLM_GATEWAY_KEY }}
          build_command: "${{ matrix.apps.build_command }}"
          instrument_command: "${{ matrix.apps.instrument_command }}"
          project_directory: "${{ matrix.apps.project_directory }}"
          applications: |
            [{
              "command": "${{ matrix.apps.command }}",
              "output_json": "${{ env.OUTPUT_JSON }}"
            }]
```
<p align="center">
  <i>Figure 5: GitHub Actions workflow configuration for integrating IntelliPerf using the IntelliPerf Action.</i>
</p>

<p align="center">
  <img src="assets/intelliperf_pr.png" alt="IntelliPerf PR Example"><br>
  <i>Figure 6: An example of an automatically generated pull request containing a validated optimization.</i>
</p>

### Illustrative Example: Resolving Atomic Contention

The following patch demonstrates IntelliPerf's output. The framework identified an inefficient atomic operation inside a kernel and replaced it with a highly efficient parallel reduction using shared memory, a non-trivial optimization that typically requires significant expertise to implement correctly.

```diff
--- a/examples/contention/reduction/reduction.hip
+++ b/examples/c/contention/reduction/reduction.hip
@@ -28,10 +28,26 @@
 __global__ void reduction_kernel(const float* input, float* result, std::size_t count) {
-  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
-  if (thread_id < count) {
-    const auto value = input[thread_id];
-    atomicAdd(result, value / (thread_id + 1));
+  extern __shared__ float sdata[];
+  const unsigned int tid = threadIdx.x;
+  const unsigned int idx = blockIdx.x * blockDim.x + tid;
+  // load input into shared memory
+  float val = 0.0f;
+  if (idx < count) {
+    val = input[idx] / (idx + 1);
   }
+  sdata[tid] = val;
+  __syncthreads();
+  // do reduction in shared memory
+  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
+    if (tid < s) {
+      sdata[tid] += sdata[tid + s];
+    }
+    __syncthreads();
+  }
+  // write result for this block to global memory
+  if (tid == 0) {
+    atomicAdd(result, sdata[0]);
+  }
 }
```

## Future Work and Roadmap

IntelliPerf's formula-driven architecture is extensible. Future work will focus on expanding its capabilities by adding new formulas to address other critical performance bottlenecks, including:
*   Register Pressure
*   Branch Divergence
*   Memory Locality

We also plan to extend support to other programming models, such as Triton, and to conduct a rigorous quantitative evaluation of IntelliPerf's effectiveness across a wide range of HPC and ML benchmarks and applications. Moreover, we are interested in supporting Radeon GPUs (RDNA) in addition to existing support for Compute GPUs (CDNA).

## Conclusion

IntelliPerf represents a significant step towards the automation of GPU performance engineering. By combining classical program analysis with the generative power of LLMs in a robust, closed-loop framework, it provides a scalable solution to a critical software development challenge. This approach not only accelerates the optimization cycle but also democratizes performance engineering, allowing non-expert developers to achieve expert-level results and freeing up domain experts to focus on more complex, architectural challenges.
