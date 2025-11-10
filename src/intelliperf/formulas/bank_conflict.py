################################################################################
# MIT License

# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

import glob
import json
import logging
import os
import shutil
import sys

import dspy

from intelliperf.formulas.formula_base import (
	Formula_Base,
	OptimizationTracker,
	Result,
	get_kernel_name,
)
from intelliperf.utils.process import capture_subprocess_output
from intelliperf.utils.regex import generate_ecma_regex_from_list


class BankConflictOptimization(dspy.Signature):
	"""Optimize GPU kernel code to reduce shared memory bank conflicts and improve performance."""

	kernel_code = dspy.InputField(desc="The current kernel source code that needs bank conflict optimization.")

	code_preservation_rules = dspy.InputField(
		desc="""CRITICAL: These rules MUST be followed to avoid compilation errors:

1. **NEVER modify, abbreviate, or remove copyright/license headers**
   - Copy the ENTIRE copyright header exactly as written, character by character
   - DO NOT use '...' or any ellipsis inside comments - this creates unterminated comment errors
   - If you see: /* Copyright (c) 2025... */ you MUST write the full text

2. **NEVER use '...' or ellipsis in C/C++ comments**
   - The sequence '...' does NOT close a /* */ comment block
   - This causes: error: unterminated /* comment
   - Either copy the full comment text or remove it entirely (but keep copyright!)

3. **Preserve the EXACT kernel signature**
   - Do not change: function name, parameter types, parameter names, or return type
   - The kernel must be callable with the same arguments as before"""
	)

	output_format_rules = dspy.InputField(
		desc="""CRITICAL: Output format requirements:

1. **No markdown code blocks or formatting**
   - Do NOT include: ```cpp, ```c, ```hip, or ``` markers
   - Do NOT include any explanatory text before or after the code

2. **Output ONLY the complete, compilable source code**
   - Include all necessary #include statements
   - Include the full copyright header (copied verbatim)
   - Include all function implementations
   - The output must be ready to write directly to a .hip file

3. **Preserve all existing comments and licenses**
   - Copy them exactly as they appear in the original code"""
	)

	problem_description = dspy.InputField(
		desc="Description of the bank conflict issue detected, including specific shared memory access patterns causing conflicts and performance impact."
	)

	# Baseline metrics for reference
	baseline_bank_conflicts = dspy.InputField(
		desc="Baseline bank conflict ratio. Indicates how many shared memory accesses result in bank conflicts - higher values mean more conflicts and worse performance. This is the initial unoptimized value before any optimization attempts."
	)
	baseline_time_ms = dspy.InputField(
		desc="Baseline kernel execution time in milliseconds. This is the initial unoptimized kernel runtime before any optimization attempts."
	)

	# History of all previous attempts
	history = dspy.History = dspy.InputField(
		desc="Complete history of previous optimization attempts, including the code changes (diffs), before/after bank conflict ratios, speedup ratios, and whether each attempt improved or regressed performance. Use this to avoid repeating failed approaches and build on successful patterns."
	)

	previous_failures = dspy.InputField(
		desc="CRITICAL: Analysis of previous failed attempts. DO NOT repeat these mistakes: compilation errors (unterminated comments, missing semicolons), correctness failures (changed semantics, wrong outputs), or performance regressions (same approach with minor variations). Learn from these failures and try fundamentally different optimization strategies."
	)

	latest_diff = dspy.InputField(
		desc="The most recent code change that was attempted. If this resulted in a failure, you MUST try a completely different approach. Do not make minor variations of the same optimization - innovate with a new strategy."
	)

	# Low-level optimization techniques
	optimization_techniques = dspy.InputField(
		desc="""Available low-level bank conflict optimization techniques to consider:

1. **Non-temporal loads**: Treat read-once data as streaming, avoid cache pollution.
   Example: float x = __builtin_nontemporal_load(&input[i]);

2. **Warp-wide cooperative loads** using shuffles:
   float val = __shfl_sync(0xffffffff, local_data, 0);

3. **Prefetching into registers**:
   float prefetch = input[i + 1]; // Use later

4. **Manual register tiling with unrolled loops**:
   float tile[4];
   #pragma unroll
   for (int i = 0; i < 4; ++i) tile[i] = input[idx + i];

5. **Double-buffering** for overlapping memory + compute:
   load_tile(buf_a);
   load_tile(buf_b); compute(buf_a); swap();

6. **Structure-of-Arrays transformation** to align thread memory access:
   float val = soa.x[threadIdx.x]; // Coalesced

7. **Aligned vectorized memory access** using float4:
   float4 x = reinterpret_cast<float4*>(input)[idx];"""
	)

	amd_specific_optimizations = dspy.InputField(
		desc="""AMD-specific optimizations for CDNA GPUs (MI250X, MI300X):

**AMD MFMA (Matrix Fused Multiply-Add) intrinsics**:
These instructions execute on a **wavefront-wide** basis (64 threads), using per-lane vector registers to load parts of matrices A, B, C.

**MFMA Intrinsic Syntax:**
  d = __builtin_amdgcn_mfma_<CDfmt>_<MxNxK><ABfmt>(a, b, c, cbsz, abid, blgp);

**Parameters:**
- CDfmt: format of C and D (e.g., f32, fp32)
- ABfmt: format of A and B (e.g., f16, bf16, i8)
- M, N, K: matrix tile dimensions
- a, b, c: registers or scalars from matrices A, B, C
- d: output register of resulting matrix tile D
- cbsz: broadcast size (0=no broadcast, 1=2-wide broadcast, etc.)
- abid: A-matrix block to broadcast from (used with cbsz)
- blgp: B-matrix swizzle pattern (0=normal, 1=lanes 0-31→32-63, 2=lanes 32-63→0-31, 3=rotate down by 16, 4-7=group-wide broadcast)

**Example (FP32, 16x16x4 GEMM):**
```cpp
__global__ void sgemm_16x16x4(const float *A, const float *B, float *D) {
  using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
  float4 d = {0};

  int mk = threadIdx.y + 4 * threadIdx.x;
  int kn = threadIdx.x + 16 * threadIdx.y;

  float amk = A[mk];
  float bkn = B[kn];
  d = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, d, 0, 0, 0);

  for (int i = 0; i < 4; ++i)
    D[threadIdx.x + i * 16 + threadIdx.y * 4 * 16] = d[i];
}
```
Launch with: dim3 block(16, 4); dim3 grid(1, 1);

**Performance (CDNA2/CDNA3 - MI250X/MI300X):**
- FP32 16x16x4: 256 flops/cycle/CU
- FP16/BF16 16x16x16: 1024 flops/cycle/CU
- INT8 16x16x16: 1024 flops/cycle/CU
- FP64 4x4x4: 128-256 flops/cycle/CU

**Key Benefits:**
- MFMA offers 2-4× throughput over vector FMAs
- Use wavefront-aligned loads and register blocking to match instruction layout
- Avoid using warp-level primitives with the _sync suffix (e.g., __shfl_sync, __ballot_sync) and use the non-sync variants instead for broader compatibility
- We are optimizing for CDNA GPUs (MI300X) so target this architecture

**LDS Bank Conflict Avoidance on AMD:**
- AMD GCN/CDNA architectures have 32 LDS banks (vs NVIDIA's 32)
- Each bank is 4 bytes wide
- Conflicts occur when multiple threads in a wavefront access different addresses in the same bank
- Use padding or stride adjustments to avoid bank conflicts"""
	)

	optimization_categories = dspy.InputField(
		desc="""Categories of optimization strategies to explore:

1. **Bank Conflict Resolution Strategies:**
   - Padding shared memory arrays to avoid conflicts
   - Changing access strides to distribute across banks
   - Transposing data layout in shared memory
   - Using conflict-free indexing schemes
   - Reorganizing thread-to-data mappings

2. **Memory Access Pattern Changes:**
   - Non-temporal loads for streaming data (__ldg, __builtin_nontemporal_load)
   - Warp-wide cooperative loads with shuffles (__shfl, __shfl_down)
   - Manual register tiling and double-buffering
   - Structure-of-Arrays transformations

3. **Advanced Vectorization:**
   - Vectorized memory access patterns (float4, int4)
   - SIMD-friendly loop unrolling
   - Memory coalescing with vector types

4. **AMD-Specific Optimizations:**
   - AMD MFMA (Matrix Fused Multiply-Add) intrinsics for CDNA GPUs (MI250X, MI300X)
   - Wavefront-level optimizations (64-thread blocks)
   - AMD-specific LDS bank organization (32 banks, 4 bytes each)
   - AMD-specific memory hierarchy optimizations

5. **Algorithmic Changes:**
   - Different loop ordering strategies
   - Alternative memory layout schemes
   - New data structure organizations
   - Wavefront/warp specialization and/or workgroup/block specialization"""
	)

	optimized_code = dspy.OutputField(
		desc="The complete, runnable optimized kernel code with reduced bank conflicts. Must preserve the exact kernel signature, all comments, licenses, and copyright notices. The code should eliminate or minimize shared memory bank conflicts through better access patterns or padding."
	)


class bank_conflict(Formula_Base):
	def __init__(
		self,
		name: str,
		build_command: list,
		instrument_command: list,
		project_directory: str,
		app_cmd: list,
		top_n: int,
		model: str = "gpt-4o",
		provider: str = "openai",
		in_place: bool = False,
		**kwargs,
	):
		super().__init__(
			name,
			build_command,
			instrument_command,
			project_directory,
			app_cmd,
			top_n,
			model,
			provider,
			in_place,
			**kwargs,
		)

		self.current_kernel = None
		self.current_args = None
		self.optimization_report = None
		self.bottleneck_report = None
		self.current_summary = None
		self.previous_source_code = None

		# Cached optimization candidate (set on first optimize_pass call)
		self._optimization_candidate = None
		self._kernel_file = None
		self._unoptimized_file_content = None

		# Initialize optimization tracker
		# Bank conflict optimization: minimize conflicts (lower is better)
		# Automatically calculates conflict_improvement = unoptimized_conflicts / optimized_conflicts
		# Higher improvement ratio is better (e.g., 3.5 / 0.5 = 7.0x improvement)
		self.optimization_tracker = OptimizationTracker(
			max_iterations=self.num_attempts,
			primary_metric="conflict_improvement",
			maximize=False,  # False = minimize raw metric (conflicts)
			before_metric="unoptimized_conflicts",
			after_metric="optimized_conflicts",
		)

		# Store baseline metrics (set during first profiling)
		self.baseline_bank_conflicts = None
		self.baseline_time_ms = None

	def build_pass(self, validate_build_result=True) -> Result:
		"""
		Build the application and store the summary.

		Args:
		    validate_build_result (bool): Whether to validate the build result

		Returns:
		    Result: Build status and the output file path
		"""
		result = super().build(validate_build_result=validate_build_result)

		# Log build result
		if not result:
			self.current_summary = result.error_report
			self.get_logger().record(
				"build_pass_failed",
				{
					"success": False,
					"error_report": result.error_report,
					"kernel_files": getattr(self, "current_kernel_files", []),
				},
			)
			# Add compilation failure to history so LLM learns from the error
			if hasattr(self, "current_kernel_files") and self.current_kernel_files:
				diff = self.compute_diff(self.current_kernel_files)
				with open(self.current_kernel_files[0], "r") as f:
					failed_code = f.read()

			error_report = f"Compilation Failed: {result.error_report}"
			self.optimization_tracker.add_step(
				diff=diff,
				report=error_report,
				metrics={
					"speedup": 1.0,  # No change
					"unoptimized_time": self.baseline_time_ms * 1e6 if self.baseline_time_ms else 0,
					"optimized_time": self.baseline_time_ms * 1e6 if self.baseline_time_ms else 0,
					"unoptimized_conflicts": self.baseline_bank_conflicts or 0,
					"optimized_conflicts": self.baseline_bank_conflicts or 0,
					"conflict_improvement": 1.0,  # Explicitly set to 1.0 (no change)
				},
				success=False,
				request=f"Optimize bank conflicts in kernel {getattr(self, 'current_kernel_signature', 'unknown')}",
				optimized_code=failed_code,
			)
		else:
			self.get_logger().record(
				"build_pass_success",
				{
					"success": True,
					"kernel_files": getattr(self, "current_kernel_files", []),
				},
			)
		return result

	def profile_pass(self) -> Result:
		"""
		Profile the application using guided-tuning and collect bank conflict data

		Returns:
		    Result: DataFrame containing the performance report card
		"""
		super().profile_pass()

		# Log profiling results
		if hasattr(self, "_initial_profiler_results") and self._initial_profiler_results:
			self.get_logger().record(
				"profile_pass_complete",
				{
					"profiler_results": self._initial_profiler_results,
					"top_n": self.top_n,
				},
			)



	def optimize_pass(
		self,
		temperature: float = 0.0,
		max_tokens: int = 3000,
		target_kernel: str = None,
	) -> Result:
		"""
		Optimize the kernel to remove shared memory bank conflicts via OpenAI API

		Args:
		    temperature (float): Sampling temperature for OpenAI API
		    max_tokens (int): Maximum tokens for OpenAI API

		Returns:
		    Result: Optimized kernel as a file path
		"""
		super().optimize_pass()

		system_prompt = (
			"You are a skilled GPU HIP programmer specializing in optimizing kernels "
			"to reduce shared memory bank conflicts and improve performance."
		)

		# Get LLM instance (initialized once per formula)
		llm = self.get_llm(system_prompt)

		# Find and cache candidate on first call
		if self._optimization_candidate is None:
			self._optimization_candidate = self._initial_profiler_results.find_optimization_candidate(
				metric_filter=lambda k: (
					k.lds_bank_conflicts is not None
					and k.lds_bank_conflicts > 0
					and (target_kernel is None or target_kernel in k.kernel_name)
				),
				require_source_code=True,
			)

			if self._optimization_candidate is None:
				error_msg = (
					"No optimization candidates found: No kernels with bank conflicts (>0) and available source code. "
					"This could mean:\n"
					"  1. All kernels have no shared memory bank conflicts\n"
					"  2. Source code is not available for kernels with bank conflicts\n"
					"  3. No kernels were profiled (check if application runs correctly)"
				)
				logging.error(error_msg)
				sys.exit(1)

			logging.info(
				f"Found optimization candidate: {self._optimization_candidate.kernel_name} "
				f"(bank conflicts: {self._optimization_candidate.lds_bank_conflicts:.2f}, "
				f"duration: {self._optimization_candidate.duration_ns / 1e6:.2f} ms)"
			)

			# Store baseline metrics
			self.baseline_bank_conflicts = self._optimization_candidate.lds_bank_conflicts
			self.baseline_time_ms = self._optimization_candidate.duration_ns / 1e6

			# Parse kernel signature
			kernel = self._optimization_candidate.kernel_name
			self._parse_kernel_signature(kernel)

			# Get source files and cache them
			files = self._optimization_candidate.source_files
			self._kernel_file, self._unoptimized_file_content = self.find_kernel_file(files, kernel)

			# Save reference code once
			kernel_name = get_kernel_name(kernel)
			self.get_logger().save_reference_code(kernel_name, self._unoptimized_file_content)

		# Use cached values
		kernel = self._optimization_candidate.kernel_name
		kernel_file = self._kernel_file
		unoptimized_file_content = self._unoptimized_file_content

		# Build problem description
		problem_description = (
			f"Shared memory bank conflicts detected in kernel {kernel}. "
			f"Please optimize to reduce bank conflicts but do not change the semantics."
		)

		# Set up previous source code for diff tracking
		if self.previous_source_code is None:
			self.previous_source_code = unoptimized_file_content

		# Build bottleneck report
		if self.current_args:
			self.bottleneck_report = (
				f"Bank Conflict Detection: IntelliPerf identified shared memory bank conflicts in kernel "
				f"`{self.current_kernel}` with arguments `{self.current_args}`. Bank conflicts occur when multiple threads "
				f"access the same memory bank simultaneously, causing serialization and performance degradation."
			)
		else:
			self.bottleneck_report = (
				f"Bank Conflict Detection: IntelliPerf identified shared memory bank conflicts in kernel "
				f"`{self.current_kernel}`. Bank conflicts occur when multiple threads "
				f"access the same memory bank simultaneously, causing serialization and performance degradation."
			)

		self.current_kernel_files = [kernel_file]

		logging.debug(f"System prompt: {system_prompt}")
		logging.debug(f"Problem description: {problem_description}")

		try:
			# Use DSPy signature with history - pass inputs as kwargs
			# Build failure analysis from history
			tracker_dict = self.optimization_tracker.to_dict()
			failed_steps = [s for s in tracker_dict.get("steps", []) if not s.get("success", False)]
			previous_failures_text = ""
			if failed_steps:
				previous_failures_text = "Previous failed attempts:\n"
				for i, step in enumerate(failed_steps[-3:], 1):  # Show last 3 failures
					previous_failures_text += f"\nAttempt {i}: {step.get('report', 'Unknown error')}\n"
			else:
				previous_failures_text = "No previous failures yet. This is an early iteration."

			# Get latest diff if available
			latest_diff_text = ""
			if tracker_dict.get("steps"):
				latest_step = tracker_dict["steps"][-1]
				latest_diff_text = latest_step.get("diff", "No diff available")
			else:
				latest_diff_text = "No previous attempts yet."

			response = llm.ask(
				signature=BankConflictOptimization,
				answer_type=None,  # Return full response object
				record_meta="bank_conflict_optimization",
				kernel_code=unoptimized_file_content,
				code_preservation_rules="",  # Field description contains the critical rules
				output_format_rules="",  # Field description contains the format requirements
				problem_description=problem_description,
				baseline_bank_conflicts=str(self.baseline_bank_conflicts),
				baseline_time_ms=str(self.baseline_time_ms),
				history=self.optimization_tracker.get_dspy_history(),
				previous_failures=previous_failures_text,
				latest_diff=latest_diff_text,
				optimization_techniques="",  # Field description contains the full content
				amd_specific_optimizations="",  # Field description contains the full content
				optimization_categories="",  # Field description contains the full content
			)

			# Extract optimized code from response
			if hasattr(response, "optimized_code"):
				optimized_file_content = response.optimized_code.strip()
			else:
				# Fallback for string response
				optimized_file_content = str(response).strip()

			optimized_file_content = self.postprocess_llm_code(optimized_file_content)

			# Log successful optimization
			self.get_logger().record(
				"optimization_success",
				{
					"optimized_code_length": len(optimized_file_content),
					"kernel_file": kernel_file,
				},
			)

			# Write and log optimized code immediately after LLM generation
			self.write_and_log_optimized_code(kernel_file, optimized_file_content)
			logging.debug(f"Optimized file content: {optimized_file_content}")
			return Result(
				success=True,
				asset={
					"optimized_code_path": kernel_file,
					"optimized_code_string": optimized_file_content,
				},
			)
		except Exception as e:
			error_msg = f"An unexpected error occurred - {str(e)}"
			self.get_logger().record(
				"optimization_error",
				{"error": error_msg, "error_type": type(e).__name__},
			)
			logging.error(error_msg)
			return Result(success=False, error_report=error_msg)

	def compiler_pass(self) -> Result:
		"""
		Compile the application

		Returns:
		    Result: Compilation status and the output file path
		"""
		return super().compile_pass()

	def correctness_validation_pass(self, accordo_absolute_tolerance: float = 1e-6) -> Result:
		"""
		Validate the optimized kernel by comparing the output with the reference kernel

		Args:
		    accordo_absolute_tolerance (float): Absolute tolerance for Accordo
		Returns:
		    Result: Validation status
		"""
		result = super().correctness_validation_pass(self.current_kernel, self.current_args, accordo_absolute_tolerance)

		# Log correctness validation result
		if not result:
			self.current_summary = result.error_report
			self.get_logger().record(
				"correctness_validation_failed",
				{
					"success": False,
					"error_report": result.error_report,
					"kernel": self.current_kernel,
					"args": self.current_args,
				},
			)
			# Add correctness failure to history so LLM learns from the error
			if hasattr(self, "current_kernel_files") and self.current_kernel_files:
				diff = self.compute_diff(self.current_kernel_files)
				with open(self.current_kernel_files[0], "r") as f:
					failed_code = f.read()

			error_report = f"Correctness Validation Failed: {result.error_report}"
			# For correctness failures, we don't have valid metrics since the code didn't run correctly
			# Set conflict_improvement to 1.0 (no change) to indicate no progress
			self.optimization_tracker.add_step(
				diff=diff,
				report=error_report,
				metrics={
					"speedup": 1.0,  # No change
					"unoptimized_time": self.baseline_time_ms * 1e6 if self.baseline_time_ms else 0,
					"optimized_time": self.baseline_time_ms * 1e6 if self.baseline_time_ms else 0,
					"unoptimized_conflicts": self.baseline_bank_conflicts or 0,
					"optimized_conflicts": self.baseline_bank_conflicts or 0,  # Same as baseline (no improvement)
					"conflict_improvement": 1.0,  # Explicitly set to 1.0 (no change)
				},
				success=False,
				request=f"Optimize bank conflicts in kernel {getattr(self, 'current_kernel_signature', 'unknown')}",
				optimized_code=failed_code,
			)
		else:
			self.get_logger().record(
				"correctness_validation_success",
				{
					"success": True,
					"kernel": self.current_kernel,
					"args": self.current_args,
				},
			)
		return result

	def performance_validation_pass(self) -> Result:
		# Get unoptimized metrics using new ProfileResult API
		# Use the cached candidate kernel name
		kernel_name = self._optimization_candidate.kernel_name
		unoptimized_kernel = self._optimization_candidate

		unoptimized_time = unoptimized_kernel.duration_ns
		unoptimized_conflicts = unoptimized_kernel.lds_bank_conflicts

		# Profile the optimized application
		self._optimization_results = self._application.profile(top_n=self.top_n)

		# Get optimized metrics
		optimized_kernel = self._optimization_results.get_kernel(kernel_name)
		if not optimized_kernel:
			return Result(
				success=False,
				error_report=f"Could not find kernel {self.current_kernel_signature} in optimized profiling results",
			)

		optimized_time = optimized_kernel.duration_ns
		optimized_conflicts = optimized_kernel.lds_bank_conflicts

		# Add step to optimization tracker (always - for learning)
		# Tracker will automatically calculate: speedup, conflict_improvement, and success
		diff = self.compute_diff(self.current_kernel_files)

		# Read the optimized code to store in history
		with open(self.current_kernel_files[0], "r") as f:
			optimized_code = f.read()

		step = self.optimization_tracker.add_step(
			diff=diff,
			report="",  # Will build report after calculations
			metrics={
				"unoptimized_time": unoptimized_time,
				"optimized_time": optimized_time,
				"unoptimized_conflicts": unoptimized_conflicts,
				"optimized_conflicts": optimized_conflicts,
			},
			request=f"Optimize bank conflicts in kernel {self.current_kernel_signature}",
			optimized_code=optimized_code,
		)

		# Query calculated values from tracker
		speedup = step.metrics.get("speedup", 1.0)
		conflict_improvement = step.metrics.get("conflict_improvement", 1.0)

		# Calculate percentage improvement for readability
		conflict_improvement_percentage = (
			(unoptimized_conflicts - optimized_conflicts) / unoptimized_conflicts * 100
			if unoptimized_conflicts > 0
			else 0
		)

		# Build report using tracker's calculated values
		self.optimization_report = ""

		# Format the conflict improvement message
		if conflict_improvement > 1:
			self.optimization_report += (
				f"Bank Conflict Reduction: Successfully reduced shared memory bank conflicts by "
				f"{conflict_improvement:.2f}x. "
				f"Conflict ratio improved from {unoptimized_conflicts:.1f} to {optimized_conflicts:.1f} "
				f"({conflict_improvement_percentage:.1f}% reduction - lower values mean fewer conflicts). "
			)
		else:
			self.optimization_report += (
				f"Bank Conflict Increase: Bank conflicts increased by "
				f"{1 / conflict_improvement:.2f}x. "
				f"Conflict ratio worsened from {unoptimized_conflicts:.1f} to {optimized_conflicts:.1f} "
				f"({abs(conflict_improvement_percentage):.1f}% increase - higher values mean more conflicts). "
			)

		# Format the performance improvement message
		if speedup > 1:
			self.optimization_report += (
				f"Performance Gain: Achieved {speedup:.2f}x speedup with execution time "
				f"reduced from {unoptimized_time / 1e6:.2f}ms to {optimized_time / 1e6:.2f}ms "
				f"({(speedup - 1) * 100:.1f}% faster)."
			)
		else:
			self.optimization_report += (
				f"Performance Loss: Experienced {1 / speedup:.2f}x slowdown with execution time "
				f"increased from {unoptimized_time / 1e6:.2f}ms to {optimized_time / 1e6:.2f}ms "
				f"({(1 / speedup - 1) * 100:.1f}% slower)."
			)

		# Update the step's report with the built message
		step.report = self.optimization_report

		# Log performance validation results (using tracker's calculated values)
		self.get_logger().record(
			"performance_validation_complete",
			{
				"success": step.success,
				"unoptimized_time_ns": unoptimized_time,
				"optimized_time_ns": optimized_time,
				"unoptimized_conflicts": unoptimized_conflicts,
				"optimized_conflicts": optimized_conflicts,
				"speedup": speedup,
				"conflict_improvement": conflict_improvement,
				"conflict_improvement_percentage": conflict_improvement_percentage,
				"optimization_report": self.optimization_report,
			},
		)

		logging.info(self.optimization_report)

		self.current_summary = self.optimization_report

		# Always return False to continue through all iterations
		return Result(success=False, error_report=self.optimization_report)

	def write_results(self, output_file: str = None):
		"""
		Writes the results to the output file using the best optimization attempt.
		"""
		# Restore best code from tracker
		best_code = self.optimization_tracker.get_best_code()
		if best_code:
			for file in self.current_kernel_files:
				with open(file, "w") as f:
					f.write(best_code)

		# Extract metrics from best optimization step
		metrics = self.optimization_tracker.get_best_metrics()

		# Determine metric_after: use best if available, otherwise null to indicate no improvement
		metric_after = None
		time_after_ms = None
		if metrics and "optimized_conflicts" in metrics:
			# Only use the measured value if it actually improved
			if metrics["optimized_conflicts"] < self.baseline_bank_conflicts:
				metric_after = metrics["optimized_conflicts"]
				time_after_ms = metrics.get("optimized_time", 0) / 1e6

		# Build structured metric fields
		metric_fields = {
			"kernel_name": self.current_kernel,
			"metric": "lds_bank_conflict",  # The counter we're optimizing
			"metric_name": "LDS Bank Conflicts",  # Human-readable name
			"metric_before": self.baseline_bank_conflicts,
			"metric_after": metric_after,  # None if no optimization succeeded
			"time_before_ms": self.baseline_time_ms if self.baseline_time_ms else 0.0,
			"time_after_ms": time_after_ms,  # None if no optimization succeeded
		}

		# Include optimization history in results
		super().write_results(
			output_file=output_file,
			additional_results={
				"optimization_history": self.optimization_tracker.to_dict(),
				"formula": "bankConflict",
				"success": self.optimization_tracker.is_successful(),
				**metric_fields,
			},
		)

	def summarize_previous_passes(self):
		"""
		Summarizes the results of the previous passes for future prompts.
		"""
		pass
