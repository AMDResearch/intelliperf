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

import dspy

from intelliperf.formulas.formula_base import (
	Formula_Base,
	OptimizationTracker,
	Result,
	filter_json_field,
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

		# This temp option allows us to toggle if we want a full or partial instrumentation report
		self._instrumentation_results = None
		self.current_kernel = None
		self.current_args = None
		self.kernel_to_optimize = None
		self.optimization_report = None
		self.bottleneck_report = None
		self.current_summary = None
		self.previous_source_code = None
		self.success = False

		# Initialize optimization tracker
		# Bank conflict optimization minimizes conflicts (lower is better)
		# Automatically calculates conflict_improvement from unoptimized_conflicts / optimized_conflicts
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

		# Track best optimization across iterations
		self.best_speedup = 1.0  # Start at 1.0x (no speedup)
		self.best_conflict_improvement = 1.0  # Start at 1.0x (no improvement)
		self.best_kernel_code = ""
		self.best_iteration_report = ""
		self.best_optimization_results = None

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
						"speedup": 0.0,
						"unoptimized_time": 0,
						"optimized_time": 0,
						"unoptimized_conflicts": self.baseline_bank_conflicts or 0,
						"optimized_conflicts": self.baseline_bank_conflicts or 0,
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

	def get_top_kernel(self) -> str:
		# Filter out any kernel with no bank conflicts
		filtered_report_card = [
			entry for entry in self._initial_profiler_results if entry.get("lds", {}).get("bc", 0) > 0
		]
		# Filter out any kernel with no source code
		filtered_report_card = [entry for entry in filtered_report_card if entry.get("source", {}).get("hip", [])]

		logging.debug(f"Filtered Report Card:\n{json.dumps(filtered_report_card, indent=4)}")

		if len(filtered_report_card) == 0:
			return None
		return filtered_report_card[0]["kernel"]

	def instrument_pass(self) -> Result:
		"""
		Instrument the application, targeting the kernels with the highest bank conflict data

		Returns:
		    Result: Instrumentation data containing the kernel name, arguments, lines, and file path as dict
		"""
		super().instrument_pass()

		# Log instrumentation completion
		self.get_logger().record(
			"instrument_pass_complete",
			{
				"success": True,
				"note": "Instrumentation pass completed via parent class",
			},
		)

		return Result(
			success=False,
			asset=self._instrumentation_results,
			error_report="Instrumentation pass not implemented for bank conflict.",
		)

		# Always instrument the first kernel
		kernel_to_instrument = self.get_top_kernel()
		if kernel_to_instrument is None:
			return Result(
				success=False,
				error_report="No source code found. Please compile your code with -g.",
			)

		omniprobe_output_dir = os.path.join(self._application.get_project_directory(), "memory_analysis_output")

		# Remove directory if it exists and create a new one
		if os.path.exists(omniprobe_output_dir):
			shutil.rmtree(omniprobe_output_dir)

		ecma_regex = generate_ecma_regex_from_list([kernel_to_instrument])
		logging.debug(f"ECMA Regex for kernel names: {ecma_regex}")
		cmd = " ".join(self._application.get_app_cmd())
		logging.debug(f"Omniprobe profiling command is: {cmd}")
		success, output = capture_subprocess_output(
			[
				"omniprobe",
				"--instrumented",
				"--analyzers",
				"MemoryAnalysis",
				"--kernels",
				ecma_regex,
				"--",
				" ".join(self._application.get_app_cmd()),
			],
			working_directory=self._application.get_project_directory(),
		)
		if not success:
			logging.warning(f"Failed to instrument the application: {output}")
			return Result(
				success=False,
				error_report=f"Failed to instrument the application: {output}",
			)

		# Try loading the memory analysis output
		# Find all files in the memory_analysis_output directory
		output_files = glob.glob(os.path.join(omniprobe_output_dir, "memory_analysis_*.json"))
		if len(output_files) == 0:
			return Result(success=False, error_report="No memory analysis output files found.")
		output_file = output_files[0]
		try:
			with open(output_file, "r") as f:
				self._instrumentation_results = json.load(f)
				# for all files, remove the [clone .kd] suffix
				# for analysis in self._instrumentation_results["kernel_analyses"]:
				#    analysis["kernel_info"]["name"] = analysis["kernel_info"]["name"].split(" [clone .kd]")[0]
				logging.debug(f"Instrumentation results: {json.dumps(self._instrumentation_results, indent=4)}")
		except FileNotFoundError:
			logging.warning(f"Memory analysis output file not found: {output_file}")
			return Result(
				success=False,
				error_report=f"Memory analysis output file not found: {output_file}",
			)
		return Result(success=True, asset=self._instrumentation_results)

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

		kernel_to_optimize = self.get_top_kernel()
		if kernel_to_optimize is None:
			return Result(
				success=False,
				error_report="No source code or bank conflicts found. Please compile your code with -g.",
			)

		kernel = None
		kernel_file = None

		# Get the file from the results
		if self._instrumentation_results is None:
			# Get the file from the results
			filtered_report_card = filter_json_field(
				self._initial_profiler_results,
				field="lds",
				subfield="bc",
				comparison_func=lambda x: x > 0,
				target_kernel=target_kernel,
			)

			if len(filtered_report_card) == 0:
				return Result(success=False, error_report="No bank conflicts found.")

			logging.debug(f"Filtered Report Card:\n{json.dumps(filtered_report_card, indent=4)}")

			kernel = filtered_report_card[0]["kernel"]
			self._parse_kernel_signature(kernel)

			# Store baseline metrics on first run
			if self.baseline_bank_conflicts is None:
				self.baseline_bank_conflicts = filtered_report_card[0]["lds"]["bc"]
				self.baseline_time_ms = filtered_report_card[0]["durations"]["ns"] / 1e6

			files = filtered_report_card[0]["source"]["files"]
			kernel_name = get_kernel_name(kernel)
			kernel_file = None

			unoptimized_file_content = None
			for file in files:
				project_dir = os.path.abspath(self._application.get_project_directory())
				file_path = os.path.abspath(file)
				isfile_in_project = os.path.commonpath([project_dir, file_path]) == project_dir
				if os.path.exists(file) and isfile_in_project:
					with open(file, "r") as f:
						unoptimized_file_content = f.read()
						if kernel_name in unoptimized_file_content:
							kernel_file = file
							break
			if kernel_file is None:
				return Result(success=False, error_report="Kernel file not found.")

			# Build problem description
			problem_description = (
				f"Shared memory bank conflicts detected in kernel {kernel}. "
				f"Please optimize to reduce bank conflicts but do not change the semantics."
			)

			self.previous_source_code = unoptimized_file_content

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
		else:
			pass

		if kernel is None:
			return Result(success=False, error_report="Failed to extract the kernel name.")
		if kernel_file is None:
			return Result(success=False, error_report="Failed to extract the kernel file path.")

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

			with open(kernel_file, "w") as f:
				f.write(optimized_file_content)
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
				self.optimization_tracker.add_step(
					diff=diff,
					report=error_report,
					metrics={
						"speedup": 0.0,
						"unoptimized_time": 0,
						"optimized_time": 0,
						"unoptimized_conflicts": self.baseline_bank_conflicts or 0,
						"optimized_conflicts": self.baseline_bank_conflicts or 0,
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
		unoptimized_results = filter_json_field(
			self._initial_profiler_results,
			field="kernel",
			comparison_func=lambda x: x == self.current_kernel_signature,
		)

		unoptimized_time = unoptimized_results[0]["durations"]["ns"]
		unoptimized_conflicts = unoptimized_results[0]["lds"]["bc"]

		# Profile the optimized application
		self._optimization_results = self._application.profile(top_n=self.top_n)

		optimized_results = filter_json_field(
			self._optimization_results,
			field="kernel",
			comparison_func=lambda x: x == self.current_kernel_signature,
		)

		optimized_time = optimized_results[0]["durations"]["ns"]
		optimized_conflicts = optimized_results[0]["lds"]["bc"]

		success = optimized_conflicts < unoptimized_conflicts
		speedup = unoptimized_time / optimized_time
		conflict_improvement = unoptimized_conflicts / optimized_conflicts if optimized_conflicts != 0 else 1
		conflict_improvement_percentage = (
			(unoptimized_conflicts - optimized_conflicts) / unoptimized_conflicts if unoptimized_conflicts != 0 else 0
		) * 100

		self.optimization_report = ""

		# Format the conflict improvement message
		if conflict_improvement_percentage > 1:
			self.optimization_report += (
				f"Bank Conflict Reduction: Successfully reduced shared memory bank conflicts by "
				f"{conflict_improvement_percentage:.1f}%. "
				f"Conflict ratio improved from {unoptimized_conflicts:.1f} to {optimized_conflicts:.1f} "
				f"(lower values indicate fewer conflicts and better performance). "
			)
		else:
			self.optimization_report += (
				f"Bank Conflict Increase: Bank conflicts increased by "
				f"{abs(conflict_improvement_percentage):.1f}%. "
				f"Conflict ratio worsened from {unoptimized_conflicts:.1f} to {optimized_conflicts:.1f} "
				f"(higher values indicate more conflicts and worse performance). "
			)

		# Format the performance improvement message
		if speedup > 1:
			self.optimization_report += (
				f"Performance Gain: Achieved {speedup:.2f}x speedup with execution time "
				f"reduced from {unoptimized_time / 1_000_000:.2f}ms to {optimized_time / 1_000_000:.2f}ms "
				f"({(speedup - 1) * 100:.1f}% faster)."
			)
		else:
			self.optimization_report += (
				f"Performance Loss: Experienced {1 / speedup:.2f}x slowdown with execution time "
				f"increased from {unoptimized_time / 1_000_000:.2f}ms to {optimized_time / 1_000_000:.2f}ms "
				f"({(1 / speedup - 1) * 100:.1f}% slower)."
			)

		# Log performance validation results (always, even if failed)
		self.get_logger().record(
			"performance_validation_complete",
			{
				"success": success,
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

		# Add step to optimization tracker (always - for learning)
		# Tracker will automatically calculate conflict_improvement from before/after values
		diff = self.compute_diff(self.current_kernel_files)

		# Read the optimized code to store in history
		with open(self.current_kernel_files[0], "r") as f:
			optimized_code = f.read()

		self.optimization_tracker.add_step(
			diff=diff,
			report=self.optimization_report,
			metrics={
				"speedup": speedup,
				"unoptimized_time": unoptimized_time,
				"optimized_time": optimized_time,
				"unoptimized_conflicts": unoptimized_conflicts,
				"optimized_conflicts": optimized_conflicts,
			},
			success=success and speedup >= 1,
			request=f"Optimize bank conflicts in kernel {self.current_kernel_signature}",
			optimized_code=optimized_code,
		)

		# Update best if this iteration improved both speedup and conflict reduction
		is_better = speedup > self.best_speedup and conflict_improvement > self.best_conflict_improvement

		if is_better:
			self.best_speedup = speedup
			self.best_conflict_improvement = conflict_improvement
			self.best_iteration_report = self.optimization_report
			self.best_optimization_results = self._optimization_results
			self.best_kernel_code = optimized_code
			# Mark as successful if we achieved any improvement
			if conflict_improvement > 1.0 or speedup > 1.0:
				self.success = True

		self.current_summary = self.optimization_report

		# Always return False to continue through all iterations
		return Result(success=False, error_report=self.optimization_report)

	def write_results(self, output_file: str = None):
		"""
		Writes the results to the output file using the best optimization attempt.
		"""
		# Restore best results for output
		self._optimization_results = self.best_optimization_results
		self.optimization_report = self.best_iteration_report

		for file in self.current_kernel_files:
			with open(file, "w") as f:
				f.write(self.best_kernel_code)

		# Extract metrics from best optimization step
		best_step = self.optimization_tracker.to_dict().get("best_step", {})
		metrics = best_step.get("metrics", {})

		# Build structured metric fields
		metric_fields = {
			"kernel_name": self.current_kernel,
			"metric": "lds_bank_conflict",  # The counter we're optimizing
			"metric_name": "LDS Bank Conflicts",  # Human-readable name
			"metric_before": metrics.get("unoptimized_conflicts", self.baseline_bank_conflicts),
			"metric_after": metrics.get("optimized_conflicts", self.baseline_bank_conflicts),
			"time_before_ms": metrics.get("unoptimized_time", 0) / 1e6,  # Convert ns to ms
			"time_after_ms": metrics.get("optimized_time", 0) / 1e6,  # Convert ns to ms
		}

		# Include optimization history in results
		super().write_results(
			output_file=output_file,
			additional_results={
				"formula": "bankConflict",
				"success": self.success,
				"optimization_history": self.optimization_tracker.to_dict(),
				**metric_fields,
			},
		)

	def summarize_previous_passes(self):
		"""
		Summarizes the results of the previous passes for future prompts.
		"""
		pass
