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

import json
import logging
import os
import difflib
import dspy

from intelliperf.core.llm_new import LLM
from intelliperf.formulas.formula_base import (
	Formula_Base,
	Result,
	filter_json_field,
	get_kernel_name,
)
from intelliperf.utils.env import get_llm_api_key


class SwizzlingOptimization(dspy.Signature):
	"""Optimize GPU kernel code by applying a swizzling pattern to improve L2 cache locality."""

	prompt = dspy.InputField(
		desc="The user prompt containing the original code, memory analysis, and optimization history."
	)

	reason_why_old_was_slow = dspy.OutputField(
		desc='JSON dictionary where keys are "iteration X" and values are sentences explaining why the swizzling pattern in that iteration was suboptimal.'
	)
	summary_of_optimization = dspy.OutputField(
		desc="An overview of the new code swizzling optimization that will be implemented."
	)
	reason_why_new_should_be_better = dspy.OutputField(
		desc="A comparison of the new optimization to the old optimizations, explaining why it should be better."
	)
	result_code = dspy.OutputField(
		desc="The full kernel code with the new swizzling optimization applied. This code should be complete and runnable."
	)
	swizzling_pattern = dspy.OutputField(
		desc="ONLY the swizzling pattern that maps old pid's to new pid's. This will be used for visualization. I need the full swizzling pattern, which starts from grabbing the tl.program_id and ending with remapping the pid. Note that I always want the original pid to be written to a variable called pid and ending in a variable called pid. If we have a 2D grid of pids, they must be called pid_m and pid_n. It is very important that you name the variables by this format and write the whole code based around these variable names so that it runs successfully. "
	)


class swizzling_test(Formula_Base):
	def __init__(
		self,
		name: str,
		build_command: list,
		instrument_command: list,
		project_directory: str,
		app_cmd: list,
		top_n: int,
		only_consider_top_kernel=False,
		model: str = "gpt-4o",
		provider: str = "openai",
		in_place: bool = False,
		output_kernel_file: str = None,
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
		)

		self.output_kernel_file = output_kernel_file
		# This temp option allows us to toggle if we want a full or partial instrumentation report
		self.only_consider_top_kernel = only_consider_top_kernel
		self._instrumentation_results = None
		self.current_kernel = None
		self.current_args = None
		self.current_kernel_signature = None
		self.kernel_to_optimize = None
		self.optimization_report = None
		self.bottleneck_report = None
		self.current_summary = None
		self.previous_source_code = None
		self.memory_analysis_output = None
		self.success = False
		
		# New fields for logging
		self.memory_analysis_prompt = None
		self.optimization_prompt = None
		self.current_swizzling_pattern = None
		self.memory_analysis_reasoning = None
		self.optimization_reasoning = None

		# New fields for iteration history tracking
		self.iteration_history = []  # List of dicts with {iteration, diff, report, success}
		self.current_iteration = 0
		self.memory_analysis_done = False
		self.last_applied_diff = None
		self.initial_source_code = None

		self.max_iterations = 10
		self.best_l2_improvement = -float("inf")
		self.best_speedup = 0.0
		self.best_diff = ""
		self.best_kernel_name = ""
		self.best_iteration_report = ""
		self.best_kernel_code = ""
		self.best_swizzling_pattern = ""
		self.l2_improvement_history = []

	def compute_diff(self, file_paths: list) -> str:
		"""
		Compute the diff between the current and initial versions of the files.

		Args:
			file_paths (list): A list of file paths.

		Returns:
			str: The diff string.
		"""
		diff_str = ""
		for file_path in file_paths:
			with open(file_path, "r") as f:
				current_content = f.readlines()
			initial_content = self.initial_source_code.splitlines(True)
			diff = difflib.unified_diff(initial_content, current_content)
			diff_str += f"--- a/{file_path}\n"
			diff_str += f"+++ b/{file_path}\n"
			diff_str += "".join(diff)
		return diff_str

	def build_pass(self, validate_build_result=True) -> Result:
		"""
		Build the application and store the summary.

		Args:
		    validate_build_result (bool): Whether to validate the build result

		Returns:
		    Result: Build status and the output file path
		"""
		result = super().build(validate_build_result=validate_build_result)
		if not result:
			self.current_summary = result.error_report
		return result

	def profile_pass(self) -> Result:
		"""
		Profile the application using guided-tuning and collect l2 hit rate data

		Returns:
		    Result: DataFrame containing the performance report card
		"""
		return super().profile_pass()

	def instrument_pass(self) -> Result:
		"""
		Instrument the application, targeting the kernels with the lowest l2 hit rate

		Returns:
		    Result: Instrumentation data containing the kernel name, arguments, lines, and file path as dict
		"""
		super().instrument_pass()

		return Result(
			success=False,
			asset=self._instrumentation_results,
			error_report="The instrumentation is not implemented for swizzling.",
		)

	def optimize_pass(self, temperature: float = 0.0, max_tokens: int = 3000) -> Result:
		"""
		Optimize the kernel to improve l2 hit rate through block swizzling via two-stage LLM approach

		Args:
		        temperature (float): Sampling temperature for OpenAI API
		        max_tokens (int): Maximum tokens for OpenAI API

		Returns:
		        Result: Optimized kernel as a file path
		"""
		super().optimize_pass()
		llm_key = get_llm_api_key()

		# Increment iteration counter
		self.current_iteration += 1

		# First stage: Memory analysis system prompt (only run once)
		analysis_system_prompt = (
			"You are a skilled GPU programmer with deep expertise in memory access patterns and cache locality. "
			"You will analyze code to understand memory access patterns and locality opportunities, "
			"but you will not modify any code. Focus on providing detailed, accurate insights about "
			"memory access patterns that can be used to improve cache locality through block swizzling."
		)

		# Second stage: Optimization system prompt
		optimization_system_prompt = (
			"You are a skilled GPU programmer specializing in block swizzling optimization. "
			"Given a kernel and memory access analysis, you will implement swizzling to improve L2 cache locality. "
			"Do not modify the kernel signature. Do not touch any other code, licenses, copyrights, or comments in the file. "
			"If you remove the copyright, your solution will be rejected. "
			"Do not include any markdown code blocks or text other than the code."
		)

		provider = self.provider
		model = self.model
		
		# Only create analysis_llm if we haven't done memory analysis yet
		if not self.memory_analysis_done:
			analysis_llm = LLM(
				api_key=llm_key,
				system_prompt=analysis_system_prompt,
				model=model,
				provider=provider,
			)
		
		optimization_llm = LLM(
			api_key=llm_key,
			system_prompt=optimization_system_prompt,
			model=model,
			provider=provider,
		)

		kernel = None
		kernel_file = None

		if self._instrumentation_results is None:
			# Get the file from the results - look for kernels with low l2 hit rate
			field = "l2"
			subfield = "hr"
			min_l2_hit_rate = 95  # Look for kernels with less than 95% l2 hit rate
			filtered_report_card = filter_json_field(
				self._initial_profiler_results,
				field=field,
				subfield=subfield,
				comparison_func=lambda x: x < min_l2_hit_rate,
			)

			if len(filtered_report_card) == 0:
				return Result(success=False, error_report="No kernels with low l2 hit rate found.")

			logging.debug(f"Filtered Report Card:\n{json.dumps(filtered_report_card, indent=4)}")

			kernel = filtered_report_card[0]["kernel"]
			files = filtered_report_card[0]["source"]["files"]
			kernel_name = get_kernel_name(kernel)

			if not self.best_kernel_name:
				self.best_kernel_name = kernel_name

			logging.debug(f"Kernel name: {kernel_name}")
			kernel_file = None
			for file in files:
				if os.path.exists(file):
					with open(file, "r") as f:
						unoptimized_file_content = f.read()
						if kernel_name in unoptimized_file_content:
							kernel_file = file
							break
			if kernel_file is None:
				return Result(
					success=False,
					error_report=f"Kernel file not found for kernel {kernel}.",
				)

			# Stage 1: Memory access pattern analysis (only run once)
			if not self.memory_analysis_done:
				with open(kernel_file, "r") as f:
					initial_file_content = f.read()
				self.initial_source_code = initial_file_content
				analysis_prompt = (
					f"{self.initial_source_code}\n\n"
					"I have this triton kernel and am trying to understand the memory access patterns of the kernel, and where there is memory locality that can be taken advantage of in the hardware cache. I will use this to swizzle the block id to better align the work so we have better cache locality.\n\n"
					"I DO NOT want you to rewrite any code. I only want you to give me an overview for the memory access patterns and memory locality of the kernel. This will be used as context for future prompts that will take advantage of your insights. Make sure these insights on memory access patterns and locality between blocks in the kernel are accuracy and insightful so that I can actually take advantage of them to improve locality."
				)
				self.memory_analysis_prompt = analysis_prompt

				self.bottleneck_report = (
					f"L2 Cache Locality Detection: IntelliPerf identified suboptimal L2 cache hit rate "
					f"in kernel `{kernel_name}`. Poor cache locality occurs when "
					f"blocks accessing related memory are scheduled to different XCDs with separate L2 caches, "
					f"reducing overall cache effectiveness."
				)

				# Stage 1: Get memory access analysis
				try:
					logging.debug(f"Analysis prompt: {analysis_prompt}")
					self.memory_analysis_output, self.memory_analysis_reasoning = analysis_llm.ask(
						analysis_prompt, answer_type="memory_analysis_output"
					)
					self.memory_analysis_output = self.memory_analysis_output.strip()
					self.memory_analysis_reasoning = self.memory_analysis_reasoning.strip()
					
					logging.debug(f"Memory analysis output: {self.memory_analysis_output}")
					logging.debug(f"Memory analysis reasoning: {self.memory_analysis_reasoning}")

					if self.output_kernel_file:
						with open(self.output_kernel_file, "w") as f:
							f.write("Memory access pattern prompt:\n")
							f.write(f"{self.memory_analysis_prompt}\n\n")
							f.write("Memory access pattern reasoning:\n")
							f.write(f"{self.memory_analysis_reasoning or 'Not provided.'}\n\n")
							f.write("Memory access pattern response:\n")
							f.write(f"{self.memory_analysis_output}\n\n")

					self.memory_analysis_done = True
				except Exception as e:
					logging.error(f"Failed to get memory analysis - {str(e)}")
					return Result(success=False, error_report=f"Failed to get memory analysis - {str(e)}")

			history_prompt_part = ""
			if self.iteration_history:
				history_prompt_part += "Here is the history of previous optimization attempts:\n\n"
				for item in self.iteration_history:
					history_prompt_part += f"--- Iteration {item['iteration']} ---\n"
					history_prompt_part += f"Applied diff:\n{item['diff']}\n"
					history_prompt_part += f"Profiling report:\n{item['report']}\n\n"

			with open(kernel_file, "r") as f:
				current_file_content = f.read()

			# Stage 2: Swizzling optimization
			optimization_prompt = (
				f"The original code is: {self.initial_source_code}\n\n"
				f"The memory analysis is: {self.memory_analysis_output}\n\n"
				f"{history_prompt_part}"
				"Pay special attention to the swizzling pattern in the diff. If you see a swizzling pattern in the diff, do not reimplement it. Instead, try to implement an completely new approach to swizzling."
				"On the MI300x GPU there are multiple XCDs, and each XCD has its own L2 cache. So that blocks on the same XCD that access the same memory will likely hit in the shared L2 cache and thus improve the L2 hit rate of the program. For this reason, blocks that share the same data should be scheduled to the same XCD. Your task is to find the swizzling formulation such that blocks that access the same memory will be scheduled to the same XCD.\n\n"
				"MI300X architecture specification\n\n"
				"The GPU contains 8 XCDs.\n\n"
				"Each XCD has its own L2 cache.\n\n"
				"XCDs contain multiple Compute Units (CUs) and blocks are assigned to CUs.\n\n"
				"We want to maximize utilization by assigning an equal number of blocks to each XCD.\n\n"
				"HIP runtime scheduling of blocks:\n\n"
				'By default, the hardware scheduler assigns each incoming block, in order, to XCDs in a cyclic ("round-robin") sequence:\n\n'
				"// pseudocode for default mapping for each block in [0, num_blocks):\n\n"
				"assigned_xcd = block % num_XCDs; // execute old_blockIdx on XCD assigned_xcd\n\n"
				'Once it reaches XCD num_XCD–1, it "wraps around" and continues assigning the next blocks to XCD 0, then XCD 1, and so on.\n\n'
				'If there are more blocks than XCDs, the scheduler effectively makes multiple "rounds," each of size num_XCD.\n\n'
				"Swizzling goal\n\n"
				"Recompute the block index with the old block index, number of XCDs on the GPU, and total number of blocks in the program so that:\n\n"
				"Blocks that share the same data map to the same XCD until that XCD's share is filled.\n\n"
				"Work remains evenly balanced across XCDs.\n\n"
				"We want to understand how the blocks are strided by the round robin scheduler. Some question that you might want to ask (but might not necessarily be relevant) are: How do you calculate the XCD id that the block was originally mapped to? How many blocks are in each XCD/ How do you understand the stride of block indexes. If we have to round robin for multiple iterations, how do we calculate the number of iterations that the block index was assigned on? How can we use this to make an offset for reassigning the block index.\n\n"
				"There are potentially many more question that you might want to ask when understanding how to best swizzle the kernel to take advantage of locality. To be very clear, the optimal swizzling pattern will change by algorithm. Different algorithms reuse data differently, and thus the blocks that should share the same L2 cache will change by different algorithms based on memory access patterns. I want you to deeply understand how to do this for the specific algorithm we are working on.\n\n"
				"I want you to consider the swizzling pattern step by step and then put everything together in the formula.\n\n"
				"Task\n\n"
				"num_XCD = 8 in this hardware architecture. In the case of this program, num_blocks is equal to num_SMS, so you can directly use that argument. Make sure you do not change the parameters in the kernel function, as this will break the code. The function signature must stay exactly the same or the code will fail.\n\n"
				"Propose a swizzling pattern as one or a few lines of code inside the kernel that reassigns the block index. For HIP kernels, you must still eventually assign threadId. For the HIP kernels, also make sure that you use the new swizzled block ids for all thread id computation. For Triton kernels, you must still eventually assign pid. Rewrite the code of the entire kernel without putting in any placeholders. I want to be able to take the code, copy it into a new file, and run it on the testbench without any extra work. Again, make sure to not change the kernel function signature and only add new swizzling lines within the kernel using the available parameters.\n\n"
				"EXTREMELY IMPORTANT - Make sure that you keep trying to push the performance of the kernel with your swizzling pattern. It is possible for the implementation to be faster, so try to find this."
				"EXTREMELY IMPORTANT - Do not include any markdown code blocks or text other than the code. DO NOT start the code with 'python'. I want you to straight directly output the code. I want to be able to copy and paste the code into a new file and run it on the testbench without any extra work."
				"EXTREMELY IMPORTANT - Make sure to try new approaches to swizzling. Do not just use the same approach as the previous time. If you have previously tried an approach and it is shown in the diff, do not reimplement it. Instead, try to implement an completely new approach to swizzling."
				"EXTREMELY IMPORTANT - Make sure to not change the kernel function signature. Do not add any new parameters to the kernel function. Do not change the return type of the kernel function. Do not change the name of the kernel function. Do not change the arguments of the kernel function. Do not change the return type of the kernel function. Do not change the name of the kernel function. Do not change the arguments of the kernel function. Do not change the return type of the kernel function. Do not change the name of the kernel function. Do not change the arguments of the kernel function."
				"EXTREMELY IMPORTANT - I always want the original pid to be written to a variable called pid and ending in a variable called pid. If we have a 2D grid of pids, they must be called pid_m and pid_n. It is very important that you name the variables by this format and write the whole code based around these variable names so that it runs successfully."
				"EXTREMELY IMPORTANT - Make sure your output is in the correct format. The fields are reason_why_old_was_slow, summary_of_optimization, reason_why_new_should_be_better, result_code, and swizzling_pattern."
			)

			if self.current_summary is not None:
				optimization_prompt += f"\n\nThe current summary is: {self.current_summary}"
				cur_diff = self.compute_diff([kernel_file])
				optimization_prompt += f"\nThe diff between the current and initial code is: {cur_diff}"
		else:
			pass

		if kernel is None:
			return Result(success=False, error_report="Failed to extract the kernel name.")
		if kernel_file is None:
			return Result(success=False, error_report="Failed to extract the kernel file path.")

		logging.debug(f"Optimization prompt: {optimization_prompt}")

		self.current_kernel = kernel.split("(")[0]
		#self.current_args = kernel.split("(")[1].split(")")[0].split(",")
		self.current_kernel_signature = kernel

		self.current_kernel_files = [kernel_file]
		try:
			with open(kernel_file, "r") as f:
				code_before_opt = f.read()

			response, self.optimization_reasoning = optimization_llm.ask(
				optimization_prompt, signature=SwizzlingOptimization
			)
			optimized_file_content = response.result_code.strip()
			self.current_swizzling_pattern = response.swizzling_pattern.strip()
   
			if self.optimization_reasoning:
				self.optimization_reasoning = self.optimization_reasoning.strip()
			
			logging.debug(f"Optimization reasoning: {self.optimization_reasoning}")
			
			diff = difflib.unified_diff(
				code_before_opt.splitlines(True),
				optimized_file_content.splitlines(True),
				fromfile=f"a/{os.path.basename(kernel_file)}",
				tofile=f"b/{os.path.basename(kernel_file)}",
			)
			self.last_applied_diff = "".join(list(diff))

			with open(kernel_file, "w") as f:
				f.write(optimized_file_content)

			if self.output_kernel_file and self.current_iteration == 1:
				with open(self.output_kernel_file, "w") as f:
					f.write(f"--- MEMORY ANALYSIS ---\n")
					f.write(f"{self.memory_analysis_output}\n\n")
					f.write(f"--- MEMORY ANALYSIS REASONING ---\n")
					f.write(f"{self.memory_analysis_reasoning}\n\n")

			if self.output_kernel_file:
				with open(self.output_kernel_file, "a") as f:
					f.write(f"Iteration {self.current_iteration}:\n")
					f.write("Code optimization reasoning:\n")
					f.write(f"{self.optimization_reasoning or 'Not provided.'}\n\n")
					f.write("Reason why old was slow:\n")
					f.write(f"{response.reason_why_old_was_slow}\n\n")
					f.write("Summary of optimization:\n")
					f.write(f"{response.summary_of_optimization}\n\n")
					f.write("Reason why new should be better:\n")
					f.write(f"{response.reason_why_new_should_be_better}\n\n")
					f.write("Swizzling formula:\n")
					f.write(f"{self.current_swizzling_pattern}\n\n")

			logging.debug(f"Optimized file content: {optimized_file_content}")
			return Result(
				success=True,
				asset={
					"optimized_code_path": kernel_file,
					"optimized_code_string": optimized_file_content,
				},
			)
		except Exception as e:
			logging.error(f"An unexpected error occurred - {str(e)}")
			return Result(success=False, error_report=f"An unexpected error occurred - {str(e)}")

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
		        accordo_absolute_tolerance (float): The absolute tolerance for the Accordo validation

		Returns:
		    Result: Validation status
		"""
		return Result(success=True, asset={"log": "Correctness validation pass not implemented for swizzling."})
		result = super().correctness_validation_pass(self.current_kernel, self.current_args, accordo_absolute_tolerance)
		if not result:
			self.current_summary = result.error_report
		return result

	def performance_validation_pass(self) -> Result:
		unoptimized_results = filter_json_field(
			self._initial_profiler_results,
			field="kernel",
			comparison_func=lambda x: x == self.current_kernel_signature,
		)

		unoptimized_time = unoptimized_results[0]["durations"]["ns"]
		unoptimized_l2_hit_rate = unoptimized_results[0]["l2"]["hr"]
		kernel = unoptimized_results[0]["kernel"]

		# Profile the optimized application
		self._optimization_results = self._application.profile(top_n=self.top_n)

		optimized_results = filter_json_field(
			self._optimization_results,
			field="kernel",
			comparison_func=lambda x: x == kernel,
		)

		optimized_time = optimized_results[0]["durations"]["ns"]
		optimized_l2_hit_rate = optimized_results[0]["l2"]["hr"]

		success = optimized_l2_hit_rate > unoptimized_l2_hit_rate
		speedup = unoptimized_time / optimized_time
		l2_improvement = optimized_l2_hit_rate - unoptimized_l2_hit_rate

		self.optimization_report = ""
		self.l2_improvement_history.append(l2_improvement)

		# Format the L2 cache improvement message
		if l2_improvement > 0:
			self.optimization_report += (
				f"L2 Cache Locality Improvement: Successfully improved L2 cache hit rate by "
				f"{l2_improvement:.2f} percentage points. "
				f"Hit rate increased from {unoptimized_l2_hit_rate:.1f}% to {optimized_l2_hit_rate:.1f}% "
				f"(higher percentages indicate better cache locality through improved block swizzling). "
			)
		else:
			self.optimization_report += (
				f"L2 Cache Locality Degradation: L2 cache hit rate decreased by "
				f"{abs(l2_improvement):.2f} percentage points. "
				f"Hit rate decreased from {unoptimized_l2_hit_rate:.1f}% to {optimized_l2_hit_rate:.1f}% "
				f"(lower percentages indicate worse cache locality). "
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

		self.iteration_history.append(
			{
				"iteration": self.current_iteration,
				"diff": self.last_applied_diff,
				"report": self.optimization_report,
				"success": success,
			}
		)

		if l2_improvement > self.best_l2_improvement:
			self.best_l2_improvement = l2_improvement
			self.best_speedup = speedup
			self.best_diff = self.last_applied_diff
			self.best_iteration_report = self.optimization_report
			with open(self.current_kernel_files[0], "r") as f:
				self.best_kernel_code = f.read()
				self.best_swizzling_pattern = self.current_swizzling_pattern

		if self.output_kernel_file:
			with open(self.output_kernel_file, "a") as f:
				f.write(f"--- PROFILING ITERATION {self.current_iteration} ---\n")
				f.write(f"{self.optimization_report}\n\n")

		terminate = False
		if self.current_iteration >= self.max_iterations:
			logging.warning(f"Max iterations reached ({self.max_iterations}). Terminating optimization.")
			terminate = True

		self.success = self.best_l2_improvement > 0
		
		if self.success and self.output_kernel_file:
			name, ext = os.path.splitext(self.output_kernel_file)
			final_output_path = f"{name}_final{ext}"
			with open(final_output_path, "w") as f:
				f.write(f"L2 Hit Rate Improvement %: {self.best_l2_improvement}\n")
				f.write(f"Speedup: {self.best_speedup}\n\n")
				f.write(f"Swizzling Pattern:\n")
				f.write(f"[[[{self.best_swizzling_pattern}]]]\n")
				f.write("Full Kernel Code:\n")
				f.write(f"[[[{self.best_kernel_code}]]]\n")
		if self.current_iteration < self.max_iterations:
			self.current_summary = self.optimization_report
			# Always return success=False to continue iterating
			return Result(success=False, error_report=self.best_iteration_report)
		
		return Result(success=True, asset={"log": self.best_iteration_report})

	def write_results(self, output_file: str = None):
		"""
		Writes the results to the output file.
		"""
		if self.success and self.best_kernel_code:
			output_dir = os.path.join(self.project_directory, "outputted_optimizations")
			os.makedirs(output_dir, exist_ok=True)
			
			# Sanitize kernel name to be a valid filename
			kernel_filename = "".join(c if c.isalnum() or c in ('_') else '_' for c in self.best_kernel_name)
			output_path = os.path.join(output_dir, f"{kernel_filename}.py")

			with open(output_path, "w") as f:
				f.write("#!/usr/bin/env python3\n")
				f.write(self.best_kernel_code)

		super().write_results(
			output_file=output_file,
			additional_results={"formula": "swizzling", "success": self.success},
		)

	def summarize_previous_passes(self):
		"""
		Summarizes the results of the previous passes for future prompts.
		"""
		pass
