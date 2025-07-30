################################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
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

from intelliperf.core.gpu_spec import GPUSpec
from intelliperf.core.llm import LLM
from intelliperf.formulas.formula_base import (
	Formula_Base,
	Result,
	filter_json_field,
	get_kernel_name,
)
from intelliperf.utils.env import get_llm_api_key


class swizzling_baseline(Formula_Base):
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
        **kwargs
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
		self.success = False
		self.gpu_spec = GPUSpec()

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
			error_report="The instrumentation is not implemented for swizzling baseline.",
		)

	def optimize_pass(self, temperature: float = 0.0, max_tokens: int = 3000) -> Result:
		"""
		Optimize the kernel to improve l2 hit rate through block swizzling via LLM

		Args:
		        temperature (float): Sampling temperature for OpenAI API
		        max_tokens (int): Maximum tokens for OpenAI API

		Returns:
		        Result: Optimized kernel as a file path
		"""
		super().optimize_pass()
		llm_key = get_llm_api_key()

		system_prompt = (
			"You are a skilled GPU programmer specializing in block swizzling optimization. "
			"Given a kernel, you will implement swizzling to improve L2 cache locality. "
			"Do not modify the kernel signature. Do not touch any other code, licenses, copyrights, or comments in the file. "
			"If you remove the copyright, your solution will be rejected. "
			"Do not include any markdown code blocks or text other than the code."
		)

		provider = self.provider
		model = self.model
		llm = LLM(
			api_key=llm_key,
			system_prompt=system_prompt,
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
			logging.debug(f"Kernel: {kernel}")
			self.current_args = kernel.split("(")[1].split(")")[0].split(",")
			files = filtered_report_card[0]["source"]["files"]
			kernel_name = get_kernel_name(kernel)

			logging.debug(f"Kernel name: {kernel_name}")
			kernel_file = None
			for file in files:
				if os.path.exists(file):
					with open(file, "r") as f:
						unoptimized_file_content = f.read()
						if kernel_name in unoptimized_file_content:
							kernel_file = file
							self.current_kernel = kernel_name
							break
			if kernel_file is None:
				return Result(
					success=False,
					error_report=f"Kernel file not found for kernel {kernel}.",
				)

			user_prompt = (
				f"The kernel {kernel} in the source code below has a low L2 cache hit rate. "
				"Please apply block swizzling to improve L2 cache locality. Do not change the semantics of the program.\n\n"
				f"{unoptimized_file_content}"
			)

			if self.current_summary is not None:
				user_prompt += f"\n\nThe current summary is: {self.current_summary}"
				cur_diff = self.compute_diff([kernel_file])
				user_prompt += f"\nThe diff between the current and initial code is: {cur_diff}"

			self.previous_source_code = unoptimized_file_content

			if self.current_args:
				self.bottleneck_report = (
					f"L2 Cache Locality Detection: IntelliPerf identified suboptimal L2 cache hit rate "
					f"`{self.current_kernel}` with arguments `{self.current_args}` - Suboptimal swizzling leads to low L2 hit rate."
				)
			else:
				self.bottleneck_report = (
					f"L2 Cache Locality Detection: IntelliPerf identified suboptimal L2 cache hit rate "
					f"`{self.current_kernel}` - Suboptimal swizzling leads to low L2 hit rate."
				)
		else:
			pass

		if kernel is None:
			return Result(success=False, error_report="Failed to extract the kernel name.")
		if kernel_file is None:
			return Result(success=False, error_report="Failed to extract the kernel file path.")

		logging.debug(f"System prompt: {system_prompt}")
		logging.debug(f"LLM prompt: {user_prompt}")
		
		self.current_kernel_signature = kernel

		self.current_kernel_files = [kernel_file]
		try:
			optimized_file_content = llm.ask(user_prompt).strip()
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
		result = super().correctness_validation_pass(
			self.current_kernel, self.current_args, accordo_absolute_tolerance
		)
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

		if not success or speedup < 1:
			self.current_summary = self.optimization_report
			return Result(success=False, error_report=self.optimization_report)
		logging.info(self.optimization_report)

		self.success = True
		return Result(success=True, asset={"log": self.optimization_report})

	def write_results(self, output_file: str = None):
		"""
		Writes the results to the output file.
		"""
		super().write_results(
			output_file=output_file,
			additional_results={"formula": "swizzling_baseline", "success": self.success},
		)

	def summarize_previous_passes(self):
		"""
		Summarizes the results of the previous passes for future prompts.
		"""
		pass
