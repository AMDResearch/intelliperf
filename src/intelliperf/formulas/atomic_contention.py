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

from intelliperf.core.llm import LLM
from intelliperf.formulas.formula_base import (
	Formula_Base,
	Result,
	filter_json_field,
	get_kernel_name,
)
from intelliperf.utils.env import get_llm_api_key


class atomic_contention(Formula_Base):
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

	def profile_pass(self) -> Result:
		"""
		Profile the application using guided-tuning and collect atomic contention data

		Returns:
		    Result: DataFrame containing the performance report card
		"""
		return super().profile_pass()

	def instrument_pass(self) -> Result:
		"""
		Instrument the application, targeting the kernels with the highest atomic contention data

		Returns:
		    Result: Instrumentation data containing the kernel name, arguments, lines, and file path as dict
		"""
		super().instrument_pass()

		return Result(
			success=False,
			asset=self._instrumentation_results,
			error_report="Instrumentation pass not implemented for atomic contention.",
		)

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
			logging.debug(f"Setting current summary to: {result.error_report}")
			self.current_summary = result.error_report
		return result

	def optimize_pass(self, temperature: float = 0.0, max_tokens: int = 3000) -> Result:
		"""
		Optimize the kernel to remove atomic contention via OpenAI API

		Args:
		    temperature (float): Sampling temperature for OpenAI API
		    max_tokens (int): Maximum tokens for OpenAI API

		Returns:
		    Result: Optimized kernel as a file path
		"""
		super().optimize_pass()
		llm_key = get_llm_api_key()

		system_prompt = (
			"You are a skilled GPU HIP programmer. Given a kernel,"
			" you will optimize it to remove atomic contention as much as possible"
			" and provide a correct performant implementation. Do not modify"
			" the kernel signature. Do not touch any other code, licenses, copyrights, or comments in the file."
			" If you remove the copyright, your solution will be rejected."
			" Do not include any markdown code blocks or text other than the code."
		)

		model = self.model
		provider = self.provider
		llm = LLM(
			api_key=llm_key,
			system_prompt=system_prompt,
			model=model,
			provider=provider,
		)

		kernel = None
		kernel_file = None

		if self._instrumentation_results is None:
			# Get the file from the results
			field = "atomics"
			subfield = "atomic_lat"
			# Average atomic latency in cycles measured experimentally
			average_atomic_lat = 1000
			filtered_report_card = filter_json_field(
				self._initial_profiler_results,
				field=field,
				subfield=subfield,
				comparison_func=lambda x: x > average_atomic_lat,
			)

			if len(filtered_report_card) == 0:
				return Result(success=False, error_report="No atomic contention found.")

			logging.debug(f"Filtered Report Card:\n{json.dumps(filtered_report_card, indent=4)}")

			kernel = filtered_report_card[0]["kernel"]
			files = filtered_report_card[0]["source"]["files"]
			kernel_name = get_kernel_name(kernel)
			kernel_file = None
			for file in files:
				if os.path.exists(file):
					with open(file, "r") as f:
						unoptimized_file_content = f.read()
						if kernel_name in unoptimized_file_content:
							kernel_file = file
							break
			if kernel_file is None:
				return Result(success=False, error_report="Kernel file not found.")

			user_prompt = (
				f"There is atomic contention in the kernel {kernel} in the source code {unoptimized_file_content}."
				f" Please fix the contention but do not change the semantics of the program."
				" Do not remove any comments or licenses."
				" Do not include any markdown code blocks or text other than the code."
			)
			if self.current_summary is not None:
				user_prompt += f"\n\nThe current summary is: {self.current_summary}"

				# Split the strings into lines for proper diff computation
				cur_diff = self.compute_diff([kernel_file])

				user_prompt += f"\nThe diff between the current and initial code is: {cur_diff}"

			self.previous_source_code = unoptimized_file_content

			args = kernel.split("(")[1].split(")")[0]
			self.bottleneck_report = (
				f"Atomic Contention Detection: IntelliPerf identified high atomic contention in kernel "
				f"`{kernel_name}` with arguments `{args}`. Atomic contention occurs when multiple threads "
				f"compete for the same atomic operations, causing serialization and increased latency."
			)

		else:
			pass

		if kernel is None:
			return Result(success=False, error_report="Failed to extract the kernel name.")
		if kernel_file is None:
			return Result(success=False, error_report="Failed to extract the kernel file path.")

		self.current_kernel_files = [kernel_file]

		logging.debug(f"LLM prompt: {user_prompt}")
		logging.debug(f"System prompt: {system_prompt}")

		self.current_kernel = kernel.split("(")[0]
		self.current_args = kernel.split("(")[1].split(")")[0].split(",")
		self.current_kernel_signature = kernel
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
		    accordo_absolute_tolerance (float): Absolute tolerance for Accordo
		Returns:
		    Result: Validation status
		"""
		result = super().correctness_validation_pass(self.current_kernel, self.current_args, accordo_absolute_tolerance)
		if not result:
			logging.info(f"Setting current summary to: {result.error_report}")
			self.current_summary = result.error_report
		return result

	def performance_validation_pass(self) -> Result:
		"""
		Validate the optimized kernel by comparing the output with the reference kernel

		Returns:
		    Result: Validation status
		"""
		unoptimized_results = filter_json_field(
			self._initial_profiler_results,
			field="kernel",
			comparison_func=lambda x: x == self.current_kernel_signature,
		)

		unoptimized_time = unoptimized_results[0]["durations"]["ns"]

		field = "atomics"
		subfield = "atomic_lat"
		unoptimized_metric = unoptimized_results[0][field][subfield]

		# Profile the optimized application
		self._optimization_results = self._application.profile(top_n=self.top_n)

		optimized_results = filter_json_field(
			self._optimization_results,
			field="kernel",
			comparison_func=lambda x: x == self.current_kernel_signature,
		)

		optimized_time = optimized_results[0]["durations"]["ns"]
		optimized_metric = optimized_results[0][field][subfield]

		success = optimized_metric < unoptimized_metric
		speedup = unoptimized_time / optimized_time
		metric_improvement = unoptimized_metric / optimized_metric if optimized_metric != 0 else 1

		# Calculate cycle latency improvement percentage
		cycle_latency_improvement = (
			(unoptimized_metric - optimized_metric) / unoptimized_metric * 100 if unoptimized_metric > 0 else 0
		)
		self.optimization_report = ""

		# Format the atomic contention improvement message
		if metric_improvement > 1:
			self.optimization_report += (
				f"Atomic Contention Reduction: Successfully reduced atomic contention by "
				f"{metric_improvement:.2f}x. "
				f"Average atomic latency improved from {unoptimized_metric:.0f} to {optimized_metric:.0f} cycles "
				f"({cycle_latency_improvement:.1f}% reduction - lower latency means less contention). "
			)
		else:
			self.optimization_report += (
				f"Atomic Contention Increase: Atomic contention increased by "
				f"{1 / metric_improvement:.2f}x. "
				f"Average atomic latency worsened from {unoptimized_metric:.0f} to {optimized_metric:.0f} cycles "
				f"({abs(cycle_latency_improvement):.1f}% increase - higher latency means more contention). "
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
			additional_results={"formula": "atomicContention", "success": self.success},
		)

	def summarize_previous_passes(self):
		"""
		Summarizes the results of the previous passes for future prompts.
		"""
		pass
