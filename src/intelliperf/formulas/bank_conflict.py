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

from intelliperf.core.llm import LLM
from intelliperf.formulas.formula_base import (
	Formula_Base,
	Result,
	filter_json_field,
	get_kernel_name,
)
from intelliperf.utils.env import get_llm_api_key
from intelliperf.utils.process import capture_subprocess_output
from intelliperf.utils.regex import generate_ecma_regex_from_list


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

		# This temp option allows us to toggle if we want a full or partial instrumentation report
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
		Profile the application using guided-tuning and collect bank conflict data

		Returns:
		    Result: DataFrame containing the performance report card
		"""
		return super().profile_pass()

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

	def optimize_pass(self, temperature: float = 0.0, max_tokens: int = 3000) -> Result:
		"""
		Optimize the kernel to remove shared memory bank conflicts via OpenAI API

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
			" you will optimize it to remove shared memory bank conflicts"
			" and provide a correct performant implementation. Do not modify"
			" the kernel signature. Do not touch any other code, licenses, copyrights, or comments in the file."
			" If you remove the copyright, your solution will be rejected."
			" Do not include any markdown code blocks or text other than the code."
		)

		provider = self.provider
		model = self.model
		llm = LLM(
			api_key=llm_key,
			system_prompt=system_prompt,
			model=model,
			provider=provider,
		)

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
			)

			if len(filtered_report_card) == 0:
				return Result(success=False, error_report="No bank conflicts found.")

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
				f"There is a bank conflict in the kernel {kernel} in the source code {unoptimized_file_content}."
				f" Please fix the conflict but do not change the semantics of the program."
				" Do not remove any comments or licenses."
				" Do not include any markdown code blocks or text other than the code."
			)
			if self.current_summary is not None:
				user_prompt += f"\n\nThe current summary is: {self.current_summary}"
				cur_diff = self.compute_diff([kernel_file])
				user_prompt += f"\nThe diff between the current and initial code is: {cur_diff}"

			self.previous_source_code = unoptimized_file_content

			args = kernel.split("(")[1].split(")")[0]
			self.bottleneck_report = (
				f"Bank Conflict Detection: IntelliPerf identified shared memory bank conflicts in kernel "
				f"`{kernel_name}` with arguments `{args}`. Bank conflicts occur when multiple threads "
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
		logging.debug(f"LLM prompt: {user_prompt}")

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
			self.current_summary = result.error_report
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
			additional_results={"formula": "bankConflict", "success": self.success},
		)

	def summarize_previous_passes(self):
		"""
		Summarizes the results of the previous passes for future prompts.
		"""
		pass
