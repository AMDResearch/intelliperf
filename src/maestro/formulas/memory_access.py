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

from maestro.core.llm import LLM
from maestro.formulas.formula_base import Formula_Base, Result, filter_json_field, get_kernel_name, write_results
from maestro.utils.env import get_llm_api_key


class memory_access(Formula_Base):
	def __init__(
		self,
		name: str,
		build_command: list,
		instrument_command: list,
		project_directory: str,
		app_cmd: list,
		top_n: int,
		only_consider_top_kernel=False,
	):
		super().__init__(name, build_command, instrument_command, project_directory, app_cmd, top_n)

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
		Profile the application using guided-tuning and collect uncoalesced memory access data

		Returns:
		    Result: DataFrame containing the performance report card
		"""
		return super().profile_pass()

	def instrument_pass(self) -> Result:
		"""
		Instrument the application, targeting the kernels with the highest uncoalesced memory access data

		Returns:
		    Result: Instrumentation data containing the kernel name, arguments, lines, and file path as dict
		"""
		super().instrument_pass()

		return Result(
			success=False,
			asset=self._instrumentation_results,
			error_report="The instrumentation is not implemented for memory access.",
		)

	def optimize_pass(self, temperature: float = 0.0, max_tokens: int = 3000) -> Result:
		"""
		Optimize the kernel to remove uncoalesced memory access via OpenAI API

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
			" you will optimize it to remove uncoalesced memory access as much as possible"
			" and provide a correct performant implementation. Do not modify"
			" the kernel signature. Do not touch any other code, licenses, copyrights, or comments in the file."
			" If you remove the copyright, your solution will be rejected."
			" Do not include any markdown code blocks or text other than the code."
		)

		server = "https://llm-api.amd.com/azure"
		deployment_id = "dvue-aoai-001-o4-mini"
		llm = LLM(
			api_key=llm_key,
			system_prompt=system_prompt,
			deployment_id=deployment_id,
			server=server,
		)

		kernel = None
		kernel_file = None

		if self._instrumentation_results is None:
			# Get the file from the results
			field = "l1"
			subfield = "coal"
			peak_coal = 100
			filtered_report_card = filter_json_field(
				self._initial_profiler_results, field=field, subfield=subfield, comparison_func=lambda x: x < peak_coal
			)

			if len(filtered_report_card) == 0:
				return Result(success=False, error_report="No uncoalesced memory access found.")

			logging.debug(f"Filtered Report Card:\n{json.dumps(filtered_report_card, indent=4)}")

			kernel = filtered_report_card[0]["kernel"]
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
							break
			if kernel_file is None:
				return Result(success=False, error_report=f"Kernel file not found for kernel {kernel}.")

			user_prompt = (
				f"There is an uncoalesced memory access in the kernel {kernel} in the source code {unoptimized_file_content}."
				f" Please fix the access pattern but do not change the semantics of the program."
				" If you remove the copyright, your solution will be rejected."
			)
			if self.current_summary is not None:
				user_prompt += f"\n\nThe current summary is: {self.current_summary}"

				# Split the strings into lines for proper diff computation
				prev_lines = self.previous_source_code.splitlines(keepends=True)
				curr_lines = unoptimized_file_content.splitlines(keepends=True)
				cur_diff = difflib.unified_diff(prev_lines, curr_lines)
				cur_diff = "".join(cur_diff)

				logging.debug(f"Previous source code: {self.previous_source_code}")
				logging.debug(f"Unoptimized file content: {unoptimized_file_content}")
				logging.debug(f"Current diff: {cur_diff}")

				user_prompt += f"\nThe diff between the current and previous code is: {cur_diff}"

			self.previous_source_code = unoptimized_file_content

			args = kernel.split("(")[1].split(")")[0]
			self.bottleneck_report = (
				f"Maestro detected uncoalesced memory accesses in the kernel `{kernel_name}` with arguments `{args}`."
			)
		else:
			pass

		if kernel is None:
			return Result(success=False, error_report="Failed to extract the kernel name.")
		if kernel_file is None:
			return Result(success=False, error_report="Failed to extract the kernel file path.")

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
			accordo_absolute_tolerance (float): The absolute tolerance for the Accordo validation

		Returns:
		    Result: Validation status
		"""
		result = super().correctness_validation_pass(self.current_kernel, self.current_args, accordo_absolute_tolerance)
		if not result:
			self.current_summary = result.error_report
		return result

	def performance_validation_pass(self) -> Result:
		unoptimized_results = filter_json_field(
			self._initial_profiler_results, field="kernel", comparison_func=lambda x: x == self.current_kernel_signature
		)

		unoptimized_time = unoptimized_results[0]["durations"]["ns"]
		unoptimized_coal = unoptimized_results[0]["l1"]["coal"]
		kernel = unoptimized_results[0]["kernel"]

		# Profile the optimized application
		self._optimization_results = self._application.profile(top_n=self.top_n)

		optimized_results = filter_json_field(
			self._optimization_results, field="kernel", comparison_func=lambda x: x == kernel
		)

		optimized_time = optimized_results[0]["durations"]["ns"]
		optimized_coal = optimized_results[0]["l1"]["coal"]

		success = optimized_coal > unoptimized_coal
		speedup = unoptimized_time / optimized_time
		coal_improvement = optimized_coal / unoptimized_coal if optimized_coal != 0 else 1

		self.optimization_report = ""
		if coal_improvement > 1:
			self.optimization_report += f"The optimized code achieved {optimized_coal}% memory coalescing (up from {unoptimized_coal}%, {coal_improvement * 100:.3f}% improvement), "
		else:
			self.optimization_report += f"The optimized code achieved {optimized_coal}% memory coalescing (down from {unoptimized_coal}%, {coal_improvement * 100:.3f}% decrease), "

		self.optimization_report += (
			"where higher coalescing percentages indicate more efficient memory access patterns. "
		)

		if speedup > 1:
			self.optimization_report += f"The optimized implementation is {speedup:.3f}x faster overall, "
			self.optimization_report += (
				f"reducing execution time from {unoptimized_time / 1e6:.3f}ms to {optimized_time / 1e6:.3f}ms."
			)
		else:
			self.optimization_report += f"The optimized implementation is {speedup:.3f}x slower overall, "
			self.optimization_report += (
				f"increasing execution time from {unoptimized_time / 1e6:.3f}ms to {optimized_time / 1e6:.3f}ms."
			)

		if not success or speedup < 1:
			self.current_summary = self.optimization_report
			return Result(success=False, error_report=self.optimization_report)
		logging.info(self.optimization_report)

		return Result(success=True, asset={"log": self.optimization_report})

	def write_results(self, output_file: str = None):
		"""
		Writes the results to the output file.
		"""
		# create a new json contining optimized and unoptimized results
		results = {
			"optimized": self._optimization_results,
			"initial": self._initial_profiler_results,
			"report_message": self.optimization_report,
			"bottleneck_report": self.bottleneck_report,
			"formula": "memoryAccess",
		}
		write_results(results, output_file)

	def summarize_previous_passes(self):
		"""
		Summarizes the results of the previous passes for future prompts.
		"""
		pass
