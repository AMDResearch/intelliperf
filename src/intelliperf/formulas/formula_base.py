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

import difflib
import json
import logging
import os
import sys
import time
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from pprint import pformat
from typing import List, Optional

import ml_dtypes
import numpy as np
import pandas as pd

from accordo import AccordoValidator, KernelArg, ValidationConfig
from intelliperf import __version__
from intelliperf.core.application import Application
from intelliperf.core.logger import Logger
from intelliperf.utils.process import capture_subprocess_output, exit_on_fail


@dataclass
class OptimizationStep:
	"""Represents a single optimization attempt"""

	iteration: int
	diff: str
	report: str
	metrics: dict
	success: bool
	timestamp: float = field(default_factory=time.time)

	def get_metric(self, key: str, default=0.0):
		"""Helper to safely get metrics"""
		return self.metrics.get(key, default)


class OptimizationTracker:
	"""Tracks optimization history using dspy.History for proper conversation management"""

	def __init__(
		self,
		max_iterations: int = 10,
		primary_metric: str = "speedup",
		maximize: bool = True,
		before_metric: Optional[str] = None,
		after_metric: Optional[str] = None,
	):
		self.steps: List[OptimizationStep] = []
		self.current_iteration: int = 0
		self.max_iterations: int = max_iterations
		self.best_step: Optional[OptimizationStep] = None
		self.primary_metric: str = primary_metric
		self.maximize: bool = maximize
		self.initial_source_code: Optional[str] = None
		# For auto-calculating improvements from before/after metrics
		self.before_metric: Optional[str] = before_metric
		self.after_metric: Optional[str] = after_metric

		# History messages for DSPy (stored as list of dicts)
		self.history_messages = []

	def add_step(
		self,
		diff: str,
		report: str,
		metrics: dict,
		success: bool,
		request: str = "",
		optimized_code: str = "",
	) -> OptimizationStep:
		"""Add step and auto-update best based on primary metric"""
		# Auto-calculate improvement if before/after metrics are configured
		if self.before_metric and self.after_metric:
			before = metrics.get(self.before_metric, 0)
			after = metrics.get(self.after_metric, 0)
			if before != 0 and after != 0:
				# Calculate improvement ratio based on whether we're maximizing or minimizing the raw metric
				# For conflicts/latency (maximize=False, lower is better): improvement = before / after
				#   Example: 3.5 conflicts → 1.0 conflicts = 3.5 / 1.0 = 3.5x improvement
				# For coalescing (maximize=True, higher is better): improvement = after / before
				#   Example: 50% → 75% = 75 / 50 = 1.5x improvement
				improvement = before / after if not self.maximize else after / before
				metrics[self.primary_metric] = improvement
			elif before != 0:
				# If after is 0, set improvement to 1.0 (no change)
				metrics[self.primary_metric] = 1.0

		step = OptimizationStep(
			iteration=self.current_iteration,
			diff=diff,
			report=report,
			metrics=metrics,
			success=success,
		)
		self.steps.append(step)
		self.current_iteration += 1

		# Add to DSPy history with explicit counters for better LLM learning
		# Extract common metrics for structured history
		history_entry = {
			"iteration": self.current_iteration - 1,
			"request": request,
			"optimized_code": optimized_code,
			"result_summary": report,
			"diff": diff,
			"success": "✓ Improved" if success else "✗ Regressed",
		}

		# Add explicit before/after counters if available
		if self.before_metric and self.after_metric:
			before_val = metrics.get(self.before_metric, 0)
			after_val = metrics.get(self.after_metric, 0)
			improvement_val = metrics.get(self.primary_metric, 1.0)

			history_entry.update(
				{
					f"before_{self.before_metric}": before_val,
					f"after_{self.after_metric}": after_val,
					f"{self.primary_metric}": improvement_val,
				}
			)

		# Add all other metrics
		history_entry["all_metrics"] = metrics

		self.history_messages.append(history_entry)

		# Auto-update best
		if self.best_step is None:
			self.best_step = step
		else:
			new_val = step.get_metric(self.primary_metric)
			cur_val = self.best_step.get_metric(self.primary_metric)

			if (self.maximize and new_val > cur_val) or (not self.maximize and new_val < cur_val):
				self.best_step = step

		return step

	def get_dspy_history(self):
		"""Get the DSPy history messages for use in signatures"""

		# Return a simple object with messages attribute for DSPy
		class HistoryWrapper:
			def __init__(self, messages):
				self.messages = messages

		return HistoryWrapper(self.history_messages)

	def has_reached_max_iterations(self) -> bool:
		"""Check if max iterations reached"""
		return self.current_iteration >= self.max_iterations

	def to_dict(self) -> dict:
		"""Serialize for JSON output"""
		return {
			"steps": [asdict(step) for step in self.steps],
			"best_step": asdict(self.best_step) if self.best_step else None,
			"current_iteration": self.current_iteration,
			"max_iterations": self.max_iterations,
			"primary_metric": self.primary_metric,
			"maximize": self.maximize,
		}


class Result:
	def __init__(self, success: bool, error_report: str = "", asset=None):
		self.success: bool = success
		# Only set error report if failure occurs
		if not self.success and error_report == "":
			logging.error("Invalid implementation of Report(). Must provide an error report if failure occurs.")
			sys.exit(1)
		self.error_report: str = error_report
		self.log: str = ""
		self.asset = asset

	def __bool__(self):
		return self.success

	def report_out(self):
		if self.success:
			logging.debug(self.log)
			if self.asset is not None:
				for asset in self.asset:
					if isinstance(asset, pd.DataFrame):
						logging.debug("\n%s", asset.to_string(index=False))
					elif isinstance(asset, dict):
						logging.debug("\n%s", json.dumps(asset, indent=2))
					else:
						logging.debug("\n%s", pformat(asset))


class Formula_Base:
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
		unittest_command: str = None,
		num_attempts: int = 10,
	):
		# Private
		self.__name = name  # name of the run
		logging.debug(f"name: {name}")
		logging.debug(f"build_command: {build_command}")
		logging.debug(f"instrument_command: {instrument_command}")
		logging.debug(f"project_directory: {project_directory}")
		logging.debug(f"app_cmd: {app_cmd}")

		# Store num_attempts
		self.num_attempts = num_attempts

		# Accordo caching (for efficient multi-iteration validation)
		self._accordo_validator = None
		self._reference_snapshot = None

		# Initialize logger
		self._logger = Logger(run_name=name)
		self._logger.record(
			"formula_init",
			{
				"name": name,
				"build_command": build_command,
				"instrument_command": instrument_command,
				"project_directory": project_directory,
				"app_cmd": app_cmd,
				"top_n": top_n,
				"model": model,
				"provider": provider,
				"in_place": in_place,
				"num_attempts": num_attempts,
			},
		)

		# Create a reference copy for comparison
		self._reference_app = Application(
			name,
			build_command,
			instrument_command,
			project_directory,
			app_cmd,
			unittest_command,
		)
		self._application = self._reference_app.clone()

		logging.debug("--------------------------------")
		self._reference_app.show_details()
		self._application.show_details()
		logging.debug("--------------------------------")

		self._reference_app.build()
		self._application.build()

		self._initial_profiler_results = None

		# Public
		self.profiler: str = None
		self.top_n: int = top_n

		self.model = model
		self.provider = provider
		self.in_place = in_place
		self.unittest_command = unittest_command
		self.current_kernel_files = []
		self.current_kernel = None
		self.current_args = None
		self.current_kernel_signature = None

		# Initialize DSPy LLM once per formula (lazily, only if needed)
		self._llm = None
		self._dspy_configured = False

		self.build()

	def get_logger(self) -> Logger:
		"""
		Get the logger instance for this formula run.

		Returns:
		        Logger: The logger instance
		"""
		return self._logger

	def get_llm(self, system_prompt: str):
		"""
		Get or create the LLM instance for this formula (lazy initialization).

		DSPy is configured once per formula instance for efficiency.

		Args:
		        system_prompt: System prompt for the LLM

		Returns:
		        LLM: The LLM instance configured for this formula
		"""
		from intelliperf.core.llm import LLM
		from intelliperf.utils.env import get_llm_api_key

		if self._llm is None:
			llm_key = get_llm_api_key()
			self._llm = LLM(
				api_key=llm_key,
				system_prompt=system_prompt,
				model=self.model,
				provider=self.provider,
				logger=self.get_logger(),
			)
			logging.debug(f"Initialized LLM once for formula: {self.model} via {self.provider}")
		return self._llm

	def _parse_kernel_signature(self, kernel_signature: str):
		"""
		Parses a kernel signature to extract the kernel name and its arguments.
		"""
		self.current_kernel_signature = kernel_signature
		if "(" in kernel_signature:
			self.current_kernel = kernel_signature.split("(")[0]
			# Safely extract arguments, handling cases with no arguments
			args_str = kernel_signature.split("(", 1)[1].rsplit(")", 1)[0]
			if args_str:
				self.current_args = [arg.strip() for arg in args_str.split(",")]
			else:
				self.current_args = []
		else:
			self.current_kernel = kernel_signature
			self.current_args = []

	def build(self, validate_build_result=True):
		if not self._application.get_build_command():
			if self._application.get_app_cmd():
				success, result = capture_subprocess_output(
					self._application.get_app_cmd(),
					working_directory=self._application.get_project_directory(),
				)
				if validate_build_result and not success:
					logging.debug(
						f"Exiting because of JIT run failure: validate_build_result={validate_build_result}, success={success}, result={result}"
					)
				if success:
					return Result(success=success, asset={"log": result})
				else:
					return Result(
						success=success,
						error_report="The application failed to run. Here is the log: " + result,
					)
			else:
				return Result(
					success=True,
					asset={"log": "No build command provided. Skipping build step."},
				)
		else:
			success, result = self._application.build()
			if validate_build_result and not success:
				logging.debug(
					f"Exiting because of build failure: validate_build_result={validate_build_result}, success={success}, result={result}"
				)
				exit_on_fail(
					success=success,
					message=f"Failed to build {self.__name} application.",
					log=result,
				)

		if success:
			return Result(success=success, asset={"log": result})
		else:
			return Result(
				success=success,
				error_report="The application contains compiler errors. Here is the compiler log: " + result,
			)

	# ----------------------------------------------------
	# Required methods to be implemented by child classes
	# ----------------------------------------------------
	@abstractmethod
	def profile_pass(self):
		"""
		Extract any required performance data from the application using the specified profiler.
		"""
		self._initial_profiler_results = self._application.profile(top_n=self.top_n)

		logging.debug(f"Initial profiler results: {json.dumps(self._initial_profiler_results, indent=2)}")

	@abstractmethod
	def instrument_pass(self):
		"""
		Instrument elements of the application to pinpoint source of bottleneck.
		"""
		self._application.build(instrumented=True)

	@abstractmethod
	def optimize_pass(self, target_kernel: str = None):
		"""
		Optimize the application based on the data collected from the instrumentation pass.
		"""
		pass

	@abstractmethod
	def correctness_validation_pass(self, kernel, kernel_args, accordo_absolute_tolerance: float = 1e-6):
		"""
		Validates the application using Accordo.

		Uses snapshot caching: reference app is captured once on first call,
		then each optimized version is captured and compared to the cached reference.
		"""
		if self.unittest_command:
			success, output = self._application.run_unit_test()
			if not success:
				return Result(
					success=False,
					error_report=f"Unit test validation failed. Output:\n{output}",
				)
			return Result(success=True, asset={"log": output})

		# Build the optimized application first (Accordo doesn't build)
		self._application.build()

		# Create validator if not already created (and cache it)
		if self._accordo_validator is None:
			kernel_arg_objects = [KernelArg(name=f"arg{i}", type=arg_type) for i, arg_type in enumerate(kernel_args)]

			config = ValidationConfig(
				kernel_name=kernel,
				kernel_args=kernel_arg_objects,
				tolerance=accordo_absolute_tolerance,
				timeout_multiplier=2.0,
			)

			self._accordo_validator = AccordoValidator(config)
			logging.debug("Created and cached Accordo validator")

		# Capture reference snapshot if not already captured (and cache it)
		if self._reference_snapshot is None:
			try:
				reference_binary = self._reference_app.get_app_cmd()
				working_dir = self._reference_app.get_project_directory()

				logging.debug("Capturing reference snapshot (will be cached)")
				self._reference_snapshot = self._accordo_validator.capture_snapshot(
					binary=reference_binary, working_directory=working_dir, timeout_seconds=30
				)
				logging.debug(f"Reference snapshot captured in {self._reference_snapshot.execution_time_ms:.2f}ms")
			except Exception as e:
				logging.error(f"Failed to capture reference snapshot: {str(e)}")
				return Result(success=False, error_report=f"Failed to capture reference snapshot: {str(e)}")

		# Capture optimized snapshot and compare with cached reference
		try:
			baseline_time = getattr(self, "baseline_time_ms", None)
			optimized_binary = self._application.get_app_cmd()
			working_dir = self._application.get_project_directory()

			# Calculate timeout for optimized
			if baseline_time:
				opt_timeout = int((baseline_time * 2.0 / 1000.0) + 30.0)
			else:
				opt_timeout = 30

			logging.debug("Capturing optimized snapshot")
			optimized_snapshot = self._accordo_validator.capture_snapshot(
				binary=optimized_binary, working_directory=working_dir, timeout_seconds=opt_timeout
			)
			logging.debug(f"Optimized snapshot captured in {optimized_snapshot.execution_time_ms:.2f}ms")

			# Compare snapshots
			validation_result = self._accordo_validator.compare_snapshots(self._reference_snapshot, optimized_snapshot)

			if validation_result.is_valid:
				logging.debug("Validation succeeded.")
				return Result(success=True)
			else:
				return Result(success=False, error_report=validation_result.error_message)

		except Exception as e:
			logging.error(f"Accordo validation error: {str(e)}")
			return Result(success=False, error_report=f"Accordo validation error: {str(e)}")

	@abstractmethod
	def performance_validation_pass(self):
		"""
		Validates the performance of the application.
		"""
		pass

	@abstractmethod
	def source_code_pass(self):
		"""
		Finds the source code.
		"""
		df_results = self._application.collect_source_code()

		# In-place append of source info
		for entry in self._initial_profiler_results:
			kernel_name = entry["kernel"]
			empty = {
				"assembly": [],
				"files": [],
				"hip": [],
				"lines": [],
				"signature": "",
			}

			# Try adding the kd suffix
			if kernel_name not in df_results["kernels"]:
				kernel_name = kernel_name + ".kd"
			entry["source"] = df_results["kernels"].get(kernel_name, empty)

		logging.debug(f"results with source code info: {json.dumps(self._initial_profiler_results, indent=2)}")

		return Result(success=True, asset=self._initial_profiler_results)

	@abstractmethod
	def summarize_previous_passes(self):
		"""
		Summarizes the results of the previous passes for future prompts.
		"""
		pass

	def postprocess_llm_code(self, optimized_file_content: str) -> str:
		"""
		Post-process the LLM generated code before writing to file.

		Removes markdown code blocks (```c++, ```python, etc.) from the LLM response.

		Args:
		        optimized_file_content (str): The LLM generated code

		Returns:
		        str: The post-processed code
		"""
		# Remove markdown code blocks if present
		content = optimized_file_content.strip()
		if content.startswith("```") and content.endswith("```"):
			# Remove the opening ``` and any language identifier
			content = content[3:]  # Remove opening ```
			# Find the first newline or end of string
			first_newline = content.find("\n")
			if first_newline != -1:
				content = content[first_newline + 1 :]  # Skip language identifier line
			# Remove the closing ```
			if content.endswith("```"):
				content = content[:-3]

		return content

	def write_and_log_optimized_code(self, kernel_file: str, optimized_code: str) -> None:
		"""
		Write optimized code to file and log it for future reference.

		This should be called immediately after LLM generates code, regardless of whether
		it will compile or pass validation.

		Args:
		    kernel_file: Path to the kernel file to write
		    optimized_code: The optimized code content
		"""
		# Write the code to the kernel file
		with open(kernel_file, "w") as f:
			f.write(optimized_code)

		# Automatically detect iteration number from optimization tracker
		iteration_num = len(self.optimization_tracker.steps) if hasattr(self, "optimization_tracker") else 0

		# Log the iteration code immediately
		kernel_name = get_kernel_name(
			self.current_kernel_signature if hasattr(self, "current_kernel_signature") else "kernel"
		)
		self.get_logger().save_iteration_code(kernel_name, iteration_num, optimized_code)

		logging.debug(f"Wrote and logged optimized code to {kernel_file} (iteration {iteration_num})")

	def find_kernel_file(self, files: list, kernel: str) -> tuple:
		"""
		Find the kernel file containing the given kernel name from a list of files.

		Args:
		        files: List of file paths to search
		        kernel: Kernel signature to find

		Returns:
		        tuple: (kernel_file_path, file_content) or (None, None) if not found

		Note:
		        - Validates files exist and are within the project directory
		        - Logs warnings for invalid files
		        - Exits with sys.exit(1) if kernel file not found after checking all files
		"""
		kernel_name = get_kernel_name(kernel)
		logging.debug(f"Searching for kernel: {kernel_name}")

		kernel_file = None
		unoptimized_file_content = None
		project_dir = os.path.abspath(self._application.get_project_directory())

		for file in files:
			file_path = os.path.abspath(file)

			# Check if file exists
			if not os.path.exists(file):
				logging.warning(f"File {file} does not exist")
				continue

			# Check if file is in project directory
			try:
				isfile_in_project = os.path.commonpath([project_dir, file_path]) == project_dir
			except ValueError:
				# Happens when paths are on different drives (Windows)
				isfile_in_project = False

			if not isfile_in_project:
				logging.warning(f"File {file} is not in the project")
				continue

			# Try to read file and find kernel
			try:
				with open(file, "r") as f:
					unoptimized_file_content = f.read()
					if kernel_name in unoptimized_file_content:
						kernel_file = file
						break
			except Exception as e:
				logging.error(f"Error reading file {file}: {e}")
				continue

		# If kernel file not found, log error and exit
		if kernel_file is None:
			logging.error(f"Kernel file not found for kernel {kernel}")
			logging.error(f"Kernel name: {kernel_name}")
			logging.error(f"Files searched: {files}")
			if unoptimized_file_content:
				logging.error(f"Last file content (first 200 chars): {unoptimized_file_content[:200]}")
			sys.exit(1)

		return kernel_file, unoptimized_file_content

	def compute_diff(self, filepaths: list[str]) -> str:
		diffs = []
		for filepath in filepaths:
			# Extract relative path from the full filepath
			# If filepath is already relative to project directory, this will work correctly
			# If filepath is absolute, we need to make it relative to the project directory
			reference_project_dir = self._reference_app.get_project_directory()
			optimized_project_dir = self._application.get_project_directory()

			# If filepath is absolute, make it relative to the optimized project directory
			if os.path.isabs(filepath):
				# Get the relative path from the optimized project directory
				relative_path = os.path.relpath(filepath, optimized_project_dir)
			else:
				# filepath is already relative
				relative_path = filepath

			reference_filepath = os.path.join(reference_project_dir, relative_path)
			optimized_filepath = os.path.join(optimized_project_dir, relative_path)

			with open(reference_filepath, "r") as f:
				prev_lines = f.read().splitlines(keepends=True)
			with open(optimized_filepath, "r") as f:
				curr_lines = f.read().splitlines(keepends=True)
			cur_diff = difflib.unified_diff(prev_lines, curr_lines)
			cur_diff = "".join(cur_diff)
			diffs.append(cur_diff)
		return "\n".join(diffs)

	def inplace_update(self, filepaths: list[str]):
		"""
		Updates the source code in place.
		"""
		for filepath in filepaths:
			relative_path = os.path.relpath(filepath, self._application.get_project_directory())
			reference_filepath = os.path.join(self._reference_app.get_project_directory(), relative_path)
			optimized_filepath = os.path.join(self._application.get_project_directory(), relative_path)
			with open(optimized_filepath, "r") as f:
				optimized_content = f.read()
			with open(reference_filepath, "w") as f:
				f.write(optimized_content)

	def write_results(
		self,
		output_file: str = None,
		additional_results: dict = {},
		diagnose_only: bool = False,
	):
		"""
		Writes the results to the output file.
		"""
		# create a new json contining optimized and unoptimized results
		if diagnose_only:
			results = {
				"version": __version__,
				"initial": self._initial_profiler_results,
				**additional_results,
			}
		else:
			results = {
				"version": __version__,
				"optimized": self._optimization_results,
				"initial": self._initial_profiler_results,
				**additional_results,
				"report_message": self.optimization_report,
				"bottleneck_report": self.bottleneck_report,
				"diff": self.compute_diff(self.current_kernel_files),
			}
			if self.in_place:
				self.inplace_update(self.current_kernel_files)
		write_results(results, output_file)


def _add_diff_lines_recursive(obj):
	"""
	Recursively add diff_lines field to any dict that contains a diff field.

	Args:
	    obj: Object to process (dict, list, or other)

	Returns:
	    Processed object with diff_lines added where applicable
	"""
	if isinstance(obj, dict):
		# Process all values in the dict recursively
		result = {}
		for key, value in obj.items():
			result[key] = _add_diff_lines_recursive(value)

		# Add diff_lines if diff exists and diff_lines doesn't
		if "diff" in result and "diff_lines" not in result:
			diff_value = result["diff"]
			if isinstance(diff_value, str):
				result["diff_lines"] = diff_value.split("\n") if diff_value else []

		return result
	elif isinstance(obj, list):
		# Process all items in the list recursively
		return [_add_diff_lines_recursive(item) for item in obj]
	else:
		# Return other types as-is
		return obj


def write_results(json_results: dict, output_file: str = None):
	"""
	Writes the results to the output file.
	"""
	log_message = f"Writing results to {output_file}" if output_file is not None else "Writing results to stdout"
	logging.info(log_message)

	# Add diff_lines to all diffs in the results recursively
	json_results = _add_diff_lines_recursive(json_results)

	if output_file is None:
		print(json.dumps(json_results, indent=2))
	elif output_file.endswith(".json"):
		with open(output_file, "w") as f:
			json.dump(json_results, f, indent=2)
	elif output_file.endswith(".csv"):
		flattened_results = [flatten_dict(entry) for entry in json_results]
		df = pd.DataFrame(flattened_results)
		df.to_csv(output_file, index=False)
	elif output_file.endswith(".txt"):
		with open(output_file, "w") as f:
			f.write(json.dumps(json_results, indent=2))
	else:
		logging.error("Invalid output file extension. Must be .json, .csv, or .txt.")
		sys.exit(1)


def flatten_dict(d, parent_key="", sep="_"):
	items = []
	for k, v in d.items():
		new_key = f"{parent_key}{sep}{k}" if parent_key else k
		if isinstance(v, dict):
			items.extend(flatten_dict(v, new_key, sep=sep).items())
		else:
			items.append((new_key, v))
	return dict(items)


def filter_json_field(d, field, subfield=None, comparison_func=lambda x: True, target_kernel=None):
	"""
	Filters a list of dictionaries based on a comparison function applied to a specified field or subfield.

	Args:
	    d (list): List of dictionaries to filter.
	    field (str): The field in each dictionary to look into.
	    subfield (str, optional): The subfield within the field to apply the comparison. Defaults to None.
	    comparison_func (function): A lambda function that takes a value and returns a boolean. Defaults to a function that always returns True.

	Returns:
	    list: A list of dictionaries that satisfy the comparison function.
	"""
	if subfield is not None:
		return [
			entry
			for entry in d
			if comparison_func(entry.get(field, {}).get(subfield, 0))
			and (target_kernel is None or get_kernel_name(entry["kernel"]) == target_kernel)
		]
	else:
		return [
			entry
			for entry in d
			if comparison_func(entry.get(field, 0))
			and (target_kernel is None or get_kernel_name(entry["kernel"]) == target_kernel)
		]


def validate_arrays(arr1, arr2, tolerance):
	"""
	Validate if two arrays are close enough, with special handling for bfloat16.

	Args:
	        arr1: First array to compare
	        arr2: Second array to compare
	        tolerance: Absolute tolerance for comparison

	Returns:
	        bool: True if arrays are close enough, False otherwise
	"""
	# Check if either array is bfloat16
	if arr1.dtype == ml_dtypes.bfloat16 or arr2.dtype == ml_dtypes.bfloat16:
		# Iterate through arrays and compare each element
		for a, b in zip(arr1, arr2):
			if abs(float(a) - float(b)) > tolerance:
				return False
		return True
	else:
		# For all other types, use regular allclose
		return np.allclose(arr1, arr2, atol=tolerance)


def get_kernel_name(kernel):
	"""
	Extracts the kernel name from the kernel signature.

	Args:
	    kernel (str): The kernel signature.

	Returns:
	    str: The kernel name.
	"""
	# Remove arguments from kernel name
	kernel_name = kernel.split("(")[0]
	# Remove template arguments from kernel name
	kernel_name = kernel_name.split("<")[0]
	# Remove namespace from kernel name
	kernel_name = kernel_name.split("::")[-1]
	return kernel_name
