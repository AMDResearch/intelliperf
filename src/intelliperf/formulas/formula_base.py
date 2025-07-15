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
from pprint import pformat

import ml_dtypes
import numpy as np
import pandas as pd

from accordo.python.code_gen import generate_header
from accordo.python.communicate import get_kern_arg_data, send_response
from accordo.python.utils import run_subprocess
from intelliperf.core.application import Application
from intelliperf.utils.env import get_accordo_path
from intelliperf.utils.process import exit_on_fail


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
	):
		# Private
		self.__name = name  # name of the run
		logging.debug(f"name: {name}")
		logging.debug(f"build_command: {build_command}")
		logging.debug(f"instrument_command: {instrument_command}")
		logging.debug(f"project_directory: {project_directory}")
		logging.debug(f"app_cmd: {app_cmd}")

		# Create a reference copy for comparison
		self._reference_app = Application(name, build_command, instrument_command, project_directory, app_cmd)
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
		self.current_kernel_files = []

		self.build()

	def build(self, validate_build_result=True):
		if not self._application.get_build_command():
			return Result(
				success=True,
				asset={"log": "No build script provided. Skipping build step."},
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
	def optimize_pass(self):
		"""
		Optimize the application based on the data collected from the instrumentation pass.
		"""
		pass

	@abstractmethod
	def correctness_validation_pass(self, kernel, kernel_args, accordo_absolute_tolerance: float = 1e-6):
		"""
		Validates the the application.
		"""
		self._application.build()

		unoptimized_binary = self._application.get_app_cmd()[0]
		optimized_binary = self._reference_app.get_app_cmd()[0]

		logging.debug(f"unoptimized_binary: {unoptimized_binary}")
		logging.debug(f"optimized_binary: {optimized_binary}")

		accordo_directory = get_accordo_path()

		results = {}
		for app, label in zip([self._reference_app, self._application], ["unoptimized", "optimized"]):
			logging.debug(f"Running accordo for {label}")
			timestamp = int(time.time())
			pipe_name = f"/tmp/kernel_pipe_{timestamp}"
			ipc_file_name = f"/tmp/ipc_handle_{timestamp}.bin"

			for file in [ipc_file_name, ipc_file_name]:
				if os.path.exists(file):
					os.remove(file)
			generate_header(kernel_args)

			run_subprocess(["cmake", "-B", "build"], accordo_directory)
			run_subprocess(["cmake", "--build", "build", "--parallel", "16"], accordo_directory)
			lib = os.path.join(accordo_directory, "build", "lib", "libaccordo.so")
			env = os.environ.copy()
			env["HSA_TOOLS_LIB"] = lib
			env["KERNEL_TO_TRACE"] = kernel

			# Get the debug level from logger and convert it
			debug_level = logging.getLogger().getEffectiveLevel()
			level_map = {
				logging.WARNING: 0,  # Warning
				logging.INFO: 1,  # Info
				logging.DEBUG: 2,  # Debug
				logging.NOTSET: 3,  # NOTEST
			}
			env["ACCORDO_LOG_LEVEL"] = str(level_map.get(debug_level, 0))  # Default to 0 (Warning) if level not found
			env["ACCORDO_PIPE_NAME"] = pipe_name
			env["ACCORDO_IPC_OUTPUT_FILE"] = ipc_file_name

			binary = app.get_app_cmd_without_args()
			binary_with_args = app.get_app_cmd()
			project_directory = app.get_project_directory()
			logging.debug(f"binary: {binary}")
			logging.debug(f"project_directory: {project_directory}")
			logging.debug(f"kernel: {kernel}")
			logging.debug(f"binary_with_args: {binary_with_args}")
			logging.debug(f"kernel_args: {kernel_args}")
			logging.debug(f"ipc_file_name: {ipc_file_name}")

			original_dir = os.getcwd()
			os.chdir(project_directory)
			os.posix_spawn(binary, binary_with_args, env)
			os.chdir(original_dir)
			try:
				results[label] = get_kern_arg_data(pipe_name, kernel_args, ipc_file_name)
			except TimeoutError as e:
				logging.error(f"Timeout while getting kernel argument data for {label}: {str(e)}")
				return Result(
					success=False,
					error_report=f"Timeout while getting kernel argument data for {label}: {str(e)}. The code may have crashed.",
				)
			send_response(pipe_name)
		logging.debug(f"results unoptimized: {results['unoptimized']}")
		logging.debug(f"results optimized: {results['optimized']}")
		key0, key1 = results.keys()
		for i in range(len(results[key0])):
			if not validate_arrays(results[key0][i], results[key1][i], accordo_absolute_tolerance):
				diff = np.abs(results[key0][i] - results[key1][i])
				logging.debug(f"Arrays at index {i} for '{key0}' and '{key1}' are NOT close.")
				logging.debug(f"  {key0}[{i}]: {results[key0][i]}")
				logging.debug(f"  {key1}[{i}]: {results[key1][i]}")
				logging.debug(f"  Difference: {diff}")
				logging.debug(f"  Max difference: {np.max(diff)}")

			else:
				argument_name = kernel_args[i]
				logging.debug(
					f"Arrays at index {i} for '{key0}' and '{key1}' are close. The argument type is '{argument_name}'."
				)
		for i in range(len(results[key0])):
			if not validate_arrays(results[key0][i], results[key1][i], accordo_absolute_tolerance):
				argument_name = kernel_args[i]
				return Result(
					success=False,
					error_report=f"The optimized code output does not match the unoptimized code output. Values at index {i} for the '{argument_name}' pointer are NOT close.",
				)
		logging.debug("Validation succeeded.")
		return Result(success=True)

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

		return Result(success=True, asset=self._initial_profiler_results)

	@abstractmethod
	def summarize_previous_passes(self):
		"""
		Summarizes the results of the previous passes for future prompts.
		"""
		pass

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

	def write_results(self, output_file: str = None, additional_results: dict = {}, diagnose_only: bool = False):
		"""
		Writes the results to the output file.
		"""
		# create a new json contining optimized and unoptimized results
		if diagnose_only:
			results = {
				"initial": self._initial_profiler_results,
				**additional_results,
			}
		else:
			results = {
				"optimized": self._optimization_results,
				"initial": self._initial_profiler_results,
				"report_message": self.optimization_report,
				"bottleneck_report": self.bottleneck_report,
				**additional_results,
				"diff": self.compute_diff(self.current_kernel_files),
			}
			if self.in_place:
				self.inplace_update(self.current_kernel_files)
		write_results(results, output_file)


def write_results(json_results: dict, output_file: str = None):
	"""
	Writes the results to the output file.
	"""
	log_message = f"Writing results to {output_file}" if output_file is not None else "Writing results to stdout"
	logging.info(log_message)

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


def filter_json_field(d, field, subfield=None, comparison_func=lambda x: True):
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
		return [entry for entry in d if comparison_func(entry.get(field, {}).get(subfield, 0))]
	else:
		return [entry for entry in d if comparison_func(entry.get(field, 0))]


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
