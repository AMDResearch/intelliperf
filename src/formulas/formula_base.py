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
import sys
import time
from abc import abstractmethod
from pprint import pformat

import ml_dtypes
import numpy as np
import pandas as pd
from core.application import Application
from utils.process import exit_on_fail

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from accordo.python.code_gen import generate_header
from accordo.python.communicate import get_kern_arg_data, send_response
from accordo.python.utils import run_subprocess


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
			logging.info(self.log)
			if self.asset is not None:
				for asset in self.asset:
					if isinstance(asset, pd.DataFrame):
						logging.info("\n%s", asset.to_string(index=False))
					elif isinstance(asset, dict):
						logging.info("\n%s", json.dumps(asset, indent=2))
					else:
						logging.info("\n%s", pformat(asset))
		else:
			logging.error(f"Error: {self.error_report}")
			sys.exit(1)


class Formula_Base:
	def __init__(
		self,
		name: str,
		build_command: list,
		instrument_command: list,
		project_directory: str,
		app_cmd: list,
		top_n: int,
	):
		# Private
		self.__name = name  # name of the run
		self._application = Application(name, build_command, instrument_command, project_directory, app_cmd)

		self._initial_profiler_results = None

		# Public
		self.profiler: str = None
		self.top_n: int = top_n

		self.build()

	def build(self):
		if not self._application.get_build_command():
			return Result(success=True, asset={"log": "No build script provided. Skipping build step."})
		else:
			success, result = self._application.build()
			# Handle critical error
			exit_on_fail(success=success, message=f"Failed to build {self.__name} application.", log=result)
		return Result(success=success, asset={"log": result})

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

		accordo_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../", "accordo"))

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
			results[label] = get_kern_arg_data(pipe_name, kernel_args, ipc_file_name)
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

		for i in range(len(results[key0])):
			if not validate_arrays(results[key0][i], results[key1][i], accordo_absolute_tolerance):
				return Result(
					success=False, error_report=f"Arrays at index {i} for '{key0}' and '{key1}' are NOT close."
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
			empty = {"assembly": [], "files": [], "hip": [], "lines": [], "signature": ""}
			entry["source"] = df_results["kernels"].get(kernel_name, empty)

		return Result(success=True, asset=self._initial_profiler_results)

	@abstractmethod
	def summarize_previous_passes(self):
		"""
		Summarizes the results of the previous passes for future prompts.
		"""
		pass


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
