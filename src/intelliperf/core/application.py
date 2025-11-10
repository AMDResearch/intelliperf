#!/usr/bin/env python3
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
import shutil
import tempfile

import pandas as pd

from intelliperf.core.profile_result import KernelMetrics, ProfileResult
from intelliperf.utils import process
from intelliperf.utils.process import capture_subprocess_output, exit_on_fail


class Application:
	def __init__(
		self,
		name: str,
		build_command: list,
		instrument_command: list,
		project_directory: str,
		app_cmd: list,
		unittest_command: list,
	):
		self.name = name
		self.build_command = None
		self.instrument_command = None
		self.app_cmd = app_cmd
		self.project_directory = project_directory
		self.unittest_command = unittest_command

		if build_command is not None:
			self.build_command = build_command if isinstance(build_command, list) else build_command.split()
		if instrument_command is not None:
			self.instrument_command = (
				instrument_command if isinstance(instrument_command, list) else instrument_command.split()
			)

	def build(self, instrumented=False):
		"""Builds the application, optionally with instrumentation."""
		if instrumented and self.instrument_command is not None:
			return process.capture_subprocess_output(
				self.instrument_command, working_directory=self.get_project_directory()
			)
		elif self.build_command is not None:
			return process.capture_subprocess_output(self.build_command, working_directory=self.get_project_directory())

	def profile(self, top_n: int) -> ProfileResult:
		"""
		Profile the application using Metrix and return performance metrics.

		Args:
		    top_n: Number of top kernels to profile (sorted by duration)

		Returns:
		    ProfileResult: Clean abstraction over profiling results with type-safe metric access
		"""
		logging.debug(f"Profiling app with name {self.get_name()}")
		logging.debug(f"Profiling app with command {self.get_app_cmd()}")

		try:
			from metrix import Metrix
		except ImportError:
			logging.error("Metrix profiler not found. Please install it from src/metrix")
			exit_on_fail(success=False, message="Metrix profiler not available")

		# Build command string from app_cmd list with absolute path
		from pathlib import Path
		import os
		app_cmd = self.get_app_cmd()
		project_dir = self.get_project_directory()

		if isinstance(app_cmd, list):
			# Convert first element (executable) to absolute path for rocprofv3
			if project_dir:
				exec_path = Path(project_dir) / app_cmd[0]
				command_parts = [str(exec_path.resolve())] + app_cmd[1:]
			else:
				# No project directory - resolve relative paths to absolute
				exec_path = Path(app_cmd[0])
				if not exec_path.is_absolute():
					exec_path = exec_path.resolve()
				command_parts = [str(exec_path)] + app_cmd[1:]
			command = " ".join(command_parts)
		else:
			command = str(app_cmd)

		logging.info(f"Profiling command: {command}")
		logging.info(f"Working directory: {project_dir or os.getcwd()}")

		# No need to change directory - we use absolute paths
		try:

			# Configure metrix logger to use IntelliPerf's logging level
			intelliperf_level = logging.getLogger().getEffectiveLevel()
			metrix_logger = logging.getLogger("metrix")
			metrix_logger.setLevel(intelliperf_level)
			logging.debug(f"Set metrix logger level to {logging.getLevelName(intelliperf_level)}")

			# Initialize metrix profiler (auto-detects GPU architecture)
			profiler = Metrix()

			# Profile the application with all available metrics
			results = profiler.profile(
				command=command,
				num_replays=3,  # Run multiple times for statistical accuracy
				aggregate_by_kernel=True,
				cwd=project_dir  # Run from project directory (or None for current dir)
			)
		finally:
			pass  # No cleanup needed

		if not results.kernels:
			logging.warning("No kernels found during profiling")
			return ProfileResult(kernels=[])

		# Sort kernels by duration (descending) and take top N
		sorted_kernels = sorted(results.kernels, key=lambda k: k.duration_us.avg, reverse=True)
		top_kernels = sorted_kernels[:top_n]

		logging.info(f"Found {len(results.kernels)} kernel(s), reporting top {len(top_kernels)}")

		# Convert metrix results to IntelliPerf ProfileResult format
		kernel_metrics_list = [KernelMetrics.from_metrix_kernel(kernel) for kernel in top_kernels]
		profile_result = ProfileResult(kernels=kernel_metrics_list)
		logging.debug(f"Profiling result: {profile_result}")
		return profile_result

	def run(self):
		"""Runs the application."""
		return process.capture_subprocess_output(self.app_cmd)

	def run_unit_test(self):
		"""Runs the unit test command."""
		cmd = self.unittest_command.split()
		return process.capture_subprocess_output(cmd, working_directory=self.get_project_directory())

	def get_name(self):
		return self.name

	def get_app_cmd(self):
		"""Returns the command for running the application."""
		return self.app_cmd

	def get_build_command(self):
		return self.build_command

	def get_instrument_command(self):
		return self.instrument_command

	def get_app_args(self):
		parts = self.app_cmd[1:]
		return parts[1] if len(parts) > 1 else ""

	def get_app_cmd_without_args(self):
		return self.app_cmd[0]

	def get_project_directory(self):
		return self.project_directory

	def clone(self):
		if not self.project_directory:
			logging.debug("Skipping cloning application without project directory")
			return self

		temp_dir = tempfile.mkdtemp()
		logging.info(f"Creating temporary project directory: {temp_dir}")

		shutil.copytree(self.project_directory, temp_dir, dirs_exist_ok=True)
		logging.debug(f"Copied project from {self.project_directory} to {temp_dir}")

		return Application(
			self.name + "_clone",
			self.build_command,
			self.instrument_command,
			temp_dir,
			self.app_cmd,
			self.unittest_command,
		)

	def collect_source_code(self):
		"""
		Collect source code for GPU kernels using Nexus.

		Returns:
			dict: Dictionary containing kernel information with assembly, HIP source, files, and line numbers
		"""
		try:
			from nexus import Nexus
		except ImportError:
			logging.error(
				"Nexus Python API not found. Please install it: pip install git+https://github.com/AMDResearch/nexus.git@main"
			)
			return {"kernels": {}}

		try:
			# Map Python logging level to Nexus log level
			# Python: NOTSET=0, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
			# Nexus: 0=none, 1=info, 2=warning, 3=error, 4=detail
			current_level = logging.getLogger().getEffectiveLevel()
			if current_level <= logging.DEBUG:
				nexus_log_level = 4  # detail (most verbose)
			elif current_level <= logging.INFO:
				nexus_log_level = 1  # info
			elif current_level <= logging.WARNING:
				nexus_log_level = 2  # warning
			else:
				nexus_log_level = 0  # none

			# Create Nexus tracer with inherited log level
			nexus = Nexus(log_level=nexus_log_level)

			# Additional environment for Triton kernels
			triton_env = {
				"TRITON_ALWAYS_COMPILE": "1",
				"TRITON_DISABLE_LINE_INFO": "0",
			}

			# Run the application and capture kernel trace
			trace = nexus.run(
				command=self.get_app_cmd(),
				env=triton_env,
				cwd=self.get_project_directory(),
			)

			# Convert trace to the expected format
			df_results = {"kernels": {}}
			for kernel in trace:
				# Normalize kernel name - strip .kd suffix if present
				kernel_name = kernel.name
				if kernel_name.endswith(".kd"):
					kernel_name = kernel_name[:-3]

				df_results["kernels"][kernel_name] = {
					"assembly": kernel.assembly,
					"hip": kernel.hip,
					"files": kernel.files,
					"lines": kernel.lines,
					"signature": kernel.signature,
				}

			return df_results

		except Exception as e:
			logging.error(f"Failed to collect source code with Nexus: {e}")
			return {"kernels": {}}

	def get_binary_absolute_path(self):
		if self.get_project_directory() != "":
			binary = self.get_app_cmd_without_args()
			if binary.startswith("./"):
				binary = binary[2:]  # Remove './'
			binary = os.path.join(self.get_project_directory(), binary)
			logging.debug(f"Binary absolute path: {binary}")
			logging.debug(f"Binary path: {binary}")
			logging.debug(f"Project directory: {self.get_project_directory()}")
			return binary
		else:
			return self.get_app_cmd_without_args()

	def show_details(self):
		logging.debug(f"Showing application details of {self.get_name()}")
		logging.debug(f"Project directory: {self.get_project_directory()}")
		logging.debug(f"Build command: {self.get_build_command()}")
		logging.debug(f"Instrument command: {self.get_instrument_command()}")
		logging.debug(f"App command: {self.get_app_cmd()}")
		logging.debug(f"App args: {self.get_app_args()}")
		logging.debug(f"App cmd without args: {self.get_app_cmd_without_args()}")
		logging.debug("--------------------------------")
