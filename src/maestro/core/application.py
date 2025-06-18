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

from maestro.utils import process
from maestro.utils.env import get_guided_tuning_path, get_nexus_path, get_rocprofiler_path
from maestro.utils.process import capture_subprocess_output, exit_on_fail


class Application:
	def __init__(self, name: str, build_command: list, instrument_command: list, project_directory: str, app_cmd: list):
		self.name = name

		self.build_command = build_command if isinstance(build_command, list) else build_command.split()
		self.instrument_command = (
			instrument_command if isinstance(instrument_command, list) else instrument_command.split()
		)
		self.app_cmd = app_cmd
		self.project_directory = project_directory

		logging.debug(f"App name: {self.name}")
		logging.debug(f"App build command: {self.build_command}")
		logging.debug(f"App instrument command: {self.instrument_command}")
		logging.debug(f"App project directory: {self.project_directory}")
		logging.debug(f"App app command: {self.app_cmd}")

	def build(self, instrumented=False):
		"""Builds the application, optionally with instrumentation."""
		if instrumented:
			return process.capture_subprocess_output(
				self.instrument_command, working_directory=self.get_project_directory()
			)
		else:
			return process.capture_subprocess_output(self.build_command, working_directory=self.get_project_directory())

	def profile(self, top_n: int):
		logging.debug(f"Profiling app with name {self.get_name()}")
		logging.debug(f"Profiling app with command {self.get_app_cmd()}")

		# Clear the cache before running the profiler
		capture_subprocess_output(
			["rm", "-rf", f"{get_guided_tuning_path()}/workloads/"],
			working_directory=self.get_project_directory(),
			additional_path=get_rocprofiler_path(),
		)

		capture_subprocess_output(
			["rm", "-rf", f"{get_guided_tuning_path()}/data/guided.db"],
			working_directory=self.get_project_directory(),
			additional_path=get_rocprofiler_path(),
		)

		# Profile the app using GT
		success, output = capture_subprocess_output(
			[
				f"{get_guided_tuning_path()}/bin/gt",
				"profile",
				"-n",
				self.get_name(),
				"--top-n",
				str(top_n),
				"--",
			]
			+ self.get_app_cmd(),
			working_directory=self.get_project_directory(),
			additional_path=get_rocprofiler_path(),
		)

		exit_on_fail(success=success, message="Failed to profile the binary", log=output)

		# Load workload summary with GT. Save list of top-n kernels for regex
		success, output = capture_subprocess_output(
			[
				f"{get_guided_tuning_path()}/bin/gt",
				"db",
				"-n",
				self.get_name(),
				"--save",
				f"{get_guided_tuning_path()}/maestro_workloads.csv",
			],
			working_directory=self.get_project_directory(),
			additional_path=get_rocprofiler_path(),
		)
		exit_on_fail(success=success, message="Failed to generate the performance report card.", log=output)

		df_maestro_workloads = pd.read_csv(f"{get_guided_tuning_path()}/maestro_workloads.csv")
		logging.debug(f"Matching DB Workloads: {df_maestro_workloads}")
		last_matching_id = df_maestro_workloads.iloc[-1]["id"]

		success, output = capture_subprocess_output(
			[
				f"{get_guided_tuning_path()}/bin/gt",
				"db",
				"-w",
				last_matching_id,
				"--save",
				f"{get_guided_tuning_path()}/maestro_summary.csv",
			],
			working_directory=self.get_project_directory(),
			additional_path=get_rocprofiler_path(),
		)
		# Handle critical error
		exit_on_fail(success=success, message="Failed to generate the performance report card.", log=output)
		df_results = pd.read_csv(f"{get_guided_tuning_path()}/maestro_summary.csv")
		# Create a targeted report card
		top_n_kernels = list(df_results.head(top_n)["Kernel"])
		logging.debug(f"top_n_kernels: {top_n_kernels}")
		success, output = capture_subprocess_output(
			[
				f"{get_guided_tuning_path()}/bin/gt",
				"db",
				"-w",
				last_matching_id,
				"-k",
				f"{'|'.join(top_n_kernels)}",
				"--separate",
				"--save",
				f"{get_guided_tuning_path()}/maestro_report_card.json",
			],
			working_directory=self.get_project_directory(),
			additional_path=get_rocprofiler_path(),
		)
		df_results = json.loads(open(f"{get_guided_tuning_path()}/maestro_report_card.json").read())
		return df_results

	def run(self):
		"""Runs the application."""
		return process.capture_subprocess_output(self.app_cmd)

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
			raise ValueError("Cannot clone application without project directory")

		temp_dir = tempfile.mkdtemp()
		logging.info(f"Creating temporary project directory: {temp_dir}")

		shutil.copytree(self.project_directory, temp_dir, dirs_exist_ok=True)
		logging.debug(f"Copied project from {self.project_directory} to {temp_dir}")

		return Application(self.name + "_clone", self.build_command, self.instrument_command, temp_dir, self.app_cmd)

	def collect_source_code(self):
		nexus_directory = get_nexus_path()
		lib = os.path.join(nexus_directory, "build", "lib", "libnexus.so")
		env = os.environ.copy()

		with tempfile.TemporaryDirectory() as tmp:
			json_result_file = os.path.join(tmp, "nexus_output.json")

			env["HSA_TOOLS_LIB"] = lib
			env["NEXUS_LOG_LEVEL"] = "2"
			env["NEXUS_OUTPUT_FILE"] = json_result_file
			capture_subprocess_output(self.get_app_cmd(), new_env=env, working_directory=self.get_project_directory())

			if os.path.exists(json_result_file):
				df_results = json.loads(open(json_result_file).read())
			else:
				df_results = {"kernels": {}}

			return df_results
