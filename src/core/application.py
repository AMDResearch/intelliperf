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

from utils import process
import shutil
import logging
from utils.process import capture_subprocess_output, exit_on_fail
from utils.env import get_guided_tuning_path
import pandas as pd
import json
import sys
import os
import tempfile

class Application:
    def __init__(self, name: str, build_command: list, instrument_command: list, app_cmd: list):
        self.name = name
        self.build_command = build_command
        self.instrument_command = instrument_command
        self.app_cmd = app_cmd
        
        # Validate app command
        if self.app_cmd and "--" in self.app_cmd:
            self.app_cmd = self.app_cmd[1:]
        else:
            logging.error("Profiling command required. Pass application executable after -- at the end of options.")
            sys.exit(1)
        

    def build(self, instrumented=False):
        """Builds the application, optionally with instrumentation."""
        if instrumented:
            return process.capture_subprocess_output(self.instrument_command)
        else:
            return process.capture_subprocess_output(self.build_command)

    def profile(self, top_n: int):
        logging.debug(f"Profiling app with name {self.get_name()}")
        logging.debug(f"Profiling app with command {self.get_app_cmd()}")
        
        # Clear the cache before running the profiler
        capture_subprocess_output([
            "rm", "-rf", f"{get_guided_tuning_path()}/workloads/"
        ])
        
        # Profile the app using GT
        success, output = capture_subprocess_output(
            [
                f"{get_guided_tuning_path()}/bin/gt", "profile", 
                "-n", self.get_name(),
                "--top-n", str(top_n),
                "--",
            ] + self.get_app_cmd()
        )
            
        exit_on_fail(success = success,
                    message = "Failed to profile the binary",
                    log = output)
                    
        # Load workload summary with GT. Save list of top-n kernels for regex
        success, output = capture_subprocess_output(
            [
                f"{get_guided_tuning_path()}/bin/gt", "db",
                "-n", self.get_name(),
            ]
        )
        exit_on_fail(success = success,
                        message = "Failed to generate the performance report card.",
                        log = output)
                    
        matching_db_workloads = {}
        for line in output.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and not parts[0].startswith("GT"):
                key, value = parts
                matching_db_workloads[key] = value
        logging.debug(f"Matching DB Workloads: {matching_db_workloads}")
        success, output = capture_subprocess_output(
            [
                f"{get_guided_tuning_path()}/bin/gt", "db",
                "-w", list(matching_db_workloads.keys())[-1],
                "--save", f"{get_guided_tuning_path()}/maestro_summary.csv",
            ]
        )
        # Handle critical error
        exit_on_fail(success = success,
                    message = "Failed to generate the performance report card.",
                    log = output)
        df_results = pd.read_csv(f"{get_guided_tuning_path()}/maestro_summary.csv")
        # Create a targeted report card
        top_n_kernels = list(df_results.head(top_n)["Kernel"])
        logging.debug(f"top_n_kernels: {top_n_kernels}")
        success, output = capture_subprocess_output(
            [
                f"{get_guided_tuning_path()}/bin/gt", "db",
                "-w", list(matching_db_workloads.keys())[-1],
                "-k", f'{"|".join(top_n_kernels)}',
                "--separate",
                "--save",
                f"{get_guided_tuning_path()}/maestro_report_card.json",
            ]
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
    
    def clone(self, suffix: str = "clone"):
        binary = self.app_cmd[0]
        backup_name = f"{binary}.{suffix}"
        logging.info(f"copying: {binary} to {backup_name}")
        shutil.copy2(binary, backup_name)

        new_app_cmd = [backup_name] + self.app_cmd[1:]
        
        # A clone of the application can't be instrumented or built
        return Application(self.name, None, None, new_app_cmd)
    
    def collect_source_code(self):
        nexus_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../external/nexus"))
        lib = os.path.join(nexus_directory, "build", "lib", "libnexus.so")
        env = os.environ.copy()


        with tempfile.TemporaryDirectory() as tmp:
            json_result_file = os.path.join(tmp, 'nexus_output.json')

            env["HSA_TOOLS_LIB"] = lib
            env["NEXUS_LOG_LEVEL"] = "2"
            env["NEXUS_OUTPUT_FILE"] = json_result_file
            capture_subprocess_output(self.get_app_cmd(), new_env=env)
            
            if os.path.exists(json_result_file):
                df_results = json.loads(open(json_result_file).read())
            else:
                df_results = {"kernels": {}}

            return df_results
