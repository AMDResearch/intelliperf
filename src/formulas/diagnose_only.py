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

import os
import sys
import json
import logging
import pandas as pd
from formulas.formula_base import Formula_Base, Result
from utils.process import capture_subprocess_output, exit_on_fail
from utils.env import get_guided_tuning_path
import re

class diagnose_only(Formula_Base):
    def __init__(self, name, build_script, app_cmd, top_n):
        super().__init__(name, build_script, app_cmd)
        self.profiler = "guided-tuning"
        self.top_n = top_n

    def profile_pass(self):
        super().profile_pass()
        logging.debug(f"Profiling app with name {self.get_app_name()}")
        logging.debug(f"Profiling app with command {self.get_app_cmd()}")
        # Profile the app using GT
        success, output = capture_subprocess_output(
            [
                f"{get_guided_tuning_path()}/bin/gt", "profile", 
                "-n", self.get_app_name(),
                "--top-n", str(self.top_n),
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
                "-n", self.get_app_name(),
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
        top_n_kernels = list(df_results.head(self.top_n)["Kernel"])
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



        return Result(
            success=True,
            asset=df_results
        )

    def instrument_pass(self):
        super().instrument_pass()

    def optimize_pass(self):
        super().optimize_pass()

    def compiler_pass(self):
        super().compiler_pass()

    def validation_pass(self):
        super().validation_pass()
    
    def source_code_pass(self):
        return super().source_code_pass()
