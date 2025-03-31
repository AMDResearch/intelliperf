import os
import sys
import json
import logging
import pandas as pd
from formulas.formula_base import Formula_Base, Result
from utils.process import capture_subprocess_output, exit_on_fail

TOP_N = 10

class diagnose_only(Formula_Base):
    def __init__(self, name, build_script, app_cmd):
        super().__init__(name, build_script, app_cmd)
        self.profiler = "guided-tuning"

    def profile_pass(self):
        super().profile_pass()
        logging.debug(f"Profiling app with name {self.get_app_name()}")
        logging.debug(f"Profiling app with command {self.get_app_cmd()}")
        # Call guided-tuning to profile the application
        success, output = capture_subprocess_output(
            [f"{os.environ['GT_TUNING']}/bin/profile_and_load.sh", self.get_app_name()]
            + self.get_app_cmd()
        )
        
        exit_on_fail(success = success,
                     message = "Failed to profile the binary",
                     log = output)
                
        # Load report card with --save flag
        success, output = capture_subprocess_output(
            [
                f"{os.environ['GT_TUNING']}/bin/show_data.sh",
                "-n",
                self.get_app_name(),
            ]
        )
        exit_on_fail(success = success,
                     message = "Failed to generate the performance report card.",
                     log = output)
                
        matching_db_workloads = {}
        for line in output.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                matching_db_workloads[key] = value
        logging.debug(f"Matching DB Workloads: {matching_db_workloads}")
        success, output = capture_subprocess_output(
            [
                f"{os.environ['GT_TUNING']}/bin/show_data.sh",
                "-w",
                list(matching_db_workloads.keys())[-1],
                "--save",
                f"{os.environ['GT_TUNING']}/maestro_summary.csv",
            ]
        )
        # Handle critical error
        exit_on_fail(success = success,
                     message = "Failed to generate the performance report card.",
                     log = output)
        df_results = pd.read_csv(f"{os.environ['GT_TUNING']}/maestro_summary.csv")
        # Create a targeted report card
        top_n_kernels = list(df_results.head(TOP_N)["Kernel"])
        logging.debug(f"top_n_kernels: {top_n_kernels}")
        success, output = capture_subprocess_output(
            [
                f"{os.environ['GT_TUNING']}/bin/show_data.sh",
                "-w",
                list(matching_db_workloads.keys())[-1],
                "-k",
                *top_n_kernels,
                "--separate",
                "--save",
                f"{os.environ['GT_TUNING']}/maestro_report_card.json",
            ]
        )
        df_results = json.loads(open(f"{os.environ['GT_TUNING']}/maestro_report_card.json").read())
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
    