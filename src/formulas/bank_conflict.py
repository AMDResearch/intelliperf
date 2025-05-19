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

import logging
import os
import pandas as pd
import tempfile
import time
import sys
import numpy as np
import re
import json

from core.llm import LLM

from formulas.formula_base import Formula_Base, Result
from utils.process import capture_subprocess_output, exit_on_fail
from utils.regex import generate_ecma_regex_from_list

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


class bank_conflict(Formula_Base):
    def __init__(self, name, build_script, app_cmd, only_consider_top_kernel=False):
        super().__init__(name, build_script, app_cmd)
        self.profiler = "guided-tuning"
        # This temp option allows us to toggle if we want a full or partial instrumentation report
        self.only_consider_top_kernel = only_consider_top_kernel

    def profile_pass(self) -> Result:
        """
        Profile the application using guided-tuning and collect bank conflict data

        Returns:
            Result: DataFrame containing the performance report card
        """
        return super().profile_pass()

    def instrument_pass(self) -> Result:
        """
        Instrument the application, targeting the kernels with the highest bank conflict data

        Returns:
            Result: Instrumentation data containing the kernel name, arguments, lines, and file path as dict
        """
        super().instrument_pass()
        # Get the kernel names with the highest bank conflict data and filter
        filtered_report_card = self._profiler_results[
            self._profiler_results["LDS Bank Conflicts"] > 0
        ]
        filtered_report_card = filtered_report_card[
            ~filtered_report_card["Kernel"].str.contains("Cijk")
        ]
        logging.debug(f"Filtered Report Card:\n{filtered_report_card}")
        kernel_names = filtered_report_card["Kernel"].tolist()

        # Generate ECMA regex from the list of kernel names
        ecma_regex = generate_ecma_regex_from_list(kernel_names)
        logging.debug(f"ECMA Regex for kernel names: {ecma_regex}")
        cmd = " ".join(self.get_app_cmd())
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
                " ".join(self.get_app_cmd()),
            ]
        )
        if not success:
            logging.error(f"Critical Error: {output}")
            logging.error("Failed to instrument the application.")
            sys.exit(1)

        bnk_conflicts_map = extract_bank_conflict_lines(output, kernel_names)

        self._instrumentation_results = (
            bnk_conflicts_map[kernel_names[0]]
            if self.only_consider_top_kernel
            else bnk_conflicts_map
        )

        return Result(success=True, asset=self._instrumentation_results)

    def optimize_pass(self, temperature: float = 0.0, max_tokens: int = 3000) -> Result:
        """
        Optimize the kernel to remove shared memory bank conflicts via OpenAI API

        Args:
            file (str): File path of the kernel
            kernel (str): Kernel name
            lines (str): Line numbers causing the conflict
            temperature (float): Sampling temperature for OpenAI API
            max_tokens (int): Maximum tokens for OpenAI API

        Returns:
            Result: Optimized kernel as a file path
        """

        kernel = self._instrumentation_results["kernel"]
        lines = self._instrumentation_results["lines"]
        file = self._instrumentation_results["file"]

        super().optimize_pass()
        model = "gpt-4o"
        llm_key  = os.getenv("LLM_GATEWAY_KEY")
        
        if not llm_key:
            return Result(success=False, error_report="Missing OpenAI API key.")
                
        server = "https://llm-api.amd.com/azure"
        deployment_id = "dvue-aoai-001-o4-mini"
        llm = LLM(
            model=model,
            api_key=llm_key,
            system_prompt=system_prompt,
            deployment_id=deployment_id,
            server=server,
        )
        
        if os.path.exists(file):
            with open(file, "r") as f:
                unoptimized_file_content = f.read()
        else:
            return Result(success=False, error_report=f"{file} does not exist.")

        user_prompt = (
            f"Line {lines} is causing the conflict within the kernel {kernel}"
            f" inside {unoptimized_file_content}. Please fix the conflict but"
            f" do not change the semantics of the program."
        )
        system_prompt = (
            "You are a skilled GPU HIP programmer. Given a kernel,"
            " you will optimize it to remove shared memory bank conflicts"
            " and provide a correct performant implementation. Do not modify"
            " the kernel signature and include the dh_comms_dev.h header."
            " Do not include any markdown code blocks or text other than the code."
        )
        logging.debug(f"LLM prompt: {user_prompt}")
        logging.debug(f"System prompt: {system_prompt}")


        try:
            optimized_file_content = llm.ask(user_prompt).strip()
            with open(file, "w") as f:
                f.write(optimized_file_content)
            return Result(
                success=True,
                asset={
                    "optimized_code_path": file,
                    "optimized_code_string": optimized_file_content,
                },
            )
        except Exception as e:
            return Result(success=False, error_report=f"An unexpected error occurred - {str(e)}")

    def compiler_pass(self) -> Result:
        """
        Compile the application

        Returns:
            Result: Compilation status and the output file path
        """
        return super().compile_pass()

    def validation_pass(
        self, unoptimized_binary: str, optimized_binary: str, kernel: str, args: list
    ) -> Result:
        """
        Validate the optimized kernel by comparing the output with the reference kernel

        Args:
            optimized_binary (str): File path of the optimized kernel
            kernel (str): Kernel name
            args (list): List of kernel arguments

        Returns:
            Result: Validation status
        """
        return super().validation_pass()

    def performance_pass(
        self,
        optimized_binary_result: Result,
        unoptimized_binary_result: Result,
        kernel_signature: str,
    ) -> Result:

        unoptimized_df = unoptimized_binary_result.asset
        unoptimized_time = unoptimized_df.loc[
            unoptimized_df["Kernel"] == kernel_signature, "Avg-Duration"
        ].sum()
        unoptimized_conflicts = unoptimized_df.loc[
            unoptimized_df["Kernel"] == kernel_signature, "LDS Bank Conflicts"
        ].sum()

        optimized_df = optimized_binary_result.asset
        optimized_time = optimized_df.loc[
            optimized_df["Kernel"] == kernel_signature, "Avg-Duration"
        ].sum()
        optimized_conflicts = optimized_df.loc[
            optimized_df["Kernel"] == kernel_signature, "LDS Bank Conflicts"
        ].sum()

        success = optimized_conflicts < unoptimized_conflicts
        speedup = unoptimized_time / optimized_time
        conflict_improvement = (
            unoptimized_conflicts / optimized_conflicts
            if optimized_conflicts != 0
            else 1
        )

        report_message = (
            f"The optimized code contains {conflict_improvement * 100}% fewer shared memory conflicts."
            f" The initial implementation contained {unoptimized_conflicts} conflicts and"
            f" the optimized code contained {optimized_conflicts} conflicts."
            f" The new code is {speedup:.3f}x faster than the original code. The initial"
            f" implementation took {unoptimized_time} ns and the new one took"
            f" {optimized_time} ns."
        )

        if not success:
            return Result(
                success=False,
                error_report=f"The optimized code had more shared memory bank conflicts."
                + report_message,
            )
        return Result(success=True, asset={"log": report_message})


def extract_bank_conflict_lines(output: str, kernel_names: list) -> dict:
    """
    Extract the bank conflict report from omniprobe output

    Args:
        output (str): Omniprobe output
        kernel_names (list): List of kernel names with bank conflicts

    Returns:
        dict: Dictionary containing kernel name, arguments, file path, and line number
    """
    kernel_reports = {}
    inside_kernel_report = False
    inside_bank_conflict_report = False
    current_kernel_name = None

    for line in output.splitlines():
        if not inside_kernel_report:
            # Check if the line marks the start of a kernel report
            for kernel_sig in kernel_names:
                if f"Memory analysis for {kernel_sig}" in line:
                    inside_kernel_report = True
                    current_kernel_name = kernel_sig

                    kernel_args = kernel_sig.split("(")[1].split(")")[0].split(",")
                    kernel_reports[current_kernel_name] = {
                        "kernel": kernel_sig.split("(")[0],
                        "kernel_signature": kernel_sig,
                        "arguments": [word.strip() for word in kernel_args],
                        "file": None,
                        "lines": set(),
                    }
                    break
        elif not inside_bank_conflict_report:
            if "Bank conflicts report" in line:
                inside_bank_conflict_report = True
        else:
            line_l = line.replace("WARNING:root:", "").split(":")
            if os.path.isfile(line_l[0]):
                kernel_reports[current_kernel_name]["file"] = str(line_l[0])
                kernel_reports[current_kernel_name]["lines"].add(str(line_l[1]))

            # Check for the end of the bank conflicts report
            if "=== End of bank conflicts report" in line:
                kernel_reports[current_kernel_name]["lines"] = ";".join(
                    kernel_reports[current_kernel_name]["lines"]
                )
                inside_kernel_report = False
                current_kernel_name = None
                inside_bank_conflict_report = False

    logging.debug(f"Parsed instrumentation output:\n{kernel_reports}")
    if len(kernel_reports) == 0:
        logging.warning("No memory analysis data parsed from instrumentation.")
    return kernel_reports
