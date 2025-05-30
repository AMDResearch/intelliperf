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

from formulas.formula_base import Formula_Base, Result, filter_json_field, write_results
from utils.process import capture_subprocess_output, exit_on_fail
from utils.regex import generate_ecma_regex_from_list


class atomic_contention(Formula_Base):
    def __init__(self, name: str, build_command: list, instrument_command: list, project_directory: str, app_cmd: list, top_n: int, only_consider_top_kernel=False):
        
        super().__init__(name, build_command, instrument_command, project_directory, app_cmd, top_n)
        
        self._reference_app = self._application.clone()
        
        # This temp option allows us to toggle if we want a full or partial instrumentation report
        self.only_consider_top_kernel = only_consider_top_kernel
        self._instrumentation_results = None
        self.current_kernel = None
        self.current_args = None
        self.current_kernel_signature = None
        self.kernel_to_optimize = None
        self.optimization_report = None        
        self.bottleneck_report = None

    def profile_pass(self) -> Result:
        """
        Profile the application using guided-tuning and collect atomic contention data

        Returns:
            Result: DataFrame containing the performance report card
        """
        return super().profile_pass()

    def instrument_pass(self) -> Result:
        """
        Instrument the application, targeting the kernels with the highest atomic contention data

        Returns:
            Result: Instrumentation data containing the kernel name, arguments, lines, and file path as dict
        """
        super().instrument_pass()
     
        return Result(success=False, asset=self._instrumentation_results,
                      error_report="Instrumentation pass not implemented for atomic contention.")

    def optimize_pass(self, temperature: float = 0.0, max_tokens: int = 3000) -> Result:
        """
        Optimize the kernel to remove atomic contention via OpenAI API

        Args:
            temperature (float): Sampling temperature for OpenAI API
            max_tokens (int): Maximum tokens for OpenAI API

        Returns:
            Result: Optimized kernel as a file path
        """

        super().optimize_pass()
        model = "gpt-4o"
        llm_key  = os.getenv("LLM_GATEWAY_KEY")
        
        if not llm_key:
            exit_on_fail(success=False, message="Missing LLM API key. Please set the LLM_GATEWAY_KEY environment variable.")
                
        
        system_prompt = (
            "You are a skilled GPU HIP programmer. Given a kernel,"
            " you will optimize it to remove atomic contention as much as possible"
            " and provide a correct performant implementation. Do not modify"
            " the kernel signature. Do not touch any other code, licenses, copyrights, or comments in the file." 
            " If you remove the copyright, your solution will be rejected."
            " Do not include any markdown code blocks or text other than the code."
        )
                        
        server = "https://llm-api.amd.com/azure"
        deployment_id = "dvue-aoai-001-o4-mini"
        llm = LLM(
            model=model,
            api_key=llm_key,
            system_prompt=system_prompt,
            deployment_id=deployment_id,
            server=server,
        )

        if self._instrumentation_results is None:
            # Get the file from the results
            field = "atomics"
            subfield = "atomic_lat"
            average_atomic_lat = 1000
            filtered_report_card = filter_json_field(self._initial_profiler_results, field=field,
                                                     subfield=subfield,
                                                     comparison_func=lambda x: x > average_atomic_lat)
            
            if len(filtered_report_card) == 0:
                return Result(success=False, error_report="No atomic contention found.")
            
            
            logging.debug(f"Filtered Report Card:\n{json.dumps(filtered_report_card, indent=4)}")


            kernel = filtered_report_card[0]["kernel"]
            files = filtered_report_card[0]["source"]["files"]
            kernel_name = kernel.split("(")[0]
            kernel_file = None
            for file in files:
                if os.path.exists(file):
                    with open(file, "r") as f:
                        unoptimized_file_content = f.read()
                        if kernel_name in unoptimized_file_content:
                            kernel_file = file
                            break
            if kernel_file is None:
                return Result(success=False, error_report=f"Kernel file not found.")
            
            user_prompt = (
                f"There is atomic contention in the kernel {kernel} in the file {unoptimized_file_content}."
                f" Please fix the contention but do not change the semantics of the program."
                " If you remove the copyright, your solution will be rejected."
            )
            args = kernel.split("(")[1].split(")")[0]
            self.bottleneck_report = f"Maestro detected atomic contention in the kernel {kernel_name} with arguments {args}."
            
        else:
            pass

        

        logging.debug(f"LLM prompt: {user_prompt}")
        logging.debug(f"System prompt: {system_prompt}")


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

    def correctness_validation_pass(
        self
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
        return super().correctness_validation_pass(self.current_kernel, self.current_args)

    def performance_validation_pass(
        self
    ) -> Result:


        unoptimized_results = filter_json_field(self._initial_profiler_results, field="kernel", 
                                                comparison_func = lambda x: x == self.current_kernel_signature)
        
        unoptimized_time = unoptimized_results[0]["durations"]["ns"]
        
        field = "atomics"
        subfield = "atomic_lat"
        unoptimized_metric = unoptimized_results[0][field][subfield]

        # Profile the optimized application
        self._optimization_results = self._application.profile(top_n=self.top_n)
        
        optimized_results = filter_json_field(self._optimization_results, field="kernel", 
                                                comparison_func = lambda x: x == self.current_kernel_signature)
        
        optimized_time = optimized_results[0]["durations"]["ns"]
        optimized_metric = optimized_results[0][field][subfield]

        success = optimized_metric < unoptimized_metric
        speedup = unoptimized_time / optimized_time
        metric_improvement = (
            unoptimized_metric / optimized_metric
            if optimized_metric != 0
            else 1
        )
        
        # Calculate cycle latency improvement percentage
        cycle_latency_improvement = (
            (unoptimized_metric - optimized_metric) / unoptimized_metric * 100
            if unoptimized_metric > 0
            else 0
        )

        self.optimization_report = (
            f"The optimized code shows {metric_improvement * 100:.1f}% reduction in atomic contention. "
            f"Average atomic instruction latency improved from {unoptimized_metric:.2f} to {optimized_metric:.2f} cycles ({cycle_latency_improvement:.1f}% reduction). "
            f"The optimized implementation is {speedup:.2f}x faster overall, "
            f"reducing execution time from {unoptimized_time/1e6:.2f}ms to {optimized_time/1e6:.2f}ms."
        )

        if not success:
            return Result(
                success=False,
                error_report=f"The optimized code had more atomic contention."
                + self.optimization_report,
            )
            
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
        }
        write_results(results, output_file)
        
    def summarize_previous_passes(self):
        """
        Summarizes the results of the previous passes for future prompts.
        """
        pass

