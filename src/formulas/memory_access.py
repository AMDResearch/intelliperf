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

from formulas.formula_base import Formula_Base, Result, filter_json_field
from utils.process import capture_subprocess_output, exit_on_fail
from utils.regex import generate_ecma_regex_from_list

class memory_access(Formula_Base):
    def __init__(self, name: str, build_command: list, instrument_command: list, project_directory: str, app_cmd: list, top_n: int, only_consider_top_kernel=False):
        
        super().__init__(name, build_command, instrument_command, project_directory, app_cmd, top_n)
        
        self._reference_app = self._application.clone()

        self.profiler = "guided-tuning"
        # This temp option allows us to toggle if we want a full or partial instrumentation report
        self.only_consider_top_kernel = only_consider_top_kernel
        self._instrumentation_results = None
        self.current_kernel = None
        self.current_args = None

    def profile_pass(self) -> Result:
        """
        Profile the application using guided-tuning and collect uncoalesced memory access data

        Returns:
            Result: DataFrame containing the performance report card
        """
        return super().profile_pass()

    def instrument_pass(self) -> Result:
        """
        Instrument the application, targeting the kernels with the highest uncoalesced memory access data

        Returns:
            Result: Instrumentation data containing the kernel name, arguments, lines, and file path as dict
        """
        super().instrument_pass()
     
        return Result(success=False, asset=self._instrumentation_results)

    def optimize_pass(self, temperature: float = 0.0, max_tokens: int = 3000) -> Result:
        """
        Optimize the kernel to remove uncoalesced memory access via OpenAI API

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
            exit_on_fail(success=False, message="Missing LLM API key.")
                
        
        system_prompt = (
            "You are a skilled GPU HIP programmer. Given a kernel,"
            " you will optimize it to remove uncoalesced memory access as much as possible"
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
            field = "l1"
            subfield = "coal"
            peak_coal = 100
            filtered_report_card = filter_json_field(self._initial_profiler_results, field, subfield, lambda x: x < peak_coal)
            
            if len(filtered_report_card) == 0:
                return Result(success=False, error_report="No uncoalesced memory access found.")
            
            logging.debug(f"Filtered Report Card:\n{filtered_report_card}")
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
                f"There is an uncoalesced memory access in the kernel {kernel} in the file {unoptimized_file_content}."
                f" Please fix the access pattern but do not change the semantics of the program."
                " If you remove the copyright, your solution will be rejected."
            )
            
        else:
            pass


        logging.debug(f"LLM prompt: {user_prompt}")
        logging.debug(f"System prompt: {system_prompt}")


        self.current_kernel = kernel.split("(")[0]
        self.current_args = kernel.split("(")[1].split(")")[0].split(",")

        try:
            optimized_file_content = llm.ask(user_prompt).strip()
            with open(file, "w") as f:
                f.write(optimized_file_content)
            logging.debug(f"Optimized file content: {optimized_file_content}")
            return Result(
                success=True,
                asset={
                    "optimized_code_path": file,
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


        unoptimized_time = self._initial_profiler_results[0]["durations"]["ns"]
        
        field = "l1"
        subfield = "coal"
        peak_coal = 100
        filtered_report_card = filter_json_field(self._initial_profiler_results, field, subfield, lambda x: x < peak_coal)
        unoptimized_coal = filtered_report_card[0][field][subfield]
        kernel = filtered_report_card[0]["kernel"]
        
        # Profile the optimized application
        self._optimization_results = self._application.profile(top_n=self.top_n)
        
        optimized_time = self._optimization_results[0]["durations"]["ns"]
        field = "kernel"
        filtered_report_card = filter_json_field(self._optimization_results, field, lambda x: x == kernel)
        optimized_coal = filtered_report_card[0][field][subfield]

        success = optimized_coal > unoptimized_coal
        speedup = unoptimized_time / optimized_time
        coal_improvement = (
            optimized_coal / unoptimized_coal
            if optimized_coal != 0
            else 1
        )

        report_message = (
            f"The optimized code contains {coal_improvement * 100}% fewer uncoalesced memory access."
            f" The initial implementation contained {unoptimized_coal} uncoalesced memory access and"
            f" the optimized code contained {optimized_coal} uncoalesced memory access."
            f" The new code is {speedup:.3f}x faster than the original code. The initial"
            f" implementation took {unoptimized_time} ns and the new one took"
            f" {optimized_time} ns."
        )

        if not success:
            return Result(
                success=False,
                error_report=f"The optimized code had more uncoalesced memory access."
                + report_message,
            )
            
        logging.info(report_message)
        
        return Result(success=True, asset={"log": report_message})

    def summarize_previous_passes(self):
        """
        Summarizes the results of the previous passes for future prompts.
        """
        pass

