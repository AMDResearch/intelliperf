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

from abc import abstractmethod
import logging
import os
import sys
import shutil
import pandas as pd
import json
from pprint import pformat
import time
import numpy as np

from utils.process import capture_subprocess_output, exit_on_fail
from core.application import Application

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from accordo.python.communicate import get_kern_arg_data, send_response
from accordo.python.code_gen import generate_header
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
    def __init__(self, name: str, build_command: list, instrument_command: list, app_cmd: list, top_n: int):
        # Private
        self.__name = name # name of the run
        self._application = Application(name, build_command, instrument_command, app_cmd)

        self._initial_profiler_results = None
        
        # Public
        self.profiler:str = None
        self.top_n:int = top_n
          

    def backup(self, suffix: str):
        """Creates a backup of the application by appending the given suffix."""
        binary = self._application.get_app_cmd()[0]
        backup_name = f"{binary}.{suffix}"
        logging.info(f"copying: {binary} to {backup_name}")
        shutil.copy2(binary, backup_name)
        return backup_name 

    def build(self):
        if not self._application.get_build_command():
            return Result(
                success=True,
                asset={
                    "log": "No build script provided. Skipping build step."
                }
            )
        else:
            success, result = self._application.build()
            # Handle critical error
            exit_on_fail(success = success,
                        message = f"Failed to build {self.__name} application.",
                        log = result)
        return Result(
            success=success,
            asset={
                "log": result
            }
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
    @abstractmethod
    def instrument_pass(self):
        """
        Instrument elements of the application to pinpoint source of bottleneck.
        """
        pass

    @abstractmethod
    def optimize_pass(self):
        """
        Optimize the application based on the data collected from the instrumentation pass.
        """
        pass


    @abstractmethod
    def validation_pass(self, kernel, args):
        """
        Validates the the application.
        """

        cloned_app = self._application.clone(suffix = "unoptimized")

        self._application.build()

        unoptimized_binary = self._application.get_app_cmd()[0]
        optimized_binary = cloned_app.get_app_cmd()[0]

        accordo_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../", "accordo"))

        results = {}
        for binary, label in zip([unoptimized_binary, optimized_binary], ["unoptimized", "optimized"]):
            timestamp = int(time.time())
            pipe_name = f"/tmp/kernel_pipe_{timestamp}"
            ipc_file_name = f"/tmp/ipc_handle_{timestamp}.bin"

            for file in [ipc_file_name, ipc_file_name]:
                if os.path.exists(file):
                    os.remove(file)
            generate_header(args)

            run_subprocess(["cmake", "-B", "build"], accordo_directory)
            run_subprocess(["cmake", "--build", "build", "--parallel", "16"], accordo_directory)
            lib = os.path.join(accordo_directory, "build", "lib", "libaccordo.so")
            env = os.environ.copy()
            env["HSA_TOOLS_LIB"] = lib
            env["KERNEL_TO_TRACE"] = kernel
            env["ACCORDO_LOG_LEVEL"] = "0"
            env["ACCORDO_PIPE_NAME"] = pipe_name
            env["ACCORDO_IPC_OUTPUT_FILE"] = ipc_file_name

            pid = os.posix_spawn(binary, [binary], env)
            results[label] = get_kern_arg_data(pipe_name, args, ipc_file_name)
            send_response(pipe_name)
        logging.debug(f"results unoptimized: results['unoptimized']")
        logging.debug(f"results optimized: results['optimized']")
        key0, key1 = results.keys()
        for i in range(len(results[key0])):
            if not np.allclose(results[key0][i], results[key1][i]):
                diff = np.abs(results[key0][i] - results[key1][i])
                logging.debug(f"Arrays at index {i} for '{key0}' and '{key1}' are NOT close.")
                logging.debug(f"  {key0}[{i}]: {results[key0][i]}")
                logging.debug(f"  {key1}[{i}]: {results[key1][i]}")
                logging.debug(f"  Difference: {diff}")

        for i in range(len(results[key0])):
            if not np.allclose(results[key0][i], results[key1][i]):
                return Result(
                    success=False,
                    error_report=f"Arrays at index {i} for '{key0}' and '{key1}' are NOT close."
                )
        logging.debug("Validation succeeded.")
        return Result(
            success=True
        )        
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
                "signature": ""}
            entry["source"] = df_results["kernels"].get(kernel_name, empty)
            
        return Result(
            success=True,
            asset=self._initial_profiler_results
        )        

    @abstractmethod
    def summarize_previous_passes(self):
        """
        Summarizes the results of the previous passes for future prompts.
        """
        pass

    @abstractmethod
    def write_results(self, output_file: str = None):
        """
        Writes the results to the output file.
        """
        if output_file is None:
            print(json.dumps(self._initial_profiler_results, indent=2))
        elif output_file.endswith(".json"):
            with open(output_file, "w") as f:
                json.dump(self._initial_profiler_results, f, indent=2)
        elif output_file.endswith(".csv"):
            flattened_results = [flatten_dict(entry) for entry in self._initial_profiler_results]
            df = pd.DataFrame(flattened_results)
            df.to_csv(output_file, index=False)
        elif output_file.endswith(".txt"):
            with open(output_file, "w") as f:
                f.write(json.dumps(self._initial_profiler_results, indent=2))
        else:
            logging.error("Invalid output file extension. Must be .json, .csv, or .txt.")
            sys.exit(1)

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
