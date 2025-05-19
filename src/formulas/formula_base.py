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
import tempfile


from utils.process import capture_subprocess_output, exit_on_fail
from core.application import Application
from accordo.python.communicate import get_kern_arg_data, send_response
from accordo.python.code_gen import generate_header
from accordo.python.utils import run_subprocess

class Result:
    def __init__(self, success:bool, error_report:str="", asset=None):
        self.success:bool = success
        # Only set error report if failure occurs
        if not self.success and error_report == "":
            logging.error("Invalid implementation of Report(). Must provide an error report if failure occurs.")
            sys.exit(1)
        self.error_report:str = error_report
        self.log:str = ""
        self.asset = asset

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
    def __init__(self, name: str, build_command: list, instrument_command: list, app_cmd: list):
        # Private
        self.__name = name # name of the run
        self.__build_command = build_command # script to build the application
        self.__instrument_command = instrument_command # command to execute application
        self.__application = Application(name, build_command, instrument_command, app_cmd)

        self.__app_cmd:list = app_cmd # command to execute application
        
        self._profiler_results = None
        
        # Public
        self.profiler:str = None
        
          
        # Validate app command
        if self.__app_cmd and "--" in self.__app_cmd:
            self.__app_cmd = self.__app_cmd[1:]
        else:
            logging.error("Profiling command required. Pass application executable after -- at the end of options.")
            sys.exit(1)

    def backup(self, suffix: str):
        """Creates a backup of the application by appending the given suffix."""
        binary = self.__app_cmd[0]
        backup_name = f"{binary}.{suffix}"
        logging.info(f"copying: {binary} to {backup_name}")
        shutil.copy2(binary, backup_name)
        return backup_name 

    def build(self):
        if not self.__build_command:
            return Result(
                success=True,
                asset={
                    "log": "No build script provided. Skipping build step."
                }
            )
        else:
            success, result = self.__application.build()
            # Handle critical error
            exit_on_fail(success = success,
                        message = f"Failed to build {self.__name} application.",
                        log = result)

    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def profile_pass(self):
        """
        Extract any required performance data from the application using the specified profiler.
        """
        self._profiler_results = self.__application.profile(top_n=self.top_n)

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
    def validation_pass(self):
        """
        Validates the the application.
        """

        cloned_app = self.__application.clone(prefix = "unoptimized")

        self.__application.build()

        unoptimized_binary = self.__application.get_app_cmd()[0]
        optimized_binary = cloned_app.get_app_cmd()[0]
        
        
        kernel = self._instrumentation_results["kernel"]
        args = self._instrumentation_results["arguments"]

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
        nexus_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../external/nexus"))
        lib = os.path.join(nexus_directory, "build", "lib", "libnexus.so")
        env = os.environ.copy()


        with tempfile.TemporaryDirectory() as tmp:
            json_result_file = os.path.join(tmp, 'nexus_output.json')

            env["HSA_TOOLS_LIB"] = lib
            env["NEXUS_LOG_LEVEL"] = "2"
            env["NEXUS_OUTPUT_FILE"] = json_result_file
            success, log = capture_subprocess_output(self.get_app_cmd(), new_env=env)
            
            if os.path.exists(json_result_file):
                df_results = json.loads(open(json_result_file).read())
            else:
                df_results = {"kernels": {}}

        if not success:
            return Result(
                success=False,
                asset=log,
                error_report="Failed to collect the source code."
            )

        # In-place append of source info
        for entry in self._profiler_results:
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
            asset=self._profiler_results
        )        

