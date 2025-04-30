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

from abc import ABC, abstractmethod
import logging
import os
import sys
import shutil
import pandas as pd
import json
from pprint import pformat
from utils.process import capture_subprocess_output, exit_on_fail


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
    def __init__(self, name: str, build_script: str, app_cmd: list):
        # Private
        self.__name = name # name of the run
        self.__build_script = build_script # script to build the application
        self.__app_cmd:list = app_cmd # command to execute application

        # Public
        self.profiler:str = None

          
        # Validate app command
        if self.__app_cmd:
            self.__app_cmd = self.__app_cmd[1:]
        else:
            logging.error("Profiling command required. Pass application executable after -- at the end of options.")
        

    def backup(self, suffix: str):
        """Creates a backup of the application by appending the given suffix."""
        binary = self.__app_cmd[0]
        backup_name = f"{binary}.{suffix}"
        logging.info(f"copying: {binary} to {backup_name}")
        shutil.copy2(binary, backup_name)
        return backup_name 

    def get_app_cmd(self):
        return self.__app_cmd
    def get_app_args(self):
        parts = self.__app_cmd[1:]
        return parts[1] if len(parts) > 1 else ""
    def get_app_name(self):
        return self.__name
    def build(self):
        if not self.__build_script:
            return Result(
                success=True,
                asset={
                    "log": "No build script provided. Skipping build step."
                }
            )
        else:
            success, result = capture_subprocess_output(self.__build_script)
            
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
        pass

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
    def compiler_pass(self):
        """
        Compiles the application.
        """
        pass

    @abstractmethod
    def validation_pass(self):
        """
        Validates the the application.
        """
        pass
    @abstractmethod
    def source_code_pass(self):
        """
        Finds the source code.
        """
        pass
