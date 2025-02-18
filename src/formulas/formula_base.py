from abc import ABC, abstractmethod
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from process import capture_subprocess_output

class Formula_Base:
    def __init__(self, name: str, build_script: str, app_cmd: list):
        # Private
        self.__name = name # name of the run
        self.__build_script = build_script # script to build the application
        self.__app_cmd:list = app_cmd # command to execute application

        # Public
        self.profiler:str = None

    def get_app_cmd(self):
        return self.__app_cmd
    def get_app_args(self):
        parts = self.__app_cmd[1:]
        return parts[1] if len(parts) > 1 else ""
    def get_app_name(self):
        return self.__name
    def build(self):
        success, result = capture_subprocess_output(self.__build_script)
        return {
            "success": success,
            "result": result
        }
    # ----------------------------------------------------
    # Required methods to be implemented by child classes
    # ----------------------------------------------------
    @abstractmethod
    def profile_pass(self):
        """
        Extract any required performance data from the application using the specified profiler.
        """
        # Validate app command
        if self.__app_cmd:
            self.__app_cmd = self.__app_cmd[1:]
        else:
            logging.error("Profiling command required. Pass application executable after -- at the end of options.")

        # Validate profiler
        if self.profiler == "guided-tuning":
            if 'GT_TUNING' not in os.environ:
                logging.error(f"Cannot resolve profiler {self.profiler}: GT_TUNING environment variable must be set to install dir.")
        else:
            logging.error(f"Profiler {self.profiler} not supported.")


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
