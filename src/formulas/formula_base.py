from abc import ABC, abstractmethod
import logging
import os

class Formula_Base:
    def __init__(self, name: str, app_cmd: list):
        # Private
        self.__name = name # name of the run
        self.__app_cmd = app_cmd # command to execute application

        # Public
        self.profiler:str = None
        
    def get_app_cmd(self):
        return self.__app_cmd
    def get_app_name(self):
        return self.__name
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
            self.__app_cmd = " ".join(self.__app_cmd)
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