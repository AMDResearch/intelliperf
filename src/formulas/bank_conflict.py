from formulas.formula_base import Formula_Base
import logging
import subprocess
import os
import io
import selectors
import pandas as pd

class bank_conflict(Formula_Base):
    def __init__(self, name, app_cmd):
        super().__init__(name, app_cmd)
        self.profiler = "guided-tuning"
        
    def profile_pass(self) -> pd.DataFrame:
        """
        Profile the application using guided-tuning and collect bank conflict data

        Returns:
            pd.DataFrame: DataFrame containing kernel report card with bank conflict data
        """
        super().profile_pass()

        # Call guided-tuning to profile the application
        success, output = capture_subprocess_output([
            f"{os.environ['GT_TUNING']}/bin/profile_and_load.sh", 
            self.get_app_name()
        ] + self.get_app_cmd())
        # Load report card with --save flag
        success, output = capture_subprocess_output([
            f"{os.environ['GT_TUNING']}/bin/show_data.sh", 
            "-n", 
            self.get_app_name(),
        ])
        matching_db_workloads = {}
        for line in output.splitlines():
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                key, value = parts
                matching_db_workloads[key] = value
        logging.debug(f"Matching DB Workloads: {matching_db_workloads}")
        success, output = capture_subprocess_output([
            f"{os.environ['GT_TUNING']}/bin/show_data.sh", 
            "-w", 
            list(matching_db_workloads.keys())[0],
            "--save",
            f"{os.environ['GT_TUNING']}/maestro_output.csv"
        ])
        # Read the saved report card
        df_results = pd.read_csv(f"{os.environ['GT_TUNING']}/maestro_output.csv")
        return df_results
    
    def instrument_pass(self, perf_report_card:pd.DataFrame):
        """
        Instrument the application, targeting the kernels with the highest bank conflict data

        Args:
            perf_report_card (pd.DataFrame): DataFrame containing kernel report card with bank conflict data
        """
        super().instrument_pass()
        #TODO: Finish instrumentation implementation
        pass

    def optimize_pass(self):
        super().optimize_pass()
        #TODO: Connect optimization agent to this pass
        pass

def capture_subprocess_output(subprocess_args:list, new_env=None) -> tuple:
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    process = (
        subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        if new_env == None
        else subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=new_env,
        )
    )

    # Create callback function for process output
    buf = io.StringIO()

    def handle_output(stream, mask):
        try:
            # Because the process' output is line buffered, there's only ever one
            # line to read when this function is called
            line = stream.readline()
            buf.write(line)
            print(line.strip())
        except UnicodeDecodeError:
            # Skip this line
            pass

    # Register callback for an "available for read" event from subprocess' stdout stream
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)

    # Loop until subprocess is terminated
    while process.poll() is None:
        # Wait for events and handle them with their registered callbacks
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    # Get process return code
    return_code = process.wait()
    selector.close()

    success = return_code == 0

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)
    