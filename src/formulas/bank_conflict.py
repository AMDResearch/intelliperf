import logging
import os
import pandas as pd
import tempfile
import time
import sys
import numpy as np
import re

import openai
from openai import OpenAIError

from formulas.formula_base import Formula_Base
from utils.process import capture_subprocess_output
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)
from tracer.python.communicate import get_kern_arg_data, send_response
from tracer.python.code_gen import generate_header
from tracer.python.utils import run_subprocess


class bank_conflict(Formula_Base):
    def __init__(self, name, build_script, app_cmd):
        super().__init__(name, build_script, app_cmd)
        self.profiler = "guided-tuning"

    def profile_pass(self) -> pd.DataFrame:
        """
        Profile the application using guided-tuning and collect bank conflict data

        Returns:
            pd.DataFrame: DataFrame containing kernel report card with bank conflict data
        """
        super().profile_pass()

        # Call guided-tuning to profile the application
        success, output = capture_subprocess_output(
            [f"{os.environ['GT_TUNING']}/bin/profile_and_load.sh", self.get_app_name()]
            + self.get_app_cmd()
        )
        # Load report card with --save flag
        success, output = capture_subprocess_output(
            [
                f"{os.environ['GT_TUNING']}/bin/show_data.sh",
                "-n",
                self.get_app_name(),
            ]
        )
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
                list(matching_db_workloads.keys())[0],
                "--save",
                f"{os.environ['GT_TUNING']}/maestro_output.csv",
            ]
        )
        if not success:
            logging.error(f"Error: {output}")
            logging.error("Failed to generate the performance report card.")
            sys.exit(1)
        # Read the saved report card
        df_results = pd.read_csv(f"{os.environ['GT_TUNING']}/maestro_output.csv")
        return df_results

    def instrument_pass(self, perf_report_card: pd.DataFrame):
        """
        Instrument the application, targeting the kernels with the highest bank conflict data

        Args:
            perf_report_card (pd.DataFrame): DataFrame containing kernel report card with bank conflict data
        """
        super().instrument_pass()
        # Get the kernel names with the highest bank conflict data and filter
        filtered_report_card = perf_report_card[perf_report_card["LDS Bank Conflicts"] > 0]
        filtered_report_card = filtered_report_card[~filtered_report_card["Kernel"].str.contains("Cijk")]
        logging.debug(f"Filtered Report Card:\n{filtered_report_card}")
        kernel_names = filtered_report_card["Kernel"].tolist()
        
        # Generate ECMA regex from the list of kernel names
        ecma_regex = generate_ecma_regex_from_list(kernel_names)
        logging.debug(f"ECMA Regex for kernel names: {ecma_regex}")

        success, output = capture_subprocess_output([
            "omniprobe",
            "--instrumented",
            "--analyzers", "MemoryAnalysis",
            "--kernels", ecma_regex,
            "--", " ".join(self.get_app_cmd())
        ])
        if not success:
            logging.error(f"Error: {output}")
            logging.error("Failed to instrument the application.")
            sys.exit(1)
        return {
            "success": True,
            "kernel": "matrixTransposeShared",
            "arguments": ["float*", "float const*", "int", "int"],
            "lines": "17; 26",
            "file": "../examples/bank_conflict/matrix_transpose/matrix_transpose.hip",
        }

    def optimize_pass(self, file, kernel, lines, temperature=0.0, max_tokens=3000):
        super().optimize_pass()
        model = "gpt-4o"
        openai_key = os.getenv("OPENAI_API_KEY")

        if os.path.exists(file):
            with open(file, "r") as f:
                file_content = f.read()
        else:
            logging.error(f"Error: {file} does not exist.")
            sys.exit(1)

        prompt = f"Lines {lines} are causing the conflict within the kernel {kernel} inside {file_content}."
        if not openai_key:
            logging.error("Error: Missing OpenAI API key.")
            sys.exit(1)
        try:
            client = openai.Client(api_key=openai_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a skilled GPU HIP programmer. Given a kernel, you will optimize it to remove shared memory bank conflicts and provide a correct performant implementation. Do not modify the kernel signature. Do not include any markdown code blocks or text other than the code.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            tmp_file_path = "/tmp/optimized.hip"
            file_content = completion.choices[0].message.content.strip()
            with open(tmp_file_path, "w") as f:
                f.write(file_content)
            return tmp_file_path

        except openai.AuthenticationError:
            logging.error("Error: Authentication failed. Check your API key.")
            sys.exit(1)
        except openai.RateLimitError:
            logging.error("Error: Rate limit exceeded. Try again later.")
            sys.exit(1)

        except openai.APIConnectionError:
            logging.error("Error: Failed to connect to OpenAI API.")
            sys.exit(1)
        except openai.OpenAIError as e:
            logging.error(f"Error: OpenAI API error - {str(e)}")
            sys.exit(1)
        except Exception as e:
            logging.error(f"Error: An unexpected error occurred - {str(e)}")
            sys.exit(1)

    def compiler_pass(self, file:str) -> dict:
        """
        Compile the optimized kernel using hipcc
        
        Args:   
            file (str): File path of the optimized kernel

        Returns:
            dict: Compilation data containing the compiler log and binary path    
        """
        super().compiler_pass()
        with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as output_file:
            output_file_path = output_file.name

        # TODO: Need to pass the command to handle CMake code
        compile_cmd = ["hipcc", file, "-o", output_file_path]

        success, message = capture_subprocess_output(compile_cmd)
        if not success:
            logging.error(f"Error: {message}")
            logging.error("Failed to compile the optimized kernel.")
            sys.exit(1)
        return {"compiler_log": message, "binary": output_file_path}

    def validation_pass(self, optimized_binary, kernel, args):
        super().validation_pass()
        
        tracer_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../", "tracer"))
        
        pipe_name = "/tmp/kernel_pipe"
        ipc_file_name = "/tmp/ipc_handle.bin"

        results = {}
        for binary, label in zip([self.get_app_cmd()[0], optimized_binary], ["unoptimized", "optimized"]):
            for file in [ipc_file_name, ipc_file_name]:
                if os.path.exists(file):
                    os.remove(file)
            generate_header(args) 
                        
            run_subprocess(["cmake", "-B", "build"], tracer_directory)
            run_subprocess(["cmake", "--build", "build", "--parallel", "16"], tracer_directory)
            
            lib = os.path.join(tracer_directory, "build", "lib", "libtracer.so")
            env = os.environ.copy()
            env["HSA_TOOLS_LIB"] = lib
            env["KERNEL_TO_TRACE"] = kernel
            env["TRACER_LOG_LEVEL"] = "0"
            env["TRACER_PIPE_NAME"] = pipe_name
            env["TRACER_IPC_OUTPUT_FILE"] = ipc_file_name

            pid = os.posix_spawn(binary, [binary], env)
            results[label] = get_kern_arg_data(pipe_name, args, ipc_file_name)
            send_response(pipe_name)
        key0, key1 = results.keys()
        for i in range(len(results[key0])):
            if not np.allclose(results[key0][i], results[key1][i]):
                logging.error(f"Arrays at index {i} for '{key0}' and '{key1}' are NOT close.")
                sys.exit(1)
                

    def performance_pass(self, optimized_binary: str):
        start_ref = time.time()
        reference_success, _ = capture_subprocess_output(self.get_app_cmd())
        ref_time = time.time() - start_ref

        start_upd = time.time()
        optimized_success, optimized_message = capture_subprocess_output(
            optimized_binary + self.get_app_args()
        )

        upd_time = time.time() - start_upd
        success = optimized_success and reference_success
        performant = upd_time < ref_time
        speedup = ref_time / upd_time
        if not success:
            logging.error(f"Execution failed with message {optimized_message}")
            sys.exit(1)

        print(f"The code is {speedup}x faster. Old code took {ref_time} seconds and the optimized code took {upd_time} seconds.")

def generate_ecma_regex_from_list(kernel_names)->str:  
    res = []
    for i in kernel_names:
        escaped_string = re.escape(i)  
        regex_string = r"^" + escaped_string + r"$"
        res.append(regex_string)
    
    regex = f"({'|'.join(res)})"
    return regex
