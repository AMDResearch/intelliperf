from formulas.formula_base import Formula_Base
import logging
import os
import pandas as pd
import tempfile
import time
import sys
import numpy as np

import openai
from openai import OpenAIError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "utils")))
from process import capture_subprocess_output

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../", "tracer", "python"))
)
from communicate import *
from code_gen import *
from utils import *


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
        # Read the saved report card
        df_results = pd.read_csv(f"{os.environ['GT_TUNING']}/maestro_output.csv")
        return {"success": True, "result": df_results}

    def instrument_pass(self, perf_report_card: pd.DataFrame):
        """
        Instrument the application, targeting the kernels with the highest bank conflict data

        Args:
            perf_report_card (pd.DataFrame): DataFrame containing kernel report card with bank conflict data
        """
        super().instrument_pass()
        # TODO: Finish instrumentation implementation
        return {
            "success": True,
            "kernel": "matrixTransposeShared",
            "arguments": ["float*", "float const*", "int", "int"],
            "lines": "17; 26",
            "file": "../examples/bank_conflict/matrix_transpose/matrix_transpose.hip",
        }
        pass

    def optimize_pass(self, file, kernel, lines, temperature=0.0, max_tokens=3000):
        super().optimize_pass()
        model = "gpt-4o"
        openai_key = os.getenv("OPENAI_API_KEY")

        if os.path.exists(file):
            with open(file, "r") as f:
                file_content = f.read()
        else:
            return {"success": False, "message": f"{file} does not exist."}

        prompt = f"Lines {lines} are causing the conflict within the kernel {kernel} inside {file_content}."
        if not openai_key:
            return {"success": False, "message": "Missing OpenAI API key."}
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
            return {"success": True, "optimized_code": tmp_file_path}

        except openai.AuthenticationError:
            return {
                "success": False,
                "message": "Error: Authentication failed. Check your API key.",
            }
        except openai.RateLimitError:
            return {
                "success": False,
                "message": "Error: Rate limit exceeded. Try again later.",
            }
        except openai.APIConnectionError:
            return {
                "success": False,
                "message": "Error: Failed to connect to OpenAI API.",
            }
        except openai.OpenAIError as e:
            return {"success": False, "message": f"Error: OpenAI API error - {str(e)}"}
        except Exception as e:
            return {
                "success": False,
                "message": f"Error: An unexpected error occurred - {str(e)}",
            }

    def compiler_pass(self, file):
        super().compiler_pass()
        with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as output_file:
            output_file_path = output_file.name

        # TODO: Need to pass the command to handle CMake code
        compile_cmd = ["hipcc", file, "-o", output_file_path]

        success, message = capture_subprocess_output(compile_cmd)
        return {"success": success, "compiler_log": message, "binary": output_file_path}

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
                return {
                    "success": False,
                    "message": f"Arrays at index {i} for '{key0}' and '{key1}' are NOT close."
                }
                
        return {
            "success": True,
        }

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
            return {
                "success": False,
                "message": f"Execution failed with message {optimized_message}",
            }

        message = f"The code is {speedup}x faster. Old code took {ref_time} seconds and the optimized code took {upd_time} seconds."
        return {"success": performant, "message": message}
