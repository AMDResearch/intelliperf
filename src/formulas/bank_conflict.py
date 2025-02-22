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

from formulas.formula_base import Formula_Base, Result
from utils.process import capture_subprocess_output
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
)
from tracer.python.communicate import get_kern_arg_data, send_response
from tracer.python.code_gen import generate_header
from tracer.python.utils import run_subprocess


class bank_conflict(Formula_Base):
    def __init__(self, name, build_script, app_cmd, only_consider_top_kernel=False):
        super().__init__(name, build_script, app_cmd)
        self.profiler = "guided-tuning"
        # This temp option allows us to toggle if we want a full or partial instrumentation report
        self.only_consider_top_kernel = only_consider_top_kernel

    def profile_pass(self) -> Result:
        """
        Profile the application using guided-tuning and collect bank conflict data

        Returns:
            Result: DataFrame containing the performance report card
        """
        super().profile_pass()
        logging.debug(f"Profiling app with name {self.get_app_name()}")
        logging.debug(f"Profiling app with command {self.get_app_cmd()}")
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
                list(matching_db_workloads.keys())[-1],
                "--save",
                f"{os.environ['GT_TUNING']}/maestro_output.csv",
            ]
        )
        # Handle critical error
        if not success:
            logging.error(f"Critical Error: {output}")
            logging.error("Failed to generate the performance report card.")
            sys.exit(1)
        # Read the saved report card
        df_results = pd.read_csv(f"{os.environ['GT_TUNING']}/maestro_output.csv")
        return Result(
            success=True,
            asset=df_results
        )

    def instrument_pass(self, perf_report_card: pd.DataFrame) -> Result:
        """
        Instrument the application, targeting the kernels with the highest bank conflict data

        Args:
            perf_report_card (pd.DataFrame): DataFrame containing kernel report card with bank conflict data

        Returns:
            Result: Instrumentation data containing the kernel name, arguments, lines, and file path as dict
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
        cmd=' '.join(self.get_app_cmd())
        logging.debug(f"Omniprobe profiling command is: {cmd}")
        success, output = capture_subprocess_output([
            "omniprobe",
            "--instrumented",
            "--analyzers", "MemoryAnalysis",
            "--kernels", ecma_regex,
            "--", " ".join(self.get_app_cmd())
        ])
        if not success:
            logging.error(f"Critical Error: {output}")
            logging.error("Failed to instrument the application.")
            sys.exit(1)

        bnk_conflicts_map = extract_bank_conflict_lines(output, kernel_names)

        return Result(
            success=True,
            asset=bnk_conflicts_map[kernel_names[0]] if self.only_consider_top_kernel else bnk_conflicts_map
        )

    def optimize_pass(self, file:str, kernel:str, lines:str, temperature:float=0.0, max_tokens:int=3000) -> Result:
        """
        Optimize the kernel to remove shared memory bank conflicts via OpenAI API

        Args:
            file (str): File path of the kernel
            kernel (str): Kernel name
            lines (str): Line numbers causing the conflict
            temperature (float): Sampling temperature for OpenAI API
            max_tokens (int): Maximum tokens for OpenAI API

        Returns:
            Result: Optimized kernel as a file path
        """
        super().optimize_pass()
        model = "gpt-4o"
        openai_key = os.getenv("OPENAI_API_KEY")

        if os.path.exists(file):
            with open(file, "r") as f:
                unoptimized_file_content = f.read()
        else:
            return Result(
                success=False,
                error_report=f"{file} does not exist."
            )

        user_prompt = f"Line {lines} is causing the conflict within the kernel {kernel} inside {unoptimized_file_content}. Please fix the conflict but do not change the semantics of the program."
        system_prompt = "You are a skilled GPU HIP programmer. Given a kernel, you will optimize it to remove shared memory bank conflicts and provide a correct performant implementation. Do not modify the kernel signature and include the dh_comms_dev.h header. Do not include any markdown code blocks or text other than the code."
        logging.debug(f"LLM prompt: {user_prompt}")
        logging.debug(f"System prompt: {system_prompt}")

        if not openai_key:
            return Result(
                success=False,
                error_report="Missing OpenAI API key."
            )
        try:
            client = openai.Client(api_key=openai_key)
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            optimized_file_content = completion.choices[0].message.content.strip()
            with open(file, "w") as f:
                f.write(optimized_file_content)
            return Result(
                success=True,
                asset={
                    "optimized_code_path": file,
                    "optimized_code_string": optimized_file_content
                }
            )

        except openai.AuthenticationError:
            return Result(
                success=False,
                error_report="Authentication failed. Check your API key."
            )
        except openai.RateLimitError:
            return Result(
                success=False,
                error_report="Rate limit exceeded. Try again later."
            )

        except openai.APIConnectionError:
            return Result(
                success=False,
                error_report="Failed to connect to OpenAI API."
            )
        except openai.OpenAIError as e:
            return Result(
                success=False,
                error_report=f"OpenAI API error - {str(e)}"
            )
        except Exception as e:
            return Result(
                success=False,
                error_report=f"An unexpected error occurred - {str(e)}"
            )

    def compiler_pass(self, file:str) -> Result:
        """
        Compile the optimized kernel using hipcc

        Args:
            file (str): File path of the optimized kernel

        Returns:
            Result: Compilation status and the output file path
        """
        super().compiler_pass()
        with tempfile.NamedTemporaryFile(suffix=".out", delete=False) as output_file:
            output_file_path = output_file.name

        # TODO: Need to pass the command to handle CMake code
        compile_cmd = ["hipcc", file, "-o", output_file_path]

        success, message = capture_subprocess_output(compile_cmd)
        return Result(
            success=success,
            asset={
                "binary": output_file_path,
                "log": message
            }
        )

    def validation_pass(self, unoptimized_binary:str, optimized_binary:str, kernel:str, args:list) -> Result:
        """
        Validate the optimized kernel by comparing the output with the reference kernel

        Args:
            optimized_binary (str): File path of the optimized kernel
            kernel (str): Kernel name
            args (list): List of kernel arguments

        Returns:
            Result: Validation status
        """
        super().validation_pass()

        tracer_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../", "tracer"))

        results = {}
        for binary, label in zip([unoptimized_binary, optimized_binary], ["unoptimized", "optimized"]):
            timestamp = int(time.time())
            pipe_name = f"/tmp/kernel_pipe_{timestamp}"
            ipc_file_name = f"/tmp/ipc_handle_{timestamp}.bin"

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
            env["TRACER_LOG_LEVEL"] = "3"
            env["TRACER_PIPE_NAME"] = pipe_name
            env["TRACER_IPC_OUTPUT_FILE"] = ipc_file_name

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


    def performance_pass(self, optimized_binary: str) -> Result:
        start_ref = time.time()
        reference_success, _ = capture_subprocess_output(self.get_app_cmd())
        ref_time = time.time() - start_ref


    def performance_pass(self, optimized_binary_result: Result,
                              unoptimized_binary_result: Result,
                              kernel_signature: str) -> Result:
        
        unoptimized_df = unoptimized_binary_result.asset
        unoptimized_time = unoptimized_df.loc[unoptimized_df['Kernel'] == kernel_signature, 'Avg-Duration'].sum()
        unoptimized_conflicts = unoptimized_df.loc[unoptimized_df['Kernel'] == kernel_signature, 'LDS Bank Conflicts'].sum()
        
        optimized_df = optimized_binary_result.asset
        optimized_time = optimized_df.loc[optimized_df['Kernel'] == kernel_signature, 'Avg-Duration'].sum()
        optimized_conflicts = optimized_df.loc[optimized_df['Kernel'] == kernel_signature, 'LDS Bank Conflicts'].sum()
        
    
        success = optimized_conflicts < unoptimized_conflicts 
        speedup = unoptimized_time / optimized_time
        conflict_improvement = unoptimized_conflicts / optimized_conflicts
        report_message = (f" The code contains {conflict_improvement}x fewer shared memory conflicts." 
                        f" The initial implementation contained {unoptimized_conflicts} conflicts and"
                        f" the optimized code contained {optimized_conflicts} conflicts."
                        f" The new code is {speedup:.3f}x faster than the original code. The initial"
                        f" implementation took {unoptimized_time} ns and the new one took"
                        f" {optimized_time} ns.")
                            
        if not success:
            return Result(
                success=False,
                error_report=f"The optimized code had more shared memory bank conflicts." + report_message

            )
        return Result(
            success=True,
            asset={
                "log": report_message
            }
        )

def generate_ecma_regex_from_list(kernel_names:set)->str:
    res = []
    for i in kernel_names:
        escaped_string = re.escape(i)
        regex_string = r"^" + escaped_string + r"$"
        res.append(regex_string)
        # Note: Temporary fix, but until bug in omniprobe is fixed we need to also
        # add the name of the instrumented kernel clone to the regex, otherwise we'll skip it
        # and exclude it from the memory analysis report
        duplicate_kernel_str = f"__amd_crk_{i.replace(')', ', void*)', 1)}"
        #duplicate_kernel_str = f"__amd_crk_{i.replace(")", ", void*)", 1)}"
        escaped_string = re.escape(duplicate_kernel_str)
        regex_string = r"^" + escaped_string + r"$"
        res.append(regex_string)

    regex = f"({'|'.join(res)})"
    return regex


def extract_bank_conflict_lines(output:str, kernel_names:list)->dict:
    """
    Extract the bank conflict report from omniprobe output

    Args:
        output (str): Omniprobe output
        kernel_names (list): List of kernel names with bank conflicts

    Returns:
        dict: Dictionary containing kernel name, arguments, file path, and line number
    """
    kernel_reports = {}
    inside_kernel_report = False
    current_kernel_name = None
    filename = None

    for line in output.splitlines():
        if "source location:" in line:
            filename = line.split()[2].split(':')[0]
        if not inside_kernel_report:
            # Check if the line marks the start of a kernel report
            for kernel_sig in kernel_names:
                if f"Memory analysis for {kernel_sig}" in line:
                    inside_kernel_report = True
                    current_kernel_name = kernel_sig

                    kernel_args = kernel_sig.split('(')[1].split(')')[0].split(',')
                    kernel_reports[current_kernel_name] = {
                        'kernel': kernel_sig.split('(')[0],
                        'kernel_signature': kernel_sig,
                        'arguments': [word.strip() for word in kernel_args],
                        'file': filename,
                        'lines': None
                    }
                    break
        else:
            # Check for the bank conflicts report
            if "bank conflicts for location" in line:
                # Extract the line number
                line_number = int(line.split()[4])
                kernel_reports[current_kernel_name]['lines'] = line_number
                # Exit early after finding the relevant information
                inside_kernel_report = False
                current_kernel_name = None

            # Check for the end of the bank conflicts report
            if "=== End of bank conflicts report" in line:
                inside_kernel_report = False
                current_kernel_name = None
                filename = None

    logging.debug(f"Parsed instrumentation output:\n{kernel_reports}")
    if len(kernel_reports) == 0:
        logging.warning("No memory analysis data parsed from instrumentation.")
    return kernel_reports
