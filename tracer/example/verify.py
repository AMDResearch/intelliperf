import os
import time
import subprocess
import struct
import stat
import logging
import numpy as np

from communicate import *
from code_gen import *
from utils import *

logging.basicConfig(level=logging.INFO, format="[Python] [%(asctime)s]: %(message)s")

pipe_name = "/tmp/kernel_pipe"
ipc_file_name = "/tmp/ipc_handle.bin"

example_directory = os.path.dirname(os.path.abspath(__file__))
tracer_directory = os.path.dirname(example_directory)


# app = "dummy"
# kernel = "emptyKernel"'
# args=["int* arg0"]
# generate_header(args)

results = {}

for app in ["initial_code", "optimized_code"]:

    kernel = "reduction_kernel"
    args = ["double* arg0", "double* arg1", "std::size_t arg2"]

    # Clear output
    for file in [ipc_file_name, ipc_file_name]:
        if os.path.exists(file):
            os.remove(file)
    logging.debug("IPC file cleared before execution.")

    # Compile
    generate_header(args)
    binary = os.path.join("/tmp", app)
    run_subprocess(
        ["hipcc", "-o", f"{binary}", f"{example_directory}/hip/{app}.hip"], "/tmp"
    )
    run_subprocess(["cmake", "-B", "build"], tracer_directory)
    run_subprocess(["cmake", "--build", "build"], tracer_directory)

    lib = os.path.join(tracer_directory, "build", "lib", "libtracer.so")

    env = os.environ.copy()
    env["HSA_TOOLS_LIB"] = lib
    env["KERNEL_TO_TRACE"] = kernel
    # env["TRACER_LOG_LEVEL"] = "0"
    env["TRACER_PIPE_NAME"] = pipe_name
    env["TRACER_IPC_OUTPUT_FILE"] = ipc_file_name

    pid = os.posix_spawn(binary, [binary], env)

    results[app] = get_kern_arg_data(pipe_name, args, ipc_file_name)

    send_response(pipe_name)

    logging.info("Python processing complete")

for key in results.keys():
    logging.info(f"Collected results for {key}:\n{results[key]}")

key0, key1 = results.keys()

for i in range(len(results[key0])):
    if np.allclose(results[key0][i], results[key1][i]):
        print(f"Arrays at index {i} for '{key0}' and '{key1}' are close.")
    else:
        print(f"Arrays at index {i} for '{key0}' and '{key1}' are NOT close.")
