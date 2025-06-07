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

import logging
import os
import time

import numpy as np

from accordo.python.code_gen import generate_header
from accordo.python.communicate import get_kern_arg_data, send_response
from accordo.python.utils import run_subprocess

logging.basicConfig(level=logging.DEBUG, format="[Python] [%(asctime)s]: %(message)s")


example_directory = os.path.dirname(os.path.abspath(__file__))
accordo_directory = os.path.dirname(example_directory)


def fetch_results(binary):
	timestamp = int(time.time())
	pipe_name = f"/tmp/kernel_pipe_{timestamp}"
	ipc_file_name = f"/tmp/ipc_handle_{timestamp}.bin"

	# Clear output
	for file in [ipc_file_name, ipc_file_name]:
		if os.path.exists(file):
			os.remove(file)
			logging.debug(f"Remove the {file} file")

	logging.debug(f"IPC files {pipe_name} and {ipc_file_name} cleared before execution.")

	# Compile
	generate_header(args)
	run_subprocess(["cmake", "-B", "build"], accordo_directory)
	run_subprocess(["cmake", "--build", "build", "--parallel", "16"], accordo_directory)

	lib = os.path.join(accordo_directory, "build", "lib", "libaccordo.so")

	env = os.environ.copy()
	env["HSA_TOOLS_LIB"] = lib
	env["KERNEL_TO_TRACE"] = kernel
	env["ACCORDO_LOG_LEVEL"] = "3"
	env["ACCORDO_PIPE_NAME"] = pipe_name
	env["ACCORDO_IPC_OUTPUT_FILE"] = ipc_file_name

	pid = os.posix_spawn(binary, [binary], env)

	results = get_kern_arg_data(pipe_name, args, ipc_file_name)

	send_response(pipe_name)
	logging.info("Sent response -- waiting.")

	_, status = os.waitpid(pid, 0)

	# Check the exit status
	if os.WIFEXITED(status):
		exit_code = os.WEXITSTATUS(status)
		print(f"Process exited with code {exit_code}")
	elif os.WIFSIGNALED(status):
		print(f"Process was terminated by signal {os.WTERMSIG(status)}")

	logging.info("Python processing complete")
	return results


# app = "dummy"
# kernel = "emptyKernel"'
# args=["int* arg0"]
# generate_header(args)

results = {}

kernel = "matrixTransposeShared"
args = ["float*", "const float*", "int", "int"]
apps = {
	"initial_code": "examples/bank_conflict/matrix_transpose/matrix_transpose.optimized",
	"optimized_code": "examples/bank_conflict/matrix_transpose/matrix_transpose.unoptimized",
}


kernel = "reduction_kernel"
args = ["const float*", "float*", "std::size_t"]
apps = {
	"initial_code": "examples/contention/reduction/reduction.optimized",
	"optimized_code": "examples/contention/reduction/reduction.unoptimized",
}


for app in ["initial_code", "optimized_code"]:
	binary = "../../" + apps[app]
	results[app] = fetch_results(binary)
	print(f"{app} results {results[app]}")

for key in results.keys():
	logging.info(f"Collected results for {key}:\n{results[key]}")

key0, key1 = results.keys()

for i in range(len(results[key0])):
	if np.allclose(results[key0][i], results[key1][i]):
		print(f"Arrays at index {i} for '{key0}' and '{key1}' are close.")
	else:
		print(f"Arrays at index {i} for '{key0}' and '{key1}' are NOT close.")
