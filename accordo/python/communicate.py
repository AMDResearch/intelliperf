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

import ctypes
import logging
import os
import stat
import sys
import time

import ml_dtypes
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hip import memcpy_d2h, open_ipc_handle


def read_ipc_handles(args, ipc_file_name):
	count = sum(1 for arg in args if "*" in arg and "const" not in arg)

	handles = []
	sizes = []
	handles_set = set()

	while len(handles) < count:
		if not os.path.exists(ipc_file_name):
			logging.debug("Waiting for IPC file...")
			time.sleep(0.1)
			continue

		with open(ipc_file_name, "rb") as file:
			data = file.read()

		messages = data.split(b"BEGIN\n")
		for message in messages:
			if b"END\n" in message:
				content = message.split(b"END\n")[0]

				if len(content) == 72:
					handle_data = content[:64]
					size_data = content[64:72]

					handle_np = np.frombuffer(handle_data, dtype=np.uint8)
					handle_tuple = tuple(handle_np)

					if handle_tuple not in handles_set:
						handles.append(handle_np)
						handles_set.add(handle_tuple)

						size_value = int.from_bytes(size_data, byteorder="little")
						sizes.append(size_value)

						logging.debug("Final IPC Handle (hex):")
						for i in range(0, len(handle_np), 16):
							chunk = handle_np[i : i + 16]
							logging.debug(" ".join(f"{b:02x}" for b in chunk))

						logging.debug(f"Corresponding Pointer Size: {size_value} bytes")

		if len(handles) < count:
			logging.debug(f"Waiting for {count - len(handles)} more IPC handles...")
			time.sleep(0.1)

	# logging.debug(f"Successfully read {len(handles)} IPC handles and sizes.")
	return handles, sizes


def send_response(pipe_name):
	with open(pipe_name, "w") as fifo:
		fifo.write("done\n")


def get_kern_arg_data(pipe_name, args, ipc_file_name, ipc_timeout_seconds=30):
	logging.debug(f"pipe_name: {pipe_name}")
	logging.debug(f"get_kern_arg_data args: {args}")
	logging.debug(f"ipc_file_name: {ipc_file_name}")
	if not os.path.exists(pipe_name):
		os.mkfifo(pipe_name)
		os.chmod(pipe_name, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

	start_time = time.time()
	with open(pipe_name, "rb") as fifo:  # noqa:  F841
		while True:
			if time.time() - start_time > ipc_timeout_seconds:
				raise TimeoutError(f"Timeout after {ipc_timeout_seconds} seconds waiting for IPC data")

			try:
				ipc_handles, ptr_sizes = read_ipc_handles(args, ipc_file_name)
				break
			except Exception as e:
				if time.time() - start_time > ipc_timeout_seconds:
					raise TimeoutError(f"Timeout after {ipc_timeout_seconds} seconds waiting for IPC data: {str(e)}")
				time.sleep(0.1)

	type_map = {
		"double*": ctypes.c_double,
		"float*": ctypes.c_float,
		"int*": ctypes.c_int,
		"std::size_t*": ctypes.c_size_t,
		"__half*": np.float16,
		"__hip_bfloat16*": ml_dtypes.bfloat16,
	}
	results = []
	pointer_args = list(filter(lambda arg: "*" in arg and "const" not in arg, args))
	logging.debug(f"pointer_args: {pointer_args}")
	for handle, arg, array_size in zip(ipc_handles, pointer_args, ptr_sizes):
		ptr = open_ipc_handle(handle)
		logging.debug(f"Opened IPC Ptr: {ptr} (0x{ptr:x})")
		arg_type = arg.split()[0]
		logging.debug(f"arg_type: {arg_type}")
		if arg_type in type_map:
			dtype = type_map[arg_type]
			logging.debug(f"dtype: {dtype}")
			# Special handling for FP16 and bfloat16
			if arg_type == "__half*":
				temp_array = memcpy_d2h(ptr, array_size // 2, ctypes.c_uint16)
				host_array = np.frombuffer(temp_array, dtype=np.float16)
			elif arg_type == "__hip_bfloat16*":
				temp_array = memcpy_d2h(ptr, array_size // 2, ctypes.c_uint16)
				host_array = np.frombuffer(temp_array, dtype=ml_dtypes.bfloat16)
			else:
				num_elements = array_size // ctypes.sizeof(dtype)
				host_array = memcpy_d2h(ptr, num_elements, dtype)
		else:
			raise TypeError(f"Unsupported pointer type: {arg_type}")

		logging.debug(f"Received data from IPC ({arg_type}/{len(host_array)}): {host_array}")
		results.append(host_array)
	return results
