import os
import ctypes
import time
import subprocess
import struct
import numpy as np
import stat
import logging

PIPE_NAME = "/tmp/kernel_pipe"
HANDSHAKE_PIPE = "/tmp/handshake_pipe"
FILE_NAME = "/tmp/ipc_handle.bin"


logging.basicConfig(level=logging.DEBUG,
                    format="[Python] [%(asctime)s]: %(message)s")

rt_path = "libamdhip64.so"
hip_runtime = ctypes.cdll.LoadLibrary(rt_path)


def prepare_ipc_file():
    for file in [FILE_NAME, HANDSHAKE_PIPE, FILE_NAME]:
        if os.path.exists(file):
            os.remove(file)
    logging.debug("IPC file cleared before execution.")



def run_subprocess(args, path):
    result = subprocess.run(
        args, cwd=path, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if result.returncode != 0:
        logging.error(f"Subprocess failed with return code {result.returncode}")
        logging.error(result.stderr)
        exit(result.returncode)


def hip_try(err):
    if err != 0:
        hip_runtime.hipGetErrorString.restype = ctypes.c_char_p
        error_string = hip_runtime.hipGetErrorString(ctypes.c_int(err)).decode("utf-8")
        raise RuntimeError(f"HIP error code {err}: {error_string}")


class hipIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def open_ipc_handle(ipc_handle_data):
    ptr = ctypes.c_void_p()
    hipIpcMemLazyEnablePeerAccess = ctypes.c_uint(1)
    hip_runtime.hipIpcOpenMemHandle.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        hipIpcMemHandle_t,
        ctypes.c_uint,
    ]
    if isinstance(ipc_handle_data, np.ndarray):
        if ipc_handle_data.dtype != np.uint8 or ipc_handle_data.size != 64:
            logging.debug(f"ipc_handle_data.size: {ipc_handle_data.size}")
            raise ValueError("ipc_handle_data must be a 64-element uint8 numpy array")
        ipc_handle_bytes = ipc_handle_data.tobytes()
        ipc_handle_data = (ctypes.c_char * 64).from_buffer_copy(ipc_handle_bytes)
    else:
        raise TypeError(
            "ipc_handle_data must be a numpy.ndarray of dtype uint8 with 64 elements"
        )

    raw_memory = ctypes.create_string_buffer(64)
    ctypes.memset(raw_memory, 0x00, 64)
    ipc_handle_struct = hipIpcMemHandle_t.from_buffer(raw_memory)
    ipc_handle_data_bytes = bytes(ipc_handle_data)
    ctypes.memmove(raw_memory, ipc_handle_data_bytes, 64)

    logging.debug(f"[ipc_handle_struct]:")
    for i in range(0, len(ipc_handle_data_bytes), 16):
        chunk = ipc_handle_data_bytes[i : i + 16]
        logging.debug(" ".join(f"{b:02x}" for b in chunk))

    hip_try(
        hip_runtime.hipIpcOpenMemHandle(
            ctypes.byref(ptr),
            ipc_handle_struct,
            hipIpcMemLazyEnablePeerAccess,
        )
    )

    return ptr.value


def get_device_buffer(ptr_value, size):
    if not ptr_value:
        raise RuntimeError("Invalid device pointer: NULL received.")

    buffer_ptr = ctypes.cast(ptr_value, ctypes.POINTER(ctypes.c_uint8 * size))
    np_array = np.ctypeslib.as_array(buffer_ptr.contents)
    return np_array

def generate_header(args: list[str]) -> str:
    header_path = "/tmp/KernelArguments.hpp"
    member_names = [arg.split()[-1] for arg in args]
    members = ";\n    ".join(args) + ";"
    as_tuple_members = ", ".join(member_names)

    header_content = f"""#pragma once
#include <tuple>
struct KernelArguments {{
    {members}

    auto as_tuple() const {{
        return std::tie({as_tuple_members});
    }}
}};
"""
    with open(header_path, "w") as header_file:
        header_file.write(header_content)
    return header_path


def read_args_pointer():
    with open(PIPE_NAME, "r") as fifo:
        ptr_str = fifo.readline().strip()
    return int(ptr_str, 16)


def read_ipc_handle(fifo):
    data = bytearray()
    while len(data) < 64:
        chunk = fifo.read(8)
        if not chunk:
            break
        data.extend(chunk)
        logging.debug(f"Received {len(chunk)} bytes: {' '.join(f'{b:02x}' for b in chunk)}")

    logging.debug(f"Final IPC Handle (hex): {' '.join(f'{b:02x}' for b in data)}")
    handle = np.frombuffer(data, dtype=np.uint8)
    return handle


def read_ipc_handles(args):
    count = sum(1 for arg in args if "*" in arg)
    handles = []
    sizes = []
    handles_set = set()

    while len(handles) < count:
        if not os.path.exists(FILE_NAME):
            logging.debug("Waiting for IPC file...")
            time.sleep(0.1)
            continue

        with open(FILE_NAME, "rb") as file:
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

                        # Convert size from bytes to an integer
                        size_value = int.from_bytes(size_data, byteorder="little")
                        sizes.append(size_value)

                        logging.debug("Final IPC Handle (hex):")
                        for i in range(0, len(handle_np), 16):
                            chunk = handle_np[i : i + 16]
                            print(" ".join(f"{b:02x}" for b in chunk))

                        logging.debug(f"Corresponding Pointer Size: {size_value} bytes")

        if len(handles) < count:
            logging.debug(f"Waiting for {count - len(handles)} more IPC handles...")
            time.sleep(0.1)

    logging.debug(f"Successfully read {len(handles)} IPC handles and sizes.")
    return handles, sizes

def send_response():
    with open(PIPE_NAME, "w") as fifo:
        fifo.write("done\n")


tracer_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.dirname(tracer_directory)

print(f"{tracer_directory=}")
print(f"{project_directory=}")

# app = "dummy"
# kernel = "emptyKernel"'
# args=["int* arg0"]
# generate_header(args)

app = "initial_code"
kernel = "reduction_kernel"
args = ["double* arg0", "double* arg1", "std::size_t arg2"]
generate_header(args)


tracer_src_directory = project_directory
binary = os.path.join("/tmp", app)

run_subprocess(["hipcc", "-o", f"{binary}", f"{tracer_directory}/hip/{app}.hip"], "/tmp")
run_subprocess(["cmake", "-B", "build"], tracer_src_directory)
run_subprocess(["cmake", "--build", "build"], tracer_src_directory)

lib = os.path.join(project_directory, "build", "lib", "libtracer.so")


env = os.environ.copy()
env["HSA_TOOLS_LIB"] = lib
env["KERNEL_TO_TRACE"] = kernel
env["TRACER_LOG_LEVEL"] = "detail"

# Remove IPC handles
prepare_ipc_file()


if not os.path.exists(PIPE_NAME):
    os.mkfifo(PIPE_NAME)
    os.chmod(PIPE_NAME, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

pid = os.posix_spawn(binary, [binary], env)

with open(PIPE_NAME, "rb") as fifo:
    ipc_handles, ptr_sizes = read_ipc_handles(args)

type_map = {
    "double*": (ctypes.POINTER(ctypes.c_double), ctypes.sizeof(ctypes.c_double)),
    "float*": (ctypes.POINTER(ctypes.c_float), ctypes.sizeof(ctypes.c_float)),
    "int*": (ctypes.POINTER(ctypes.c_int), ctypes.sizeof(ctypes.c_int)),
    "std::size_t*": (ctypes.POINTER(ctypes.c_size_t), ctypes.sizeof(ctypes.c_size_t)),
}


typed_pointers = []
for arg, handle, array_size in zip(args, ipc_handles, ptr_sizes):
    if "*" in arg:
        ptr = open_ipc_handle(handle)
        arg_type = arg.split()[0]
        if arg_type in type_map:
            ctype_ptr, type_size = type_map[arg_type]
            typed_ptr = ctypes.cast(ptr, ctype_ptr)
        else:
            raise TypeError(f"Unsupported pointer type: {arg_type}")

        typed_pointers.append(typed_ptr)
        num_elements = array_size // type_size
        max_num_elements=32

        num_elements_to_copy = min(max_num_elements, num_elements)
        host_array = np.zeros(num_elements_to_copy, dtype=np.dtype(ctype_ptr._type_))
        hipMemcpyDeviceToHost = 2,
        hip_runtime.hipMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ]
        result = hip_runtime.hipMemcpy(
            ctypes.c_void_p(host_array.ctypes.data),
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(num_elements_to_copy * type_size),
            2
        )


        data = np.ctypeslib.as_array(typed_ptr, shape=(min(max_num_elements, num_elements),))
        logging.info(f"Received data from IPC ({arg_type}/{num_elements}): {host_array}")
        logging.debug(f"Opened IPC Ptr: {ptr}")


send_response()

logging.info("Python processing complete")

# os.waitpid(pid, 0)
