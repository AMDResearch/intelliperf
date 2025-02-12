import subprocess
import os
import ctypes
import time

PIPE_NAME = "/tmp/kernel_pipe"

# Define the struct layout in Python
class KernelArgs(ctypes.Structure):
    _fields_ = [("value", ctypes.c_int)]

def read_args_pointer():
    """Reads the pointer from the named pipe (blocks until data is available)."""
    print("read_args_pointer")
    with open(PIPE_NAME, "r") as fifo:
        ptr_str = fifo.readline().strip()
    return int(ptr_str, 16)  # Convert pointer address from hex to int

def send_response():
    """Writes a response to the named pipe to unblock C code."""
    print("send_response")
    with open(PIPE_NAME, "w") as fifo:
        fifo.write("done\n")

tracer_directory = os.path.dirname(os.path.abspath(__file__))
project_directory = os.path.dirname(tracer_directory)

lib = os.path.join(project_directory, "tracer", "build", "lib", "libtracer.so")

app = "dummy"
binary = os.path.join(tracer_directory, app)

env = os.environ.copy()
env["HSA_TOOLS_LIB"] = lib
env["KERNEL_TO_TRACE"] = "emptyKernel"
env["TRACER_LOG_LEVEL"] = "detail"
env["LD_PRELOAD"] = lib  # If LD_PRELOAD is required

# Step 1: Launch binary as a separate process
binary_process = subprocess.Popen([binary], env=env)

# Step 2: Read the pointer from the named pipe (blocks until C writes)
args_ptr = read_args_pointer()
print(f"Received: {args_ptr}")
print(f"Received: {hex(args_ptr)}")

# Step 3: Cast pointer to struct
# kernel_args = ctypes.cast(args_ptr, ctypes.POINTER(KernelArgs)).contents
# print(f"Received struct: value={kernel_args}")

# Step 4: Send response to unblock C
send_response()

print("Python processing complete. C can continue.")

# Step 5: Wait for the binary to finish execution
binary_process.wait()
