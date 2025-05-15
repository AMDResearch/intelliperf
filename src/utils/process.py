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

import subprocess
import io
import selectors
import logging
import sys
import json
import tempfile
import os

def exit_on_fail(success: bool, message: str, log: str = ""):
    if not success:
        full_msg = f"{message}\n{log.strip()}" if log.strip() else message
        logging.error("Critical Error: %s", full_msg)
        sys.exit(1)
    
def capture_subprocess_output(subprocess_args:list, new_env=None) -> tuple:
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
    logging.debug(f"Running the command: {' '.join(subprocess_args)}")
    if new_env != None:
        logging.debug("Inside the environment:\n%s", json.dumps(new_env, indent=2))
    
    # Dump to a tmp shell script for debugging
    tmp_dir = tempfile.mkdtemp()
    tmp_script = os.path.join(tmp_dir, "subprocess_args.sh")
    
    # Wirte the env to the tmp script
    with open(tmp_script, "w") as f:
        new_env = new_env if new_env != None else os.environ
        f.write("#!/bin/bash\n")
        for key, value in new_env.items():
            f.write(f"{key}=\"{value}\"\n")
    
    # Write the command to the tmp script
    with open(tmp_script, "a") as f:
        for arg in subprocess_args:
            if " " in arg:
                f.write(f'"{arg}" ')
            else:
                f.write(f"{arg} ")
        f.write("\n")
    f.close()
    
    # Chmod +x the tmp script
    os.chmod(tmp_script, 0o755)
    
    logging.debug(f"Dumped script to: {tmp_script}")

    process = (
        subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
        )
        if new_env == None
        else subprocess.Popen(
            subprocess_args,
            bufsize=1,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
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

    # If the process terminated, capture any output that remains.
    remaining = process.stdout.read()
    if remaining:
        buf.write(remaining)
        for line in remaining.splitlines():
            print(line.strip())
             
    # Get process return code
    return_code = process.wait()
    selector.close()
    success = return_code == 0

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)
