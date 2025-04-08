
import subprocess
import io
import selectors
import logging
import sys
import json

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
