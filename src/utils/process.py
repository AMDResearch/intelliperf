
import subprocess
import io
import selectors

def capture_subprocess_output(subprocess_args:list, new_env=None) -> tuple:
    # Start subprocess
    # bufsize = 1 means output is line buffered
    # universal_newlines = True is required for line buffering
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

    # Get process return code
    return_code = process.wait()
    selector.close()

    success = return_code == 0

    # Store buffered output
    output = buf.getvalue()
    buf.close()

    return (success, output)
