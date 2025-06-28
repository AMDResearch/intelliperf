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

import io
import logging
import os
import selectors
import subprocess
import sys


def exit_on_fail(success: bool, message: str, log: str = ""):
	if not success:
		full_msg = f"{message}\n{log.strip()}" if log.strip() else message
		logging.error("Critical Error: %s", full_msg)
		sys.exit(1)


def capture_subprocess_output(
	subprocess_args: list, working_directory: str = None, new_env=None, additional_path=None
) -> tuple:
	"""
	Simple version that just runs the process and waits for completion.
	"""
	verbose = logging.getLogger().getEffectiveLevel() <= logging.DEBUG

	logging.debug(f"Running the command: {' '.join(subprocess_args)}")

	if working_directory is not None:
		logging.debug(f"Working directory: {working_directory}")

	# Create the environment
	env = new_env.copy() if new_env else os.environ.copy()
	if working_directory is not None:
		env["PWD"] = working_directory

	if additional_path is not None:
		env["PATH"] = str(additional_path) + ":" + env["PATH"]

	logging.debug(f"PATH: {env['PATH']}")

	# Run the process and wait for completion
	try:
		result = subprocess.run(
			subprocess_args,
			cwd=working_directory,
			env=env,
			capture_output=True,
			text=True,
			encoding="utf-8",
			errors="replace",
		)

		success = result.returncode == 0
		output = result.stdout + result.stderr

		if verbose:
			print(output)

		return (success, output)

	except Exception as e:
		logging.error(f"Failed to run command: {e}")
		return (False, str(e))


def capture_subprocess_output_v0(
	subprocess_args: list, working_directory: str = None, new_env=None, additional_path=None
) -> tuple:
	verbose = logging.getLogger().getEffectiveLevel() <= logging.DEBUG

	logging.debug(f"Running the command: {' '.join(subprocess_args)}")

	if working_directory is not None:
		logging.debug(f"Working directory: {working_directory}")
	# Create the environment with working directory
	env = new_env.copy() if new_env else os.environ.copy()
	if working_directory is not None:
		env["PWD"] = working_directory

	if additional_path is not None:
		env["PATH"] = str(additional_path) + ":" + env["PATH"]

	logging.debug(f"PATH: {env['PATH']}")

	# Start subprocess
	# bufsize = 1 means output is line buffered
	# universal_newlines = True is required for line buffering
	process = subprocess.Popen(
		subprocess_args,
		bufsize=1,
		stdout=subprocess.PIPE,
		stderr=subprocess.STDOUT,
		universal_newlines=True,
		encoding="utf-8",
		errors="replace",
		env=env,
		cwd=working_directory,
	)

	# Create callback function for process output
	buf = io.StringIO()

	def handle_output(stream, mask):
		try:
			# Because the process' output is line buffered, there's only ever one
			# line to read when this function is called
			line = stream.readline()
			buf.write(line)
			if verbose:
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
		if verbose:
			for line in remaining.splitlines():
				print(line.strip())

	# Get process return code
	return_code = process.wait()
	selector.close()
	success = return_code == 0

	# Store buffered output
	output = buf.getvalue()
	buf.close()

	# Execute a sync to ensure the output is written to disk
	subprocess.run(["sync"])

	return (success, output)
