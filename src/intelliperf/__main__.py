#!/usr/bin/env python3
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


def intelliperf_parser():
	import argparse

	parser = argparse.ArgumentParser(
		description="Optimize and analyze the given application based on available IntelliPerf formulas.",
		prog="intelliperf",
		formatter_class=lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=30),
		usage="""
        intelliperf [options] -- <profile_cmd>

        Example:
        # Run intelliperf to optimize bank conflicts in a HIP app
        intelliperf -b ~/rocBLAS/build.sh -f bankConflict -- ~/rocBLAS/build/bin/rocblas_gemm
        # Run intelliperf to diagnose a Triton app
        intelliperf -- python3 gemm.py
        """,
	)
	parser.add_argument(
		"-v",
		"--verbose",
		action="count",
		default=0,
		help="Increase verbosity level (e.g., -v, -vv, -vvv).",
	)

	# Required arguments group
	required_args = parser.add_argument_group("required arguments")
	required_args.add_argument(
		"remaining",
		metavar="-- [ ...]",
		nargs=argparse.REMAINDER,
		help="Provide the command to launch the application.",
	)

	# Optional arguments group
	optional_args = parser.add_argument_group("optional arguments")
	# Input arguments
	optional_args.add_argument(
		"-b",
		"--build_command",
		type=str,
		metavar="",
		help="A command to build your application. When project_directory is provided,\nthe command must be relative to the project directory.",
	)
	optional_args.add_argument(
		"-i",
		"--instrument_command",
		type=str,
		metavar="",
		help="A command to instrument your application (required when formula is not diagnoseOnly). When project_directory is provided,\nthe command must be relative to the project directory.",
	)
	optional_args.add_argument(
		"-p",
		"--project_directory",
		type=str,
		metavar="",
		help="The directory containing your entire codebase (required when formula is not diagnoseOnly)",
	)
	optional_args.add_argument(
		"-f",
		"--formula",
		choices=["bankConflict", "memoryAccess", "atomicContention", "diagnoseOnly"],
		default="diagnoseOnly",
		metavar="",
		type=str,
		help="Specify the formula to use for optimization.\nAvailable options: bankConflict, memoryAccess, atomicContention, diagnoseOnly (default: diagnoseOnly)",
	)
	optional_args.add_argument(
		"--top_n",
		type=int,
		default=10,
		metavar="",
		help="Control the top-n kernels collected in diagnoseOnly mode (default: 10)",
	)
	optional_args.add_argument(
		"--trace_path",
		type=str,
		default="trace",
		metavar="PATH",
		help="Enable comprehensive tracing and save trace logs to specified path (e.g., --trace_path logs/run_1.json)",
	)
	optional_args.add_argument(
		"--num_attempts",
		type=int,
		default=10,
		metavar="",
		help="Control the number of max optimization attempts (default: 10)",
	)
	optional_args.add_argument(
		"-t",
		"--accordo_absolute_tolerance",
		type=float,
		default=1e-6,
		metavar="",
		help="Control the tolerance for the Accordo validation (default: 1e-6)",
	)
	optional_args.add_argument(
		"--unittest_command",
		type=str,
		metavar="",
		help="Command to run unit tests for correctness validation; must be relative to the project directory.",
	)
	optional_args.add_argument(
		"-m",
		"--model",
		type=str,
		default="gpt-4o",
		metavar="",
		help="Specify the model to use for optimization (default: gpt-4o-mini)",
	)
	optional_args.add_argument(
		"-r",
		"--provider",
		type=str,
		default="openai",
		metavar="",
		help="Specify the provider to use for optimization (default: openai)",
	)
	optional_args.add_argument(
		"-l",
		"--in_place",
		action="store_true",
		help="Modify source files in place during optimization (default: false)",
	)

	# Output arguments
	optional_args.add_argument("-o", "--output_file", type=str, metavar="", help="Path to the output file")

	args = parser.parse_args()

	# Validate that project_directory is provided when formula is not diagnoseOnly
	if args.formula != "diagnoseOnly" and not args.project_directory:
		parser.error(
			"--project_directory is required when --formula is not diagnoseOnly and all paths must be relative to the project directory."
		)

	if args.remaining and "--" in args.remaining:
		args.remaining = args.remaining[1:]
	else:
		parser.error("Profiling command required. Pass application executable after -- at the end of options.")

	return args


def main():
	args = intelliperf_parser()

	# Generate a name for the IntelliPerf run using timestamp
	from datetime import datetime

	generated_name = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

	# Set logging level based on verbosity
	import logging

	logging.raiseExceptions = True
	if args.verbose == 1:
		logging.basicConfig(level=logging.INFO, format="[INTELLIPERF] %(levelname)s: %(message)s")
	elif args.verbose == 2:
		logging.basicConfig(level=logging.DEBUG, format="[INTELLIPERF] %(levelname)s: %(message)s")
	elif args.verbose >= 3:
		logging.basicConfig(level=logging.NOTSET, format="[INTELLIPERF] %(levelname)s: %(message)s")
	else:
		logging.basicConfig(level=logging.WARNING, format="[INTELLIPERF] %(levelname)s: %(message)s")

	# Create an optimizer based on available IntelliPerf formulas.
	if args.formula == "bankConflict":
		from intelliperf.formulas.bank_conflict import bank_conflict

		formula = bank_conflict
	elif args.formula == "diagnoseOnly":
		from intelliperf.formulas.diagnose_only import diagnose_only

		formula = diagnose_only
	elif args.formula == "memoryAccess":
		from intelliperf.formulas.memory_access import memory_access

		formula = memory_access
	elif args.formula == "atomicContention":
		from intelliperf.formulas.atomic_contention import atomic_contention

		formula = atomic_contention
	else:
		logging.error(f"Invalid formula specified. {args.formula} is not supported.")
		import sys

		sys.exit(1)

	optimizer = formula(
		name=generated_name,
		build_command=args.build_command,
		instrument_command=args.instrument_command,
		project_directory=args.project_directory,
		app_cmd=args.remaining,
		top_n=args.top_n,
		model=args.model,
		provider=args.provider,
		in_place=args.in_place,
		unittest_command=args.unittest_command,
	)

	# Helper function to flush logs if tracing is enabled
	def flush_logs_if_enabled():
		print(f"args.trace_path: {args.trace_path}")
		if hasattr(optimizer, "get_logger") and args.trace_path:
			print(f"Flushing logs to {args.trace_path}")
			optimizer.get_logger().flush(args.trace_path)

	num_attempts = 0 if args.formula == "diagnoseOnly" else args.num_attempts

	# Build the application
	optimizer.build()

	# Profile the application and collect the results.
	optimizer.profile_pass()
	flush_logs_if_enabled()  # Flush after profiling

	# Get source code mappings
	optimizer.source_code_pass()
	flush_logs_if_enabled()  # Flush after source code collection

	# Instrument the application based on the results.
	optimizer.instrument_pass()
	flush_logs_if_enabled()  # Flush after instrumentation

	# Initialize performance_result for diagnoseOnly case
	performance_result = None

	for attempt in range(num_attempts):
		logging.info(f"Executing pass {attempt + 1} of {num_attempts}.")

		# Optimize the application based on insights from instrumentation.
		optimize_result = optimizer.optimize_pass()
		if not optimize_result:
			optimize_result.report_out()
			logging.warning(f"Optimization pass {attempt + 1} failed. Retrying...")
			flush_logs_if_enabled()  # Flush after failed optimization
			continue

		# Compile the new application
		build_result = optimizer.build_pass(validate_build_result=False)
		if not build_result:
			build_result.report_out()
			logging.warning(f"Build pass {attempt + 1} failed. Retrying...")
			flush_logs_if_enabled()  # Flush after failed build
			continue

		# Validate the new application
		correctness_result = optimizer.correctness_validation_pass(
			accordo_absolute_tolerance=args.accordo_absolute_tolerance
		)
		if not correctness_result:
			correctness_result.report_out()
			logging.warning(f"Correctness validation pass {attempt + 1} failed. Retrying...")
			flush_logs_if_enabled()  # Flush after failed correctness validation
			continue

		performance_result = optimizer.performance_validation_pass()
		if not performance_result:
			performance_result.report_out()
			logging.warning(f"Performance validation pass {attempt + 1} failed. Retrying...")
			flush_logs_if_enabled()  # Flush after failed performance validation
			continue

		# If the optimization is successful, exit the loop
		if performance_result:
			flush_logs_if_enabled()  # Flush after successful optimization
			break

	import sys

	try:
		if args.formula == "diagnoseOnly" or performance_result:
			# Flush logger if tracing is enabled
			if hasattr(optimizer, "get_logger") and args.trace_path:
				logger = optimizer.get_logger()
				logger.flush(args.trace_path)

			optimizer.write_results(args.output_file)
			sys.exit(0)
	except Exception as e:
		logging.error(f"Error writing results: {e}")
		# Try to flush logger even if results writing fails
		try:
			if hasattr(optimizer, "get_logger") and args.trace_path:
				logger = optimizer.get_logger()
				logger.flush(args.trace_path)
		except Exception as log_error:
			logging.error(f"Error flushing logger: {log_error}")
		sys.exit(1)


if __name__ == "__main__":
	main()
