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


import subprocess

try:
	import tomllib
except ModuleNotFoundError:
	import tomli as tomllib

from pathlib import Path


def run_command(cmd: str, cwd: Path):
	print(f"Running in {cwd}:\n{cmd}")
	subprocess.run(cmd, cwd=cwd, shell=True, check=True, executable="/bin/bash")


def get_current_branch_or_hash(cwd: Path) -> str:
	result = subprocess.run(
		"git rev-parse --abbrev-ref HEAD", cwd=cwd, shell=True, check=True, capture_output=True, text=True
	)
	branch_or_hash = result.stdout.strip()
	if branch_or_hash == "HEAD":
		# Detached HEAD state, get the commit hash
		result = subprocess.run("git rev-parse HEAD", cwd=cwd, shell=True, check=True, capture_output=True, text=True)
		branch_or_hash = result.stdout.strip()
	return branch_or_hash


def is_commit_hash(value: str) -> bool:
	return len(value) == 40 and all(c in "0123456789abcdef" for c in value)


def install_tool(tool: str, config: dict, clean: bool):
	tool_data = config.get("tool", {}).get(tool)
	if not tool_data:
		raise RuntimeError(f"[tool.{tool}] section not found in pyproject.toml")

	build_command = tool_data["build_command"]
	repo = tool_data.get("git", None)
	branch_or_hash = tool_data.get("branch", "main")

	if repo:
		tool_dir = Path("external") / tool
		tool_dir.parent.mkdir(exist_ok=True)

		if clean:
			print(f"🧹 Deleting existing {tool} from {tool_dir}.")
			run_command(f"rm -rf {tool}", cwd="external")

		if not tool_dir.exists():
			print(f"Cloning {tool} from {repo}")
			run_command(f"git clone --recurse-submodules {repo} {tool}", cwd="external")
			print(f"Checking out {branch_or_hash}")
			run_command(f"git checkout {branch_or_hash}", cwd=tool_dir)
			run_command("git submodule update --init --recursive", cwd=tool_dir)
		else:
			print(f"Found existing checkout at {tool_dir}, verifying branch or hash")
			current_branch_or_hash = get_current_branch_or_hash(tool_dir)
			if current_branch_or_hash != branch_or_hash:
				print(
					f"Branch/hash mismatch: expected {branch_or_hash}, found {current_branch_or_hash}. Switching to expected branch/hash."
				)
				run_command("git fetch", cwd=tool_dir)
				if is_commit_hash(branch_or_hash):
					run_command(f"git checkout {branch_or_hash}", cwd=tool_dir)
				else:
					run_command(f"git checkout {branch_or_hash}", cwd=tool_dir)
					run_command("git pull", cwd=tool_dir)
				run_command("git submodule update --init --recursive", cwd=tool_dir)
			else:
				print(f"Branch/hash matches: {current_branch_or_hash}")

	else:
		tool_dir = tool
		print(f"Using local subdirectory for '{tool}' (no git clone)")

	print(f"Building {tool}")
	run_command(build_command, cwd=tool_dir)


def main():
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("tool", nargs="?", help="Tool name (e.g., omniprobe)")
	parser.add_argument("-a", "--all", action="store_true", help="Install all tools listed in [tool.*]")
	parser.add_argument("-c", "--clean", action="store_true", help="Clean before installing")
	args = parser.parse_args()

	pyproject_path = Path("pyproject.toml")
	if not pyproject_path.exists():
		raise FileNotFoundError("Could not find pyproject.toml")

	with open(pyproject_path, "rb") as f:
		config = tomllib.load(f)

	run_command("mkdir -p external", cwd=".")
	# This is a workaround to avoid docker errors when cloning repos
	run_command("git config --global --add safe.directory '*'", cwd=".")

	if args.all:
		tools = config.get("tool", {}).keys()

		for tool in tools:
			print(f"\n=== Installing '{tool}' ===")
			tool_data = config.get("tool", {}).get(tool)
			if tool_data.get("build_command"):
				install_tool(tool, config, args.clean)
			else:
				print(f"Skipping '{tool}' as it does not have a build command")
	elif args.tool:
		install_tool(args.tool, config, args.clean)
	else:
		parser.print_help()


if __name__ == "__main__":
	main()
