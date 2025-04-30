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


import os
import subprocess

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from pathlib import Path

def run_command(cmd: str, cwd: Path):
    print(f"Running in {cwd}:\n{cmd}")
    subprocess.run(cmd, cwd=cwd, shell=True, check=True, executable="/bin/bash")

def install_tool(tool: str, config: dict, clean: bool):
    tool_data = config.get("tool", {}).get(tool)
    if not tool_data:
        raise RuntimeError(f"[tool.{tool}] section not found in pyproject.toml")

    build_command = tool_data["build_command"]
    repo = tool_data.get("git", None)
    branch = tool_data.get("branch", "main")

    if repo:
        tool_dir = Path("external") / tool
        tool_dir.parent.mkdir(exist_ok=True)

        if clean:
            print(f"🧹 Deleting existing {tool} from {tool_dir}.")
            run_command(f"rm -rf {tool}", cwd="external")

        if not tool_dir.exists():
            print(f"Cloning {tool} from {repo}")
            run_command(f"git clone --recurse-submodules {repo} {tool}", cwd="external")
            print(f"Checking out {branch}")
            run_command(f"git checkout {branch}", cwd=tool_dir)
            run_command(f"git submodule update --init --recursive", cwd=tool_dir)
        else:
            print(f"Found existing checkout at {tool_dir}, skipping clone")
    else:
        tool_dir = tool
        print(f"Using local subdirectory for '{tool}' (no git clone)")

    print(f"Building {tool}")
    run_command(build_command, cwd=tool_dir)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("tool", nargs="?", help="Tool name (e.g., logduration)")
    parser.add_argument("--all", action="store_true", help="Install all tools listed in [tool.*]")
    parser.add_argument("--clean", action="store_true", help="Clean before installing")
    args = parser.parse_args()

    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("Could not find pyproject.toml")

    with open(pyproject_path, "rb") as f:
        config = tomllib.load(f)

    if args.all:
        tools = config.get("tool", {}).keys()
        for tool in tools:
            print(f"\n=== Installing '{tool}' ===")
            install_tool(tool, config, args.clean)
    elif args.tool:
        install_tool(args.tool, config, args.clean)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
