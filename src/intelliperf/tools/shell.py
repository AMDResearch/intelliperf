"""Shell command execution tool."""
from pathlib import Path
from typing import List
import subprocess
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ShellInput(BaseModel):
    """Input for shell tool."""
    command: str = Field(description="Shell command to execute")
    working_dir: str = Field(default=".", description="Working directory (relative to project root)")


def shell_impl(work_dir: Path, command: str, working_dir: str = ".") -> str:
    """Execute a shell command and return output."""
    try:
        # Resolve working directory
        if working_dir == ".":
            cwd = work_dir
        else:
            cwd = work_dir / working_dir

        if not cwd.exists():
            return f"Error: Directory {working_dir} not found"

        # Execute command
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=str(cwd),
            timeout=120
        )

        output = []
        output.append(f"Exit code: {result.returncode}")

        if result.stdout:
            output.append(f"\n=== STDOUT ===\n{result.stdout}")

        if result.stderr:
            output.append(f"\n=== STDERR ===\n{result.stderr}")

        return "\n".join(output)

    except subprocess.TimeoutExpired:
        return "Error: Command timeout after 120s"
    except Exception as e:
        return f"Error: {e}"


def create_shell_tools(work_dir: Path) -> List[StructuredTool]:
    """Create shell execution tool."""
    return [
        StructuredTool.from_function(
            func=lambda command="", cmd="", working_dir=".", cwd=".", directory=".", **kwargs: shell_impl(
                work_dir, 
                command or cmd, 
                working_dir or cwd or directory
            ),
            name="shell",
            description="Execute a shell command. Use this to explore files (ls, find, cat), build (make, cmake), run executables, etc. Returns stdout, stderr, and exit code.",
            args_schema=ShellInput
        ),
    ]
