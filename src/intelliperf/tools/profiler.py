"""GPU profiling tools using Metrix."""
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import List
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class ProfileInput(BaseModel):
    """Input for profile tool."""
    executable_path: str = Field(default="", description="Path to executable or full command (e.g., './app', 'python3 script.py --args')", alias="executable")
    command: str = Field(default="", description="Alias for executable_path (some agents use 'command').")
    num_replays: int = Field(default=5, description="Number of times to replay kernels for averaging")
    metrics: List[str] = Field(default=[], description="List of specific metrics to collect. Leave empty for profiling with all current metrics.")
    profile: str = Field(default="", description="Use a preset profile: 'quick', 'memory', 'memory_bandwidth', 'memory_cache', 'compute'. Overrides metrics if specified. Use list_available_metrics to see all options.")
    kernel_filter: str = Field(default="", description="Kernel name substring to filter results")


class ListMetricsInput(BaseModel):
    """Input for list_available_metrics tool."""
    pass


def _load_bottleneck_diagnosis():
    """
    Load custom bottleneck diagnosis function if available.
    Tries bottleneck_diagnosis.py (user customization, not tracked), otherwise no diagnosis.
    """
    from pathlib import Path
    import importlib.util
    
    workspace_root = Path(__file__).parent.parent.parent.parent
    module_file = workspace_root / "bottleneck_diagnosis.py"
    
    if module_file.exists():
        try:
            spec = importlib.util.spec_from_file_location("bottleneck_diagnosis", module_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, 'diagnose_bottleneck'):
                return module.diagnose_bottleneck
        except Exception as e:
            print(f"Warning: Failed to load bottleneck diagnosis from bottleneck_diagnosis.py: {e}")
    
    # Fallback: no diagnosis
    return lambda metrics: []


def profile_impl(work_dir: Path, executable_path: str, num_replays: int = 5, metrics: List[str] = None, profile: str = "", kernel_filter: str = "") -> str:
    """Profile a GPU kernel using Metrix."""
    from metrix import Metrix

    if not executable_path:
        return (
            "Error: No executable provided to profile().\n"
            "Tip: Pass `executable` (or `executable_path`) like './app' or 'python3 script.py --args'.\n"
            "Common pitfall: passing `command` instead of `executable` (now supported), or forgetting './' for local binaries."
        )

    # Profile with Metrix
    metrix = Metrix()

    # Get unsupported metrics from backend (automatically extracted from @metric decorator)
    unsupported_metrics = list(metrix.backend._unsupported_metrics.keys())

    # If proprietary metrics are available, include them with the default memory profile
    proprietary_metrics = [m for m in metrix.list_metrics() if m.startswith("proprietary.")]
    if proprietary_metrics and not metrics and not profile:
        try:
            from metrix.metrics.catalog import METRIC_PROFILES
            base_metrics = METRIC_PROFILES.get("memory", {}).get("metrics", [])
            # Filter out unsupported metrics
            base_metrics = [m for m in base_metrics if m not in unsupported_metrics]
            metrics = list(dict.fromkeys(base_metrics + proprietary_metrics))
            profile = ""
        except Exception:
            # Fallback: just use proprietary metrics if catalog isn't available
            metrics = proprietary_metrics
            profile = ""

    
    # Check if this is a full command (contains spaces/arguments) or just an executable path
    is_full_command = ' ' in executable_path or executable_path.endswith('.py')
    
    if is_full_command:
        # Full command string - pass directly to Metrix (supports "python3 script.py --args")
        # Auto-prepend python3 for .py files if not already present
        if executable_path.endswith('.py') and not executable_path.startswith('python'):
            command_str = f"python3 {executable_path}"
        else:
            command_str = executable_path
    else:
        # Single executable path - validate and resolve
        exe_full_path = Path(executable_path)
        if not exe_full_path.is_absolute():
            exe_full_path = work_dir / executable_path

        if not exe_full_path.exists():
            return (f"Error: Executable not found at '{exe_full_path}'\n"
                    f"Working directory: {work_dir}\n"
                    f"Tip: Use 'shell' to run 'ls' or 'find . -name <binary_name>' to locate the executable.\n"
                    f"Tip: If the binary is in the current directory, run it as './<name>' (and profile executable='./<name>').")

        if exe_full_path.is_dir():
            return (
                f"Error: profile() expected an executable file, but got a directory: '{exe_full_path}'.\n"
                f"Tip: Make sure you're passing the binary path (e.g. './conv1d_bench'), not the project directory.\n"
                f"Tip: The profile tool argument name is `executable` (aliases accepted: `executable_path`, `command`)."
            )
        
        # Handle Python scripts
        if str(exe_full_path).endswith('.py'):
            command_str = f"python3 {exe_full_path}"
        else:
            command_str = str(exe_full_path)

    # Build profile args
    profile_args = {
        "command": command_str,
        "num_replays": num_replays,
        "aggregate_by_kernel": True,
        "cwd": str(work_dir),
        "timeout_seconds": 600  # Increased to handle multiple profiling passes with replays
    }

    # ALWAYS collect comprehensive metrics (never time-only)
    if metrics and len(metrics) > 0:
        profile_args["metrics"] = metrics
    elif profile:
        profile_args["profile"] = profile
    else:
        # Default to comprehensive memory+compute metrics
        profile_args["profile"] = "memory"

    if kernel_filter:
        profile_args["kernel_filter"] = kernel_filter

    # Print profiling status to stdout (for tee capture)
    print(f"Profiling {profile_args.get('command', 'executable')} with {profile_args['num_replays']} replays...", flush=True)
    
    results = metrix.profile(**profile_args)
    print("✓ Profiling completed successfully", flush=True)

    # NEW: Extract self-reported timing if kernel prints it
    self_reported_times = []
    try:
        # Re-run command to capture stdout
        import subprocess
        result = subprocess.run(
            command_str,
            shell=True,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=30
        )
        # Look for patterns like "Average time: 0.011 ms" or "Time: 1.23 ms"
        time_patterns = [
            r'Average time:\s*(\d+\.\d+)\s*ms',
            r'Time:\s*(\d+\.\d+)\s*ms',
            r'Elapsed:\s*(\d+\.\d+)\s*ms',
            r'Duration:\s*(\d+\.\d+)\s*ms'
        ]
        for pattern in time_patterns:
            matches = re.findall(pattern, result.stdout)
            if matches:
                self_reported_times.extend([float(m) for m in matches])
                break
    except Exception as e:
        # Not critical if we can't extract self-reported time
        pass

    # Calculate average self-reported time if available
    avg_self_reported = None
    if self_reported_times:
        avg_self_reported = sum(self_reported_times) / len(self_reported_times)


    if not results.kernels:
        return "Error: No kernels profiled"

    # Filter kernels: if kernel_filter is provided, only show matching kernels
    if kernel_filter:
        matching_kernels = [k for k in results.kernels if kernel_filter.lower() in k.name.lower()]
        ignored_kernels = [k for k in results.kernels if kernel_filter.lower() not in k.name.lower()]
    else:
        # No filter - show all kernels
        matching_kernels = results.kernels
        ignored_kernels = []
    
    output_lines = ["Profile Results:\n"]
    
    if not matching_kernels:
        output_lines.append(f"⚠️  No kernels matched filter '{kernel_filter}'")
        output_lines.append(f"Found {len(results.kernels)} kernel(s) total, but none matched the filter.\n")
        return "\n".join(output_lines)
    
    # Show summary for matching kernels
    output_lines.append("=" * 60)
    output_lines.append(f"📊 PROFILING RESULTS ({len(matching_kernels)} kernel(s)):")
    for kernel in matching_kernels:
        time_ms = kernel.duration_us.avg / 1000.0
        count = kernel.duration_us.count
        total_ms = kernel.duration_us.avg * count / 1000.0
        output_lines.append(f"  • {kernel.name}")
        output_lines.append(f"    {time_ms:.3f} ms/call × {count} calls = {total_ms:.1f} ms total")
    output_lines.append("=" * 60)
    output_lines.append("")
    
    # Note about ignored kernels
    if ignored_kernels:
        ignored_names = [k.name for k in ignored_kernels[:3]]  # Show first 3
        if len(ignored_kernels) > 3:
            ignored_names.append(f"... and {len(ignored_kernels) - 3} more")
        output_lines.append(f"ℹ️  {len(ignored_kernels)} other kernel(s) ignored (not matching filter): {', '.join(ignored_names)}\n")
    
    # Show full details for matching kernels only
    for kernel in matching_kernels:
        time_ms = kernel.duration_us.avg / 1000.0
        count = kernel.duration_us.count
        total_ms = kernel.duration_us.avg * count / 1000.0
        
        output_lines.append(f"Kernel: {kernel.name}")
        output_lines.append(f"  Time: {time_ms:.3f} ms/call (avg over {count} runs, {total_ms:.1f} ms total)")

        # Show metrics from the metrics dictionary
        metrics_dict = {}
        if hasattr(kernel, 'metrics') and isinstance(kernel.metrics, dict):
            for metric_name, metric_value in kernel.metrics.items():
                if hasattr(metric_value, 'avg'):
                    val = metric_value.avg
                    metrics_dict[metric_name] = val
                    output_lines.append(f"  {metric_name}: {val:.3f}")
                else:
                    metrics_dict[metric_name] = metric_value
                    output_lines.append(f"  {metric_name}: {metric_value}")
        
        # Add automated bottleneck diagnosis
        if metrics_dict:
            output_lines.append("\n  📊 BOTTLENECK ANALYSIS:")
            diagnose_fn = _load_bottleneck_diagnosis()
            diagnosis = diagnose_fn(metrics_dict)
            for line in diagnosis:
                output_lines.append(f"  {line}")
        

        
        output_lines.append("")

    return "\n".join(output_lines)


def list_metrics_impl() -> str:
    """List all available metrics from Metrix."""
    from metrix import Metrix
    metrix = Metrix()
    available_metrics = metrix.list_metrics()
    available_profiles = metrix.list_profiles()

    output = ["Available GPU performance metrics:\n"]
    output.append("=" * 60)
    output.append("\nPreset Profiles (recommended for quick profiling):")
    for profile in available_profiles:
        output.append(f"  - {profile}")

    output.append("\nIndividual Metrics:")
    for metric in available_metrics:
        output.append(f"  - {metric}")

    output.append("\n" + "=" * 60)
    output.append("\nUsage Examples:")
    output.append("  1. All metrics (default): profile(executable_path='./app')")
    output.append("  2. Use preset: profile(executable_path='./app', profile='compute')")
    output.append("  3. Specific metrics: profile(executable_path='./app', metrics=['memory.l2_hit_rate', 'compute.occupancy_percent'])")
    return "\n".join(output)


def create_profiler_tools(work_dir: Path) -> List[StructuredTool]:
    """Create profiling tools."""
    return [
        StructuredTool.from_function(
            func=lambda executable_path="", executable="", command="", num_replays=10, metrics=[], profile="", kernel_filter="", **kwargs: profile_impl(
                work_dir,
                executable_path or executable or command,
                num_replays,
                metrics if metrics else None,
                profile,
                kernel_filter
            ),
            name="profile",
            description="Profile GPU kernels using Metrix profiler. Supports executables (./app) and Python scripts (python3 script.py --args or just kernel.py). ALWAYS collects comprehensive metrics and provides automated bottleneck diagnosis. IMPORTANT: Always set kernel_filter to the target kernel name from the optimization goal to filter out system kernels and focus only on the user kernel being optimized.",
            args_schema=ProfileInput
        ),
        StructuredTool.from_function(
            func=list_metrics_impl,
            name="list_available_metrics",
            description="List all available GPU performance metrics that can be collected with the profile tool.",
            args_schema=ListMetricsInput
        ),
    ]
