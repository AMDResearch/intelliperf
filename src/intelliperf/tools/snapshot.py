"""Snapshot management tools for tracking optimization attempts."""
from pathlib import Path
from typing import List
import shutil
import json
from datetime import datetime
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class SaveSnapshotInput(BaseModel):
    """Input for save_snapshot tool."""
    name: str = Field(description="Name for this snapshot (e.g., 'baseline', 'shared_mem_opt', 'warp_reduction')")
    description: str = Field(description="Description of what this version does and its performance")
    performance_ms: float = Field(description="Performance in milliseconds (REQUIRED - extract from PROFILER measurement)")
    files: List[str] = Field(description="List of files to save (e.g., ['kernel.hip', 'main.hip']). If empty list [], saves entire directory.")
    measurement_method: str = Field(default="profiler", description="Measurement method used: 'profiler' (default and required), 'self-reported', or 'unknown'")


class ListSnapshotsInput(BaseModel):
    """Input for list_snapshots tool."""
    pass


class RestoreSnapshotInput(BaseModel):
    """Input for restore_snapshot tool."""
    name: str = Field(description="Name of snapshot to restore")


def save_snapshot_impl(work_dir: Path, name: str, description: str, performance_ms: float, files: List[str], measurement_method: str = "profiler") -> str:
    """Save current code state as a named snapshot."""
    try:
        # Create snapshots directory
        snapshots_dir = work_dir / ".intelliperf" / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalize baseline naming and prevent multiple baselines
        if "baseline" in name.lower():
            name = "baseline"
            existing_baselines = [d for d in snapshots_dir.iterdir() if d.is_dir() and "baseline" in d.name.lower()]
            if existing_baselines:
                return f"Error: Baseline already exists at {existing_baselines[0].name}. Use a different name for subsequent optimizations (e.g., 'optimized_v1', 'reduced_bank_conflicts', etc.)"

        # Create snapshot directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_dir = snapshots_dir / f"{name}_{timestamp}"
        snapshot_dir.mkdir(exist_ok=True)

        copied_files = []

        if files and len(files) > 0:
            # Copy only specified files
            for filename in files:
                src = work_dir / filename
                if src.exists() and src.is_file():
                    dest = snapshot_dir / src.name
                    shutil.copy2(src, dest)
                    copied_files.append(src.name)
                else:
                    return f"Error: File not found: {filename}"
        else:
            # Copy entire working directory (excluding .intelliperf)
            for item in work_dir.iterdir():
                if item.name == '.intelliperf':
                    continue
                if item.is_file():
                    shutil.copy2(item, snapshot_dir / item.name)
                    copied_files.append(item.name)
                elif item.is_dir():
                    shutil.copytree(item, snapshot_dir / item.name)
                    copied_files.append(f"{item.name}/")

        # Save metadata
        metadata = {
            "name": name,
            "description": description,
            "performance_ms": performance_ms,
            "measurement_method": measurement_method,
            "timestamp": timestamp,
            "files": copied_files
        }

        with open(snapshot_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        return f"Snapshot saved: {name}\n  Location: .intelliperf/snapshots/{snapshot_dir.name}\n  Files: {', '.join(copied_files)}\n  Performance: {performance_ms:.4f}ms"

    except Exception as e:
        return f"Error saving snapshot: {e}"


def list_snapshots_impl(work_dir: Path) -> str:
    """List all saved snapshots with their metadata."""
    try:
        snapshots_dir = work_dir / ".intelliperf" / "snapshots"
        if not snapshots_dir.exists():
            return "No snapshots found. Use save_snapshot to create one."

        snapshots = []
        for snapshot_dir in sorted(snapshots_dir.iterdir()):
            if snapshot_dir.is_dir():
                metadata_file = snapshot_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    snapshots.append({
                        "dir": snapshot_dir.name,
                        **metadata
                    })

        if not snapshots:
            return "No snapshots found."

        # Format output
        output = ["Available snapshots:\n"]
        for i, snap in enumerate(snapshots, 1):
            perf = snap.get('performance_ms', 0.0)
            perf_str = f"{perf:.4f}ms" if perf > 0 else "not measured"
            output.append(f"{i}. {snap['name']} ({perf_str})")
            output.append(f"   Description: {snap['description']}")
            output.append(f"   Location: .intelliperf/snapshots/{snap['dir']}")
            output.append("")

        return "\n".join(output)

    except Exception as e:
        return f"Error listing snapshots: {e}"


def restore_snapshot_impl(work_dir: Path, name: str) -> str:
    """Restore files from a named snapshot."""
    try:
        snapshots_dir = work_dir / ".intelliperf" / "snapshots"
        if not snapshots_dir.exists():
            return "Error: No snapshots directory found"

        # Find snapshot by name (match prefix)
        matching = [d for d in snapshots_dir.iterdir() if d.is_dir() and d.name.startswith(name)]

        if not matching:
            return f"Error: No snapshot found matching '{name}'"

        if len(matching) > 1:
            return f"Error: Multiple snapshots match '{name}': {[d.name for d in matching]}\nBe more specific."

        snapshot_dir = matching[0]

        # Read metadata
        metadata_file = snapshot_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Restore files
        restored = []
        for src_file in snapshot_dir.iterdir():
            if src_file.name == "metadata.json":
                continue

            dest = work_dir / src_file.name
            shutil.copy2(src_file, dest)
            restored.append(src_file.name)

        return f"Restored snapshot: {snapshot_dir.name}\n  Files restored: {', '.join(restored)}\n  Description: {metadata.get('description', 'N/A')}\n  Performance: {metadata.get('performance_ms', 0.0):.4f}ms"

    except Exception as e:
        return f"Error restoring snapshot: {e}"


def create_snapshot_tools(work_dir: Path) -> List[StructuredTool]:
    """Create snapshot management tools."""
    return [
        StructuredTool.from_function(
            func=lambda name, description, performance_ms, files, **kwargs: save_snapshot_impl(work_dir, name, description, performance_ms, files),
            name="save_snapshot",
            description="Save current code state as a named snapshot. ALL parameters are REQUIRED: name, description, performance_ms (from profile output), and files list (or [] for all files).",
            args_schema=SaveSnapshotInput
        ),
        StructuredTool.from_function(
            func=lambda **kwargs: list_snapshots_impl(work_dir),
            name="list_snapshots",
            description="List all saved snapshots with their performance metrics. Use this to review all optimization attempts.",
            args_schema=ListSnapshotsInput
        ),
        StructuredTool.from_function(
            func=lambda name, **kwargs: restore_snapshot_impl(work_dir, name),
            name="restore_snapshot",
            description="Restore files from a named snapshot. Use this to revert to a previous version or recover the best performing code.",
            args_schema=RestoreSnapshotInput
        ),
    ]
