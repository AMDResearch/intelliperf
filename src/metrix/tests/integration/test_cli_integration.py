"""
Integration tests for CLI - validates end-to-end functionality
"""

import pytest
import subprocess
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"
VECTOR_ADD = EXAMPLES_DIR / "01_vector_add" / "vector_add"


@pytest.mark.timeout(60)
@pytest.mark.skipif(not VECTOR_ADD.exists(), reason="vector_add not compiled")
def test_cli_time_only():
    """Test metrix --time-only -n 1 (single run, single dispatch)"""
    result = subprocess.run(
        ["metrix", "--time-only", "-n", "1", str(VECTOR_ADD)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "vector_add" in result.stdout
    assert "Duration:" in result.stdout
    assert "μs" in result.stdout


@pytest.mark.timeout(60)
@pytest.mark.skipif(not VECTOR_ADD.exists(), reason="vector_add not compiled")
def test_cli_time_only_aggregated():
    """Test metrix profile --time-only --aggregate"""
    result = subprocess.run(
        [
            "metrix",
            "profile",
            "--time-only",
            "--runs",
            "3",
            "--aggregate",
            str(VECTOR_ADD),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "vector_add" in result.stdout
    assert "Duration:" in result.stdout
    assert "μs" in result.stdout
    assert " - " in result.stdout  # Shows range (min - max)


@pytest.mark.timeout(60)
@pytest.mark.skipif(not VECTOR_ADD.exists(), reason="vector_add not compiled")
def test_cli_with_metric():
    """Test metrix --metrics (with runs, aggregates by dispatch)"""
    result = subprocess.run(
        ["metrix", "--metrics", "memory.l2_hit_rate", "-n", "3", str(VECTOR_ADD)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "vector_add" in result.stdout
    assert "L2 Cache Hit Rate" in result.stdout  # Metric name shown
    assert "CACHE PERFORMANCE" in result.stdout  # Section header
    assert "Dispatch #1" in result.stdout  # Shows per-dispatch aggregation


@pytest.mark.timeout(60)
@pytest.mark.skipif(not VECTOR_ADD.exists(), reason="vector_add not compiled")
def test_cli_with_metric_aggregated():
    """Test metrix profile --metrics --aggregate"""
    result = subprocess.run(
        [
            "metrix",
            "profile",
            "--metrics",
            "memory.l2_hit_rate",
            "--aggregate",
            str(VECTOR_ADD),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Command failed: {result.stderr}"
    assert "vector_add" in result.stdout
    assert "L2 Cache Hit Rate" in result.stdout
    assert "CACHE PERFORMANCE" in result.stdout  # Section header in aggregated mode


def test_cli_list_metrics():
    """Test metrix list metrics"""
    result = subprocess.run(
        ["metrix", "list", "metrics"], capture_output=True, text=True, timeout=5
    )

    assert result.returncode == 0
    assert "memory.l2_hit_rate" in result.stdout
