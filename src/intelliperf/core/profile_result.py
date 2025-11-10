# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""Clean abstraction layer for profiling results."""

from dataclasses import dataclass, asdict
from typing import Callable, Optional


@dataclass
class KernelMetrics:
	"""Type-safe container for kernel performance metrics."""

	# Required metrics (always present)
	kernel_name: str
	duration_ns: float
	call_count: int = 1

	# Memory bandwidth
	hbm_read_bw_gb_s: Optional[float] = None
	hbm_write_bw_gb_s: Optional[float] = None
	hbm_utilization_pct: Optional[float] = None

	# Cache metrics
	l1_hit_rate: Optional[float] = None
	l2_hit_rate: Optional[float] = None
	l2_bandwidth_gb_s: Optional[float] = None

	# Memory access patterns
	coalescing_efficiency_pct: Optional[float] = None

	# LDS metrics
	lds_bank_conflicts: Optional[float] = None

	# Atomic metrics
	atomic_latency_cycles: Optional[float] = None

	# Source code availability (set by nexus integration)
	has_source_code: bool = False
	source_file: Optional[str] = None  # Primary source file
	source_files: list[str] = None  # All source files

	def __post_init__(self):
		"""Initialize mutable defaults."""
		if self.source_files is None:
			self.source_files = []

	@classmethod
	def from_metrix_kernel(cls, kernel):
		"""Create KernelMetrics from a metrix kernel object.

		Args:
			kernel: Metrix kernel object with .name, .duration_us, and .metrics attributes

		Returns:
			KernelMetrics instance populated from metrix data
		"""
		metrics_dict = kernel.metrics

		# Normalize kernel name - strip .kd suffix if present
		kernel_name = kernel.name
		if kernel_name.endswith(".kd"):
			kernel_name = kernel_name[:-3]

		return cls(
			kernel_name=kernel_name,
			duration_ns=kernel.duration_us.avg * 1000,  # Convert μs to ns
			call_count=1,  # Aggregated by kernel name
			# Memory bandwidth
			hbm_read_bw_gb_s=metrics_dict.get("memory.hbm_read_bandwidth").avg
			if "memory.hbm_read_bandwidth" in metrics_dict
			else None,
			hbm_write_bw_gb_s=metrics_dict.get("memory.hbm_write_bandwidth").avg
			if "memory.hbm_write_bandwidth" in metrics_dict
			else None,
			hbm_utilization_pct=metrics_dict.get("memory.hbm_bandwidth_utilization").avg
			if "memory.hbm_bandwidth_utilization" in metrics_dict
			else None,
			# Cache metrics
			l1_hit_rate=metrics_dict.get("memory.l1_hit_rate").avg if "memory.l1_hit_rate" in metrics_dict else None,
			l2_hit_rate=metrics_dict.get("memory.l2_hit_rate").avg if "memory.l2_hit_rate" in metrics_dict else None,
			l2_bandwidth_gb_s=metrics_dict.get("memory.l2_bandwidth").avg
			if "memory.l2_bandwidth" in metrics_dict
			else None,
			# Memory access patterns
			coalescing_efficiency_pct=metrics_dict.get("memory.coalescing_efficiency").avg
			if "memory.coalescing_efficiency" in metrics_dict
			else None,
			# LDS metrics
			lds_bank_conflicts=metrics_dict.get("memory.lds_bank_conflicts").avg
			if "memory.lds_bank_conflicts" in metrics_dict
			else None,
			# Atomic metrics
			atomic_latency_cycles=metrics_dict.get("memory.atomic_latency").avg
			if "memory.atomic_latency" in metrics_dict
			else None,
		)

	def __repr__(self) -> str:
		"""Pretty representation of metrics."""
		lines = [f"KernelMetrics({self.kernel_name})"]
		lines.append(f"  Duration: {self.duration_ns:.2f} ns")

		if self.hbm_read_bw_gb_s is not None:
			lines.append(f"  HBM Read BW: {self.hbm_read_bw_gb_s:.2f} GB/s")
		if self.hbm_write_bw_gb_s is not None:
			lines.append(f"  HBM Write BW: {self.hbm_write_bw_gb_s:.2f} GB/s")
		if self.hbm_utilization_pct is not None:
			lines.append(f"  HBM Util: {self.hbm_utilization_pct:.1f}%")

		if self.l1_hit_rate is not None:
			lines.append(f"  L1 Hit Rate: {self.l1_hit_rate:.1f}%")
		if self.l2_hit_rate is not None:
			lines.append(f"  L2 Hit Rate: {self.l2_hit_rate:.1f}%")

		if self.coalescing_efficiency_pct is not None:
			lines.append(f"  Coalescing: {self.coalescing_efficiency_pct:.1f}%")

		if self.lds_bank_conflicts is not None:
			lines.append(f"  LDS Bank Conflicts: {self.lds_bank_conflicts:.2f}")

		if self.atomic_latency_cycles is not None:
			lines.append(f"  Atomic Latency: {self.atomic_latency_cycles:.2f} cycles")

		if self.has_source_code:
			lines.append(f"  Source: {self.source_file}")

		return "\n".join(lines)


@dataclass
class ProfileResult:
	"""Container for profiling results with clean query interface."""

	kernels: list[KernelMetrics]

	def get_kernel(self, kernel_name: str) -> Optional[KernelMetrics]:
		"""Get metrics for a specific kernel by name.

		Args:
			kernel_name: Name of the kernel to find

		Returns:
			KernelMetrics if found, None otherwise
		"""
		for kernel in self.kernels:
			if kernel.kernel_name == kernel_name:
				return kernel
		return None

	def get_top_kernels(self, n: int = 1) -> list[KernelMetrics]:
		"""Get top N kernels by duration.

		Args:
			n: Number of top kernels to return

		Returns:
			List of up to N kernels, sorted by duration (descending)
		"""
		sorted_kernels = sorted(self.kernels, key=lambda k: k.duration_ns, reverse=True)
		return sorted_kernels[:n]

	def find_optimization_candidate(
		self,
		metric_filter: Callable[[KernelMetrics], bool],
		require_source_code: bool = True,
	) -> Optional[KernelMetrics]:
		"""Find the first kernel matching optimization criteria.

		This is the common pattern for formulas:
		1. Sort kernels by duration (most impactful first)
		2. Filter by formula-specific metric thresholds
		3. Ensure source code is available (if required)
		4. Return the first match

		Args:
			metric_filter: Function that returns True if kernel meets optimization criteria.
						   Example: lambda k: k.coalescing_efficiency_pct < 80.0
			require_source_code: Only return kernels with available source code

		Returns:
			First kernel matching all criteria, or None if no match found

		Example:
			>>> # Find kernel with poor coalescing
			>>> result.find_optimization_candidate(
			...     metric_filter=lambda k: k.coalescing_efficiency_pct and k.coalescing_efficiency_pct < 80.0
			... )
		"""
		# Sort by duration (most impactful first)
		sorted_kernels = sorted(self.kernels, key=lambda k: k.duration_ns, reverse=True)

		for kernel in sorted_kernels:
			# Check source code requirement
			if require_source_code and not kernel.has_source_code:
				continue

			# Check formula-specific metric filter
			if metric_filter(kernel):
				return kernel

		return None

	def to_dict(self) -> list[dict]:
		"""Convert ProfileResult to a JSON-serializable list of dicts.

		Returns:
			List of kernel metric dictionaries
		"""
		return [asdict(k) for k in self.kernels]

	def filter_kernels(
		self,
		metric_filter: Callable[[KernelMetrics], bool],
		require_source_code: bool = False,
	) -> list[KernelMetrics]:
		"""Filter kernels by metric criteria.

		Args:
			metric_filter: Function that returns True if kernel should be included
			require_source_code: Only include kernels with available source code

		Returns:
			List of kernels matching criteria, sorted by duration
		"""
		filtered = []
		for kernel in self.kernels:
			if require_source_code and not kernel.has_source_code:
				continue
			if metric_filter(kernel):
				filtered.append(kernel)

		return sorted(filtered, key=lambda k: k.duration_ns, reverse=True)

	def __len__(self) -> int:
		"""Number of kernels in result."""
		return len(self.kernels)

	def __repr__(self) -> str:
		"""Pretty representation."""
		return f"ProfileResult({len(self.kernels)} kernels)"

	def __str__(self) -> str:
		"""String representation with detailed kernel metrics."""
		if not self.kernels:
			return "ProfileResult(0 kernels)"

		lines = [f"ProfileResult({len(self.kernels)} kernels):"]
		for i, kernel in enumerate(self.kernels):
			lines.append(f"  Kernel {i+1}: {kernel.kernel_name}")
			lines.append(f"    Duration: {kernel.duration_ns / 1e6:.2f} ms")
			if kernel.coalescing_efficiency_pct is not None:
				lines.append(f"    Coalescing: {kernel.coalescing_efficiency_pct:.1f}%")
			if kernel.lds_bank_conflicts is not None:
				lines.append(f"    Bank Conflicts: {kernel.lds_bank_conflicts:.2f}")
			if kernel.atomic_latency_cycles is not None:
				lines.append(f"    Atomic Latency: {kernel.atomic_latency_cycles:.2f} cycles")
			if kernel.l2_hit_rate is not None:
				lines.append(f"    L2 Hit Rate: {kernel.l2_hit_rate:.1f}%")
			if kernel.has_source_code:
				lines.append(f"    Source: {kernel.source_file}")
		return "\n".join(lines)

