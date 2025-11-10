"""
GFX942 (MI300X) Backend

Each metric is defined with @metric decorator.
Counter names appear EXACTLY ONCE - as function parameters.
"""

from .base import CounterBackend, DeviceSpecs, ProfileResult
from .decorator import metric
from ..profiler.rocprof_wrapper import ROCProfV3Wrapper
from typing import List, Optional


class GFX942Backend(CounterBackend):
    """
    AMD MI300X (gfx942) counter backend

    All metrics are defined with @metric decorator.
    Hardware counter names appear ONLY as function parameter names.
    """

    def _get_device_specs(self) -> DeviceSpecs:
        """MI300X specifications"""
        return DeviceSpecs(
            arch="gfx942",
            name="AMD Instinct MI300X",
            num_cu=304,
            max_waves_per_cu=32,
            wavefront_size=64,
            base_clock_mhz=2100.0,
            hbm_bandwidth_gbs=5300.0,
            l2_bandwidth_gbs=11000.0,
            l2_size_mb=256.0,
            lds_size_per_cu_kb=64.0,
            fp32_tflops=163.4,
            fp64_tflops=81.7,
            int8_tops=1307.4,
            boost_clock_mhz=2100
        )

    def _get_counter_groups(self) -> List[List[str]]:
        """
        Return hardware counter compatibility groups for gfx942 (MI300X).

        These groups are based on guided-tuning's tested configurations and ensure
        counters in each group can be collected simultaneously without hardware conflicts.

        Reference: external/guided-tuning/configs/MI3/input.json

        Note: MI300/MI350 use TCC_EA0_* counters (not TCC_EA_* which are MI200-specific)

        CRITICAL: TCC_EA0_RDREQ_sum and TCC_EA0_WRREQ_sum CANNOT be collected together
        with TCC_EA0_ATOMIC_* counters due to hardware resource conflicts!
        """
        return [
            # Group 1: Atomics and write requests (from pmc_perf_0)
            # Note: RDREQ cannot be in this group!
            [
                "SQ_LDS_BANK_CONFLICT",
                "TCC_EA0_WRREQ_sum",
                "TCC_EA0_ATOMIC_LEVEL_sum",
                "TCC_EA0_ATOMIC_sum",
                "GRBM_GUI_ACTIVE",
            ],
            # Group 2: LDS and L2 cache counters (from SQ_INSTS_LDS)
            [
                "SQ_INSTS_LDS",
                "TCP_TOTAL_ACCESSES_sum",
                "TCC_HIT_sum",
                "TCC_MISS_sum",
            ],
            # Group 3: Memory instructions and read requests (from SQ_INSTS_VMEM)
            # Note: RDREQ can only be collected here, not with atomics!
            [
                "TCP_TCC_READ_REQ_sum",
                "TCP_TCC_WRITE_REQ_sum",
                "TCP_TOTAL_CACHE_ACCESSES_sum",
                "TCC_EA0_RDREQ_sum",
                "SQ_INSTS_VMEM_RD",
                "SQ_INSTS_VMEM_WR",
            ],
        ]

    def _run_rocprof(self, command: str, counters: List[str],
                     kernel_filter: Optional[str] = None, cwd: Optional[str] = None) -> List[ProfileResult]:
        """Run rocprofv3 and return results (single pass only - base class handles multi-pass)"""
        wrapper = ROCProfV3Wrapper(timeout=None)  # No timeout - profiling can take as long as it needs
        return wrapper.profile(command, counters, kernel_filter=kernel_filter, cwd=cwd)

    # Memory bandwidth metrics

    @metric("memory.hbm_read_bandwidth")
    def _hbm_read_bandwidth(self, TCC_EA0_RDREQ_sum, GRBM_GUI_ACTIVE):
        """
        HBM read bandwidth in GB/s

        Formula: (read_requests * 64 bytes) / (active_cycles / clock_freq)

        Note: TCC_EA0_RDREQ_sum aggregates across all memory controllers on MI300
        """
        bytes_read = TCC_EA0_RDREQ_sum * 64  # Each request is 64 bytes

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        return (bytes_read / 1e9) / time_seconds if time_seconds > 0 else 0.0

    @metric("memory.hbm_write_bandwidth")
    def _hbm_write_bandwidth(self, TCC_EA0_WRREQ_sum, GRBM_GUI_ACTIVE):
        """
        HBM write bandwidth in GB/s

        Formula: (write_requests * 64 bytes) / (active_cycles / clock_freq)

        Note: TCC_EA0_WRREQ_sum aggregates across all memory controllers on MI300
        """
        bytes_written = TCC_EA0_WRREQ_sum * 64  # Each request is 64 bytes

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        return (bytes_written / 1e9) / time_seconds if time_seconds > 0 else 0.0

    @metric("memory.hbm_bandwidth_utilization")
    def _hbm_bandwidth_utilization(self, TCC_EA0_RDREQ_sum, TCC_EA0_WRREQ_sum, GRBM_GUI_ACTIVE):
        """
        HBM bandwidth utilization as percentage of peak

        Formula: (actual_bandwidth / peak_bandwidth) * 100

        Note: TCC_EA0_* counters aggregate across all memory controllers on MI300
        """
        total_bytes = (TCC_EA0_RDREQ_sum + TCC_EA0_WRREQ_sum) * 64

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        actual_bw_gbs = (total_bytes / 1e9) / time_seconds if time_seconds > 0 else 0.0

        return (actual_bw_gbs / self.device_specs.hbm_bandwidth_gbs) * 100

    @metric("memory.bytes_transferred_hbm")
    def _bytes_transferred_hbm(self, TCC_EA0_RDREQ_sum, TCC_EA0_WRREQ_sum):
        """
        Total bytes transferred through HBM

        Formula: (read_requests + write_requests) * 64 bytes

        Note: TCC_EA0_* counters aggregate across all memory controllers on MI300
        """
        return (TCC_EA0_RDREQ_sum + TCC_EA0_WRREQ_sum) * 64

    # Cache metrics

    @metric("memory.l2_hit_rate")
    def _l2_hit_rate(self, TCC_HIT_sum, TCC_MISS_sum):
        """
        L2 cache hit rate as percentage

        Formula: (hits / (hits + misses)) * 100
        """
        total = TCC_HIT_sum + TCC_MISS_sum
        return (TCC_HIT_sum / total) * 100 if total > 0 else 0.0

    @metric("memory.l1_hit_rate")
    def _l1_hit_rate(self, TCP_TCC_READ_REQ_sum, TCP_TOTAL_CACHE_ACCESSES_sum):
        """
        L1 cache hit rate as percentage

        Formula: ((total_accesses - l1_misses) / total_accesses) * 100
        L1 misses go to L2 (TCC), so misses = TCP_TCC_READ_REQ
        """
        if TCP_TOTAL_CACHE_ACCESSES_sum == 0:
            return 0.0

        l1_hits = TCP_TOTAL_CACHE_ACCESSES_sum - TCP_TCC_READ_REQ_sum
        return (l1_hits / TCP_TOTAL_CACHE_ACCESSES_sum) * 100

    @metric("memory.l2_bandwidth")
    def _l2_bandwidth(self, TCC_HIT_sum, TCC_MISS_sum, GRBM_GUI_ACTIVE):
        """
        L2 cache bandwidth in GB/s

        Formula: (total_accesses * 128 bytes) / time
        Note: L2 cacheline is 128 bytes
        """
        total_accesses = TCC_HIT_sum + TCC_MISS_sum
        bytes_accessed = total_accesses * 128  # L2 cacheline size

        if GRBM_GUI_ACTIVE == 0:
            return 0.0

        time_seconds = GRBM_GUI_ACTIVE / (self.device_specs.base_clock_mhz * 1e6)
        return (bytes_accessed / 1e9) / time_seconds if time_seconds > 0 else 0.0

    # Coalescing metrics

    @metric("memory.coalescing_efficiency")
    def _coalescing_efficiency(self, SQ_INSTS_VMEM_RD, SQ_INSTS_VMEM_WR, TCP_TOTAL_ACCESSES_sum):
        """
        Memory coalescing efficiency as percentage

        Formula: (total_memory_instructions * 16 / total_cache_accesses) * 100

        Physical meaning:
        - Perfect coalescing (stride=1): 100% (minimal cache accesses)
        - Poor coalescing (stride>1): 25% for float, 50% for double

        This represents actual bandwidth efficiency, not rescaled.
        """
        total_instructions = SQ_INSTS_VMEM_RD + SQ_INSTS_VMEM_WR

        if TCP_TOTAL_ACCESSES_sum == 0:
            return 0.0

        # 16 = 64 threads per wavefront / 4 threads per cacheline
        efficiency = (total_instructions * 16 / TCP_TOTAL_ACCESSES_sum) * 100

        # Cap at 100% (can happen due to prefetching)
        return min(efficiency, 100.0)

    @metric("memory.global_load_efficiency")
    def _global_load_efficiency(self, SQ_INSTS_VMEM_RD, TCP_TCC_READ_REQ_sum):
        """
        Global load efficiency - ratio of requested vs fetched memory

        Formula: (read_instructions * 64 bytes / read_requests * 64 bytes) * 100
        Simplifies to: (read_instructions / read_requests) * 100
        """
        if TCP_TCC_READ_REQ_sum == 0:
            return 0.0

        return min((SQ_INSTS_VMEM_RD / TCP_TCC_READ_REQ_sum) * 100, 100.0)

    @metric("memory.global_store_efficiency")
    def _global_store_efficiency(self, SQ_INSTS_VMEM_WR, TCP_TCC_WRITE_REQ_sum):
        """
        Global store efficiency - ratio of requested vs written memory

        Formula: (write_instructions / write_requests) * 100
        """
        if TCP_TCC_WRITE_REQ_sum == 0:
            return 0.0

        return min((SQ_INSTS_VMEM_WR / TCP_TCC_WRITE_REQ_sum) * 100, 100.0)

    # LDS metrics

    @metric("memory.lds_bank_conflicts")
    def _lds_bank_conflicts(self, SQ_LDS_BANK_CONFLICT, SQ_INSTS_LDS):
        """
        LDS bank conflicts per instruction

        Formula: total_conflicts / total_lds_instructions
        """
        if SQ_INSTS_LDS == 0:
            return 0.0

        return SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS

    # Atomic metrics

    @metric("memory.atomic_latency")
    def _atomic_latency(self, TCC_EA0_ATOMIC_LEVEL_sum, TCC_EA0_ATOMIC_sum):
        """
        Average atomic operation latency in cycles (L2 cache atomic latency)

        Formula: TCC_EA0_ATOMIC_LEVEL_sum / TCC_EA0_ATOMIC_sum (MI300/MI350 counters)

        Note: This measures atomic operations to/from L2 cache, not GDS operations.
        GDS (Global Data Share) is a special feature rarely used by most kernels.
        """
        if TCC_EA0_ATOMIC_sum == 0:
            return 0.0

        return TCC_EA0_ATOMIC_LEVEL_sum / TCC_EA0_ATOMIC_sum

