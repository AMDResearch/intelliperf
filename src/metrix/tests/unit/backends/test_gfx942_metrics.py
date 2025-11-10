"""
Unit tests for GFX942Backend metric computations

Tests use MOCK counter data (no hardware counters in test code!)
"""

import pytest
from metrix.backends.gfx942 import GFX942Backend


class TestL2HitRate:
    """Test L2 cache hit rate computation"""

    def test_perfect_hit_rate(self):
        """100% hit rate"""
        backend = GFX942Backend()
        backend._raw_data = {
            'TCC_HIT_sum': 1000,
            'TCC_MISS_sum': 0
        }

        result = backend._l2_hit_rate()
        assert result == 100.0

    def test_zero_hit_rate(self):
        """0% hit rate (all misses)"""
        backend = GFX942Backend()
        backend._raw_data = {
            'TCC_HIT_sum': 0,
            'TCC_MISS_sum': 1000
        }

        result = backend._l2_hit_rate()
        assert result == 0.0

    def test_fifty_percent_hit_rate(self):
        """50% hit rate"""
        backend = GFX942Backend()
        backend._raw_data = {
            'TCC_HIT_sum': 500,
            'TCC_MISS_sum': 500
        }

        result = backend._l2_hit_rate()
        assert result == 50.0

    def test_no_accesses(self):
        """Handle zero total accesses"""
        backend = GFX942Backend()
        backend._raw_data = {
            'TCC_HIT_sum': 0,
            'TCC_MISS_sum': 0
        }

        result = backend._l2_hit_rate()
        assert result == 0.0


class TestCoalescingEfficiency:
    """Test memory coalescing efficiency computation"""

    def test_perfect_coalescing(self):
        """100% coalescing (stride-1 access)"""
        backend = GFX942Backend()
        backend._raw_data = {
            'SQ_INSTS_VMEM_RD': 100,
            'SQ_INSTS_VMEM_WR': 0,
            'TCP_TOTAL_ACCESSES_sum': 1600  # 100 * 16
        }

        result = backend._coalescing_efficiency()
        assert result == 100.0

    def test_poor_coalescing(self):
        """25% coalescing (completely uncoalesced float access)"""
        backend = GFX942Backend()
        backend._raw_data = {
            'SQ_INSTS_VMEM_RD': 100,
            'SQ_INSTS_VMEM_WR': 0,
            'TCP_TOTAL_ACCESSES_sum': 6400  # 4x more accesses
        }

        result = backend._coalescing_efficiency()
        assert result == 25.0

    def test_mixed_read_write(self):
        """Coalescing with both reads and writes"""
        backend = GFX942Backend()
        backend._raw_data = {
            'SQ_INSTS_VMEM_RD': 50,
            'SQ_INSTS_VMEM_WR': 50,
            'TCP_TOTAL_ACCESSES_sum': 1600  # (50 + 50) * 16
        }

        result = backend._coalescing_efficiency()
        assert result == 100.0

    def test_no_memory_instructions(self):
        """Handle zero memory instructions"""
        backend = GFX942Backend()
        backend._raw_data = {
            'SQ_INSTS_VMEM_RD': 0,
            'SQ_INSTS_VMEM_WR': 0,
            'TCP_TOTAL_ACCESSES_sum': 1000
        }

        result = backend._coalescing_efficiency()
        assert result == 0.0


class TestLDSBankConflicts:
    """Test LDS bank conflict computation"""

    def test_no_conflicts(self):
        """Perfect LDS access pattern"""
        backend = GFX942Backend()
        backend._raw_data = {
            'SQ_LDS_BANK_CONFLICT': 0,
            'SQ_INSTS_LDS': 1000
        }

        result = backend._lds_bank_conflicts()
        assert result == 0.0

    def test_high_conflicts(self):
        """2 conflicts per instruction"""
        backend = GFX942Backend()
        backend._raw_data = {
            'SQ_LDS_BANK_CONFLICT': 2000,
            'SQ_INSTS_LDS': 1000
        }

        result = backend._lds_bank_conflicts()
        assert result == 2.0

    def test_no_lds_instructions(self):
        """Handle zero LDS instructions"""
        backend = GFX942Backend()
        backend._raw_data = {
            'SQ_LDS_BANK_CONFLICT': 100,
            'SQ_INSTS_LDS': 0
        }

        result = backend._lds_bank_conflicts()
        assert result == 0.0


class TestBandwidthMetrics:
    """Test HBM bandwidth computations"""

    def test_hbm_read_bandwidth(self):
        """Test read bandwidth calculation"""
        backend = GFX942Backend()
        backend._raw_data = {
            'TCC_EA0_RDREQ_sum': 1000,
            'TCC_EA1_RDREQ_sum': 1000,
            'GRBM_GUI_ACTIVE': 2100000  # 1 ms at 2.1 GHz
        }

        result = backend._hbm_read_bandwidth()
        # (2000 requests * 64 bytes) / 0.001 seconds = 128 MB/s = 0.128 GB/s
        assert 0.1 < result < 0.2

    def test_hbm_write_bandwidth(self):
        """Test write bandwidth calculation"""
        backend = GFX942Backend()
        backend._raw_data = {
            'TCC_EA0_WRREQ_sum': 500,
            'TCC_EA1_WRREQ_sum': 500,
            'GRBM_GUI_ACTIVE': 2100000  # 1 ms at 2.1 GHz
        }

        result = backend._hbm_write_bandwidth()
        # (1000 requests * 64 bytes) / 0.001 seconds = 64 MB/s = 0.064 GB/s
        assert 0.05 < result < 0.1

    def test_zero_active_cycles(self):
        """Handle zero active cycles"""
        backend = GFX942Backend()
        backend._raw_data = {
            'TCC_EA0_RDREQ_sum': 1000,
            'TCC_EA1_RDREQ_sum': 1000,
            'GRBM_GUI_ACTIVE': 0
        }

        result = backend._hbm_read_bandwidth()
        assert result == 0.0


class TestAtomicLatency:
    """Test atomic operation latency computation"""

    def test_low_latency(self):
        """10 cycles per atomic operation"""
        backend = GFX942Backend()
        backend._raw_data = {
            'SQ_INSTS_GDS': 1000,
            'GDS_BUSY': 10000
        }

        result = backend._atomic_latency()
        assert result == 10.0

    def test_high_latency(self):
        """1000 cycles per atomic (contention)"""
        backend = GFX942Backend()
        backend._raw_data = {
            'SQ_INSTS_GDS': 100,
            'GDS_BUSY': 100000
        }

        result = backend._atomic_latency()
        assert result == 1000.0

    def test_no_atomics(self):
        """Handle zero atomic instructions"""
        backend = GFX942Backend()
        backend._raw_data = {
            'SQ_INSTS_GDS': 0,
            'GDS_BUSY': 5000
        }

        result = backend._atomic_latency()
        assert result == 0.0


class TestMetricDiscovery:
    """Test backend auto-discovers metrics"""

    def test_discovers_all_metrics(self):
        """Backend should auto-discover all @metric decorated methods"""
        backend = GFX942Backend()

        metrics = backend.get_available_metrics()

        # Should have all the metrics we defined
        assert "memory.l2_hit_rate" in metrics
        assert "memory.coalescing_efficiency" in metrics
        assert "memory.lds_bank_conflicts" in metrics
        assert "memory.hbm_read_bandwidth" in metrics
        assert "memory.atomic_latency" in metrics

    def test_get_required_counters(self):
        """Backend should correctly report required counters for a metric"""
        backend = GFX942Backend()

        counters = backend.get_required_counters(["memory.l2_hit_rate"])

        # Should require TCC_HIT_sum and TCC_MISS_sum (counter names appear in function signature)
        assert "TCC_HIT_sum" in counters
        assert "TCC_MISS_sum" in counters
        assert len(counters) == 2

