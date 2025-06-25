from intelliperf.core import gpu_spec


def test_warp_size_exact():
	# Directly call get_warp_size without touching real hardware
	warp = gpu_spec.get_warp_size(device_id=0)
	assert warp == 64, "Expected warp size of exactly 64 threads"


def test_l2_cache_positive():
	# get_l2_cache_size returns KB/1024 => 2048/1024 = 2.0
	l2 = gpu_spec.get_l2_cache_size(device_id=0)
	assert l2 > 0, "Expected L2 cache size to be > 0 KB"
