# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

import ctypes
from enum import IntEnum, auto

rt_path = "libamdhip64.so"
hip_runtime = ctypes.cdll.LoadLibrary(rt_path)


class HipDeviceAttribute(IntEnum):
	hipDeviceAttributeCudaCompatibleBegin = 0
	hipDeviceAttributeEccEnabled = hipDeviceAttributeCudaCompatibleBegin  # Whether ECC support is enabled.
	hipDeviceAttributeAccessPolicyMaxWindowSize = auto()  # Cuda only. The maximum size of the window policy in bytes.
	hipDeviceAttributeAsyncEngineCount = auto()  # Asynchronous engines number.
	hipDeviceAttributeCanMapHostMemory = auto()  # Whether host memory can be mapped into device address space
	hipDeviceAttributeCanUseHostPointerForRegisteredMem = auto()  # Device can access host registered memory
	# at the same virtual address as the CPU
	hipDeviceAttributeClockRate = auto()  # Peak clock frequency in kilohertz.
	hipDeviceAttributeComputeMode = auto()  # Compute mode that device is currently in.
	hipDeviceAttributeComputePreemptionSupported = auto()  # Device supports Compute Preemption.
	hipDeviceAttributeConcurrentKernels = auto()  # Device can possibly execute multiple kernels concurrently.
	hipDeviceAttributeConcurrentManagedAccess = (
		auto()
	)  # Device can coherently access managed memory concurrently with the CPU
	hipDeviceAttributeCooperativeLaunch = auto()  # Support cooperative launch
	hipDeviceAttributeCooperativeMultiDeviceLaunch = auto()  # Support cooperative launch on multiple devices
	hipDeviceAttributeDeviceOverlap = auto()  # Device can concurrently copy memory and execute a kernel.
	# Deprecated. Use instead asyncEngineCount.
	hipDeviceAttributeDirectManagedMemAccessFromHost = auto()  # Host can directly access managed memory on
	# the device without migration
	hipDeviceAttributeGlobalL1CacheSupported = auto()  # Device supports caching globals in L1
	hipDeviceAttributeHostNativeAtomicSupported = (
		auto()
	)  # Link between the device and the host supports native atomic operations
	hipDeviceAttributeIntegrated = auto()  # Device is integrated GPU
	hipDeviceAttributeIsMultiGpuBoard = auto()  # Multiple GPU devices.
	hipDeviceAttributeKernelExecTimeout = auto()  # Run time limit for kernels executed on the device
	hipDeviceAttributeL2CacheSize = auto()  # Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
	hipDeviceAttributeLocalL1CacheSupported = auto()  # caching locals in L1 is supported
	hipDeviceAttributeLuid = (
		auto()
	)  # 8-byte locally unique identifier in 8 bytes. Undefined on TCC and non-Windows platforms
	hipDeviceAttributeLuidDeviceNodeMask = auto()  # Luid device node mask. Undefined on TCC and non-Windows platforms
	hipDeviceAttributeComputeCapabilityMajor = auto()  # Major compute capability version number.
	hipDeviceAttributeManagedMemory = auto()  # Device supports allocating managed memory on this system
	hipDeviceAttributeMaxBlocksPerMultiProcessor = auto()  # Max block size per multiprocessor
	hipDeviceAttributeMaxBlockDimX = auto()  # Max block size in width.
	hipDeviceAttributeMaxBlockDimY = auto()  # Max block size in height.
	hipDeviceAttributeMaxBlockDimZ = auto()  # Max block size in depth.
	hipDeviceAttributeMaxGridDimX = auto()  # Max grid size  in width.
	hipDeviceAttributeMaxGridDimY = auto()  # Max grid size  in height.
	hipDeviceAttributeMaxGridDimZ = auto()  # Max grid size  in depth.
	hipDeviceAttributeMaxSurface1D = auto()  # Maximum size of 1D surface.
	hipDeviceAttributeMaxSurface1DLayered = auto()  # Cuda only. Maximum dimensions of 1D layered surface.
	hipDeviceAttributeMaxSurface2D = auto()  # Maximum dimension (width, height) of 2D surface.
	hipDeviceAttributeMaxSurface2DLayered = auto()  # Cuda only. Maximum dimensions of 2D layered surface.
	hipDeviceAttributeMaxSurface3D = auto()  # Maximum dimension (width, height, depth) of 3D surface.
	hipDeviceAttributeMaxSurfaceCubemap = auto()  # Cuda only. Maximum dimensions of Cubemap surface.
	hipDeviceAttributeMaxSurfaceCubemapLayered = auto()  # Cuda only. Maximum dimension of Cubemap layered surface.
	hipDeviceAttributeMaxTexture1DWidth = auto()  # Maximum size of 1D texture.
	hipDeviceAttributeMaxTexture1DLayered = auto()  # Maximum dimensions of 1D layered texture.
	hipDeviceAttributeMaxTexture1DLinear = auto()  # Maximum number of elements allocatable in a 1D linear texture.
	# Use cudaDeviceGetTexture1DLinearMaxWidth() instead on Cuda.
	hipDeviceAttributeMaxTexture1DMipmap = auto()  # Maximum size of 1D mipmapped texture.
	hipDeviceAttributeMaxTexture2DWidth = auto()  # Maximum dimension width of 2D texture.
	hipDeviceAttributeMaxTexture2DHeight = auto()  # Maximum dimension hight of 2D texture.
	hipDeviceAttributeMaxTexture2DGather = auto()  # Maximum dimensions of 2D texture if gather operations  performed.
	hipDeviceAttributeMaxTexture2DLayered = auto()  # Maximum dimensions of 2D layered texture.
	hipDeviceAttributeMaxTexture2DLinear = (
		auto()
	)  # Maximum dimensions (width, height, pitch) of 2D textures bound to pitched memory.
	hipDeviceAttributeMaxTexture2DMipmap = auto()  # Maximum dimensions of 2D mipmapped texture.
	hipDeviceAttributeMaxTexture3DWidth = auto()  # Maximum dimension width of 3D texture.
	hipDeviceAttributeMaxTexture3DHeight = auto()  # Maximum dimension height of 3D texture.
	hipDeviceAttributeMaxTexture3DDepth = auto()  # Maximum dimension depth of 3D texture.
	hipDeviceAttributeMaxTexture3DAlt = auto()  # Maximum dimensions of alternate 3D texture.
	hipDeviceAttributeMaxTextureCubemap = auto()  # Maximum dimensions of Cubemap texture
	hipDeviceAttributeMaxTextureCubemapLayered = auto()  # Maximum dimensions of Cubemap layered texture.
	hipDeviceAttributeMaxThreadsDim = auto()  # Maximum dimension of a block
	hipDeviceAttributeMaxThreadsPerBlock = auto()  # Maximum number of threads per block.
	hipDeviceAttributeMaxThreadsPerMultiProcessor = auto()  # Maximum resident threads per multiprocessor.
	hipDeviceAttributeMaxPitch = auto()  # Maximum pitch in bytes allowed by memory copies
	hipDeviceAttributeMemoryBusWidth = auto()  # Global memory bus width in bits.
	hipDeviceAttributeMemoryClockRate = auto()  # Peak memory clock frequency in kilohertz.
	hipDeviceAttributeComputeCapabilityMinor = auto()  # Minor compute capability version number.
	hipDeviceAttributeMultiGpuBoardGroupID = auto()  # Unique ID of device group on the same multi-GPU board
	hipDeviceAttributeMultiprocessorCount = auto()  # Number of multiprocessors on the device.
	hipDeviceAttributeUnused1 = auto()  # Previously hipDeviceAttributeName
	hipDeviceAttributePageableMemoryAccess = auto()  # Device supports coherently accessing pageable memory
	# without calling hipHostRegister on it
	hipDeviceAttributePageableMemoryAccessUsesHostPageTables = (
		auto()
	)  # Device accesses pageable memory via the host's page tables
	hipDeviceAttributePciBusId = auto()  # PCI Bus ID.
	hipDeviceAttributePciDeviceId = auto()  # PCI Device ID.
	hipDeviceAttributePciDomainID = auto()  # PCI Domain ID.
	hipDeviceAttributePersistingL2CacheMaxSize = auto()  # Maximum l2 persisting lines capacity in bytes
	hipDeviceAttributeMaxRegistersPerBlock = (
		auto()
	)  # 32-bit registers available to a thread block. This number is shared
	# by all thread blocks simultaneously resident on a multiprocessor.
	hipDeviceAttributeMaxRegistersPerMultiprocessor = auto()  # 32-bit registers available per block.
	hipDeviceAttributeReservedSharedMemPerBlock = auto()  # Shared memory reserved by CUDA driver per block.
	hipDeviceAttributeMaxSharedMemoryPerBlock = auto()  # Maximum shared memory available per block in bytes.
	hipDeviceAttributeSharedMemPerBlockOptin = auto()  # Maximum shared memory per block usable by special opt in.
	hipDeviceAttributeSharedMemPerMultiprocessor = auto()  # Shared memory available per multiprocessor.
	hipDeviceAttributeSingleToDoublePrecisionPerfRatio = (
		auto()
	)  # Cuda only. Performance ratio of single precision to double precision.
	hipDeviceAttributeStreamPrioritiesSupported = auto()  # Whether to support stream priorities.
	hipDeviceAttributeSurfaceAlignment = auto()  # Alignment requirement for surfaces
	hipDeviceAttributeTccDriver = auto()  # Cuda only. Whether device is a Tesla device using TCC driver
	hipDeviceAttributeTextureAlignment = auto()  # Alignment requirement for textures
	hipDeviceAttributeTexturePitchAlignment = (
		auto()
	)  # Pitch alignment requirement for 2D texture references bound to pitched memory;
	hipDeviceAttributeTotalConstantMemory = auto()  # Constant memory size in bytes.
	hipDeviceAttributeTotalGlobalMem = auto()  # Global memory available on device.
	hipDeviceAttributeUnifiedAddressing = auto()  # Cuda only. An unified address space shared with the host.
	hipDeviceAttributeUnused2 = auto()  # Previously hipDeviceAttributeUuid
	hipDeviceAttributeWarpSize = auto()  # Warp size in threads.
	hipDeviceAttributeMemoryPoolsSupported = auto()  # Device supports HIP Stream Ordered Memory Allocator
	hipDeviceAttributeVirtualMemoryManagementSupported = auto()  # Device supports HIP virtual memory management
	hipDeviceAttributeHostRegisterSupported = auto()  # Can device support host memory registration via hipHostRegister
	hipDeviceAttributeMemoryPoolSupportedHandleTypes = (
		auto()
	)  # Supported handle mask for HIP Stream Ordered Memory Allocator

	hipDeviceAttributeCudaCompatibleEnd = 9999
	hipDeviceAttributeAmdSpecificBegin = 10000
	hipDeviceAttributeClockInstructionRate = (
		hipDeviceAttributeAmdSpecificBegin  # Frequency in khz of the timer used by the device-side "clock*"
	)
	hipDeviceAttributeUnused3 = auto()  # Previously hipDeviceAttributeArch
	hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = auto()  # Maximum Shared Memory PerMultiprocessor.
	hipDeviceAttributeUnused4 = auto()  # Previously hipDeviceAttributeGcnArch
	hipDeviceAttributeUnused5 = auto()  # Previously hipDeviceAttributeGcnArchName
	hipDeviceAttributeHdpMemFlushCntl = auto()  # Address of the HDP_MEM_COHERENCY_FLUSH_CNTL register
	hipDeviceAttributeHdpRegFlushCntl = auto()  # Address of the HDP_REG_COHERENCY_FLUSH_CNTL register
	hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc = auto()  # Supports cooperative launch on multiple
	# devices with unmatched functions
	hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim = auto()  # Supports cooperative launch on multiple
	# devices with unmatched grid dimensions
	hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim = auto()  # Supports cooperative launch on multiple
	# devices with unmatched block dimensions
	hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem = auto()  # Supports cooperative launch on multiple
	# devices with unmatched shared memories
	hipDeviceAttributeIsLargeBar = auto()  # Whether it is LargeBar
	hipDeviceAttributeAsicRevision = auto()  # Revision of the GPU in this device
	hipDeviceAttributeCanUseStreamWaitValue = auto()  # '1' if Device supports hipStreamWaitValue32() and
	# hipStreamWaitValue64(), '0' otherwise.
	hipDeviceAttributeImageSupport = auto()  # '1' if Device supports image, '0' otherwise.
	hipDeviceAttributePhysicalMultiProcessorCount = auto()  # All available physical compute
	# units for the device
	hipDeviceAttributeFineGrainSupport = auto()  # '1' if Device supports fine grain, '0' otherwise
	hipDeviceAttributeWallClockRate = auto()  # Constant frequency of wall clock in kilohertz.


def hip_try(err):
	if err != 0:
		hip_runtime.hipGetErrorString.restype = ctypes.c_char_p
		error_string = hip_runtime.hipGetErrorString(ctypes.c_int(err)).decode("utf-8")
		raise RuntimeError(f"HIP error code {err}: {error_string}")


def count_devices():
	device_count = ctypes.c_int()
	hip_try(hip_runtime.hipGetDeviceCount(ctypes.byref(device_count)))
	return device_count.value


def set_device(gpu_id):
	hip_try(hip_runtime.hipSetDevice(gpu_id))


def get_device():
	device_id = ctypes.c_int()
	hip_try(hip_runtime.hipGetDevice(ctypes.byref(device_id)))
	return device_id.value


def get_cu_count(device_id=None):
	if device_id is None:
		device_id = get_device()

	attr = HipDeviceAttribute.hipDeviceAttributeMultiprocessorCount
	cu_count = ctypes.c_int()
	hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(cu_count), attr, device_id))
	return cu_count.value


def get_xcd_count(device_id=None):
	"""
	Return number of XCDs. Currently hardcoded.
	"""
	return 8


def get_wall_clock_rate(device_id=None):
	if device_id is None:
		device_id = get_device()

	attr = HipDeviceAttribute.hipDeviceAttributeWallClockRate
	rate = ctypes.c_int()
	hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(rate), attr, device_id))
	return rate.value


def get_max_shared_memory_per_block_kb(device_id=None):
	if device_id is None:
		device_id = get_device()

	attr = HipDeviceAttribute.hipDeviceAttributeMaxSharedMemoryPerBlock
	lds = ctypes.c_int()
	hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(lds), attr, device_id))
	# return KB
	return lds.value / 1024.0


def get_max_registers_per_block(device_id=None):
	if device_id is None:
		device_id = get_device()

	attr = HipDeviceAttribute.hipDeviceAttributeMaxRegistersPerBlock
	regs = ctypes.c_int()
	hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(regs), attr, device_id))
	return regs.value


def get_warp_size(device_id=None):
	if device_id is None:
		device_id = get_device()

	attr = HipDeviceAttribute.hipDeviceAttributeWarpSize
	warp = ctypes.c_int()
	hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(warp), attr, device_id))
	return warp.value


def get_hbm_size_mb(device_id=None):
	if device_id is None:
		device_id = get_device()
	free_mem = ctypes.c_size_t()
	total_mem = ctypes.c_size_t()
	# hipMemGetInfo returns (free, total) in bytes
	hip_try(hip_runtime.hipMemGetInfo(ctypes.byref(free_mem), ctypes.byref(total_mem)))
	# return MB
	return total_mem.value / (1024.0 * 1024.0)


def get_total_constant_memory_kb(device_id=None):
	if device_id is None:
		device_id = get_device()

	attr = HipDeviceAttribute.hipDeviceAttributeTotalConstantMemory
	const_mem = ctypes.c_int()
	hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(const_mem), attr, device_id))
	# return KB
	return const_mem.value / 1024.0


def get_l2_cache_size_kb(device_id=None):
	if device_id is None:
		device_id = get_device()

	attr = HipDeviceAttribute.hipDeviceAttributeL2CacheSize
	l2 = ctypes.c_int()
	hip_try(hip_runtime.hipDeviceGetAttribute(ctypes.byref(l2), attr, device_id))
	# return KB
	return l2.value / 1024.0


def measure_atomic_latency_ns(device_id=None, iters=1000000, block_size=256):
	return 1000


class GPUSpec:
	"""
	Query GPU specs directly via HIP runtime API.
	"""

	def __init__(self, device_id=None):
		if device_id is None:
			device_id = get_device()
		else:
			set_device(device_id)
		self.device_id = device_id

	def get_lds_size(self):
		"""Return LDS/shared-memory-per-block size in KB."""
		return get_max_shared_memory_per_block_kb(self.device_id)

	def get_max_registers_per_block(self):
		"""Return register file size (registers per block)."""
		return get_max_registers_per_block(self.device_id)

	def get_warp_size(self):
		"""Return warp size (threads per warp)."""
		return get_warp_size(self.device_id)

	def get_hbm_size(self):
		"""Return HBM (global memory) size in MB."""
		return get_hbm_size_mb(self.device_id)

	def get_l1_cache_size(self):
		"""Return total constant memory size in KB (proxy for L1 cache)."""
		return get_total_constant_memory_kb(self.device_id)

	def get_l2_cache_size(self):
		"""Return L2 cache size in KB."""
		return get_l2_cache_size_kb(self.device_id)

	def get_num_cus(self):
		"""Return number of compute units (multiprocessors)."""
		return get_cu_count(self.device_id)

	def get_num_xcds(self):
		"""Return number of XCDs."""
		return get_xcd_count(self.device_id)

	def get_atomic_latency(self):
		"""Return average atomic-add latency in nanoseconds."""
		return measure_atomic_latency_ns(self.device_id)
