/****************************************************************************
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
****************************************************************************/

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <map>
#include <string>
#include <vector>

#define hsa_try(expr)                                                            \
  do {                                                                           \
    hsa_status_t status = (expr);                                                \
    if (status != HSA_STATUS_SUCCESS) {                                          \
      const char* status_string;                                                 \
      hsa_status_string(status, &status_string);                                 \
      throw std::runtime_error(std::string("HSA Error at ") + __FILE__ + ":" +   \
                               std::to_string(__LINE__) + ": " + status_string); \
    }                                                                            \
  } while (0)

#define hip_try(expr)                                                          \
  do {                                                                         \
    hipError_t status = (expr);                                                \
    if (status != hipSuccess) {                                                \
      throw std::runtime_error(std::string("HIP Error at ") + __FILE__ + ":" + \
                               std::to_string(__LINE__) + ": " +               \
                               hipGetErrorString(status));                     \
    }                                                                          \
  } while (0)

namespace vmem_alloc {

std::string get_agent_name(hsa_agent_t agent) {
  hsa_device_type_t device_type;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);
  return device_type == HSA_DEVICE_TYPE_GPU ? "GPU" : "CPU";
}

struct AgentInfo {
  hsa_agent_t agent;
  std::string name;
};

std::vector<AgentInfo> get_gpu_agents() {
  std::vector<AgentInfo> agents;
  hsa_try(hsa_iterate_agents(
      [](hsa_agent_t agent, void* data) {
        const auto agents = (std::vector<AgentInfo>*)data;
        if (get_agent_name(agent) == "GPU") {
          agents->push_back({agent, get_agent_name(agent)});
        }
        return HSA_STATUS_SUCCESS;
      },
      &agents));
  return agents;
}

struct AllocationInfo {
  size_t size;
  hsa_amd_vmem_alloc_handle_t handle;
};

std::map<void*, AllocationInfo> g_allocations;

// Base virtual memory allocation function that takes a fine-grained flag and
// alignment
void* malloc_vmem_base(size_t alignment,
                       size_t size,
                       void* requested_address,
                       bool fine_grained) {
  // Step 1: Find GPU agents and get a memory pool
  auto gpu_agents = get_gpu_agents();
  if (gpu_agents.empty()) {
    std::fprintf(stderr, "No GPU agents found\n");
    return nullptr;
  }

  // Find memory pool with appropriate flags
  struct memory_pool_data_t {
    hsa_amd_memory_pool_t memory_pool;
    bool want_fine_grained;
  };
  memory_pool_data_t memory_pool_data = {hsa_amd_memory_pool_t{}, fine_grained};
  hsa_try(hsa_amd_agent_iterate_memory_pools(
      gpu_agents[0].agent,
      [](hsa_amd_memory_pool_t pool, void* data) {
        hsa_amd_memory_pool_global_flag_t flags;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        auto memory_pool_data = static_cast<memory_pool_data_t*>(data);
        bool want_fine_grained = memory_pool_data->want_fine_grained;
        if (want_fine_grained) {
          if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED) {
            memory_pool_data->memory_pool = pool;
            return HSA_STATUS_SUCCESS;
          }
        } else {
          if (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
            memory_pool_data->memory_pool = pool;
            return HSA_STATUS_SUCCESS;
          }
        }
        return HSA_STATUS_SUCCESS;
      },
      &memory_pool_data));

  // Step 2: Reserve virtual address range
  void* reserved_address = requested_address;
  hsa_try(hsa_amd_vmem_address_reserve_align(
      &reserved_address, size, (uint64_t)requested_address, alignment, 0));

  if (reserved_address != requested_address) {
    std::fprintf(stderr,
                 "Failed to reserve at requested address %p, got %p instead\n",
                 requested_address,
                 reserved_address);
    hsa_amd_vmem_address_free(reserved_address, size);
    return nullptr;
  }

  // Step 3: Create memory handle
  hsa_amd_vmem_alloc_handle_t memory_handle;
  hsa_try(hsa_amd_vmem_handle_create(
      memory_pool_data.memory_pool, size, MEMORY_TYPE_NONE, 0, &memory_handle));

  // Step 4: Map virtual address to memory handle
  hsa_try(hsa_amd_vmem_map(reserved_address, size, 0, memory_handle, 0));

  // Step 5: Set access permissions for all GPU agents
  std::vector<hsa_amd_memory_access_desc_t> access_descs;
  access_descs.reserve(gpu_agents.size());
  for (const auto& agent_info : gpu_agents) {
    access_descs.push_back({HSA_ACCESS_PERMISSION_RW, agent_info.agent});
  }
  hsa_try(hsa_amd_vmem_set_access(
      reserved_address, size, access_descs.data(), access_descs.size()));

  // Step 6: Store allocation info for cleanup
  AllocationInfo info = {size, memory_handle};
  g_allocations[reserved_address] = info;

  return reserved_address;
}

// Fine-grained virtual memory allocation
void* malloc_vmem_fine_grained(size_t size, void* requested_address) {
  return malloc_vmem_base(4096, size, requested_address, true);
}

// Coarse-grained virtual memory allocation
void* malloc_vmem_coarse_grained(size_t size, void* requested_address) {
  return malloc_vmem_base(4096, size, requested_address, false);
}

// Default malloc_vmem implementation that uses fine-grained memory
void* malloc_vmem(size_t size, void* requested_address) {
  return malloc_vmem_fine_grained(size, requested_address);
}

// Aligned fine-grained virtual memory allocation
void* aligned_alloc_vmem_fine_grained(size_t alignment,
                                      size_t size,
                                      void* requested_address) {
  return malloc_vmem_base(alignment, size, requested_address, true);
}

// Aligned coarse-grained virtual memory allocation
void* aligned_alloc_vmem_coarse_grained(size_t alignment,
                                        size_t size,
                                        void* requested_address) {
  return malloc_vmem_base(alignment, size, requested_address, false);
}

// Default aligned malloc_vmem implementation that uses fine-grained memory
void* aligned_alloc_vmem(size_t alignment, size_t size, void* requested_address) {
  return aligned_alloc_vmem_fine_grained(alignment, size, requested_address);
}

// Virtual memory deallocation sequence (reverse of allocation):
// 1. Find allocation info
// 2. Unmap virtual address
// 3. Release memory handle
// 4. Free virtual address range
// 5. Remove from tracking map
void vmem_free(void* address) {
  auto it = g_allocations.find(address);
  if (it == g_allocations.end()) {
    std::fprintf(stderr, "Error: Attempting to free unknown address %p\n", address);
    return;
  }

  const auto& info = it->second;

  // Step 2: Unmap virtual address
  hsa_try(hsa_amd_vmem_unmap(address, info.size));

  // Step 3: Release memory handle
  hsa_try(hsa_amd_vmem_handle_release(info.handle));

  // Step 4: Free virtual address range
  hsa_try(hsa_amd_vmem_address_free(address, info.size));

  // Step 5: Remove from tracking map
  g_allocations.erase(it);
}
}  // namespace vmem_alloc