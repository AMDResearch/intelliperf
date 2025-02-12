#include <cxxabi.h>
#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <dlfcn.h>
#include <link.h>
#include <csetjmp>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

#include "ipc_helper.hpp"

#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <fstream>
#include <iostream>

#include <hip/hip_runtime.h>
#include <iostream>

void print_all_symbols_from_elf(const char* binary_path) {
  std::cout << "print_all_symbols_from_elf" << std::endl;
  int fd = open(binary_path, O_RDONLY);
  if (fd < 0) {
    std::cerr << "[Maestro Tracer] Failed to open ELF file: " << binary_path
              << std::endl;
    return;
  }

  void* map = mmap(nullptr, sizeof(Elf64_Ehdr), PROT_READ, MAP_PRIVATE, fd, 0);
  if (map == MAP_FAILED) {
    std::cerr << "[Maestro Tracer] Failed to mmap ELF header" << std::endl;
    close(fd);
    return;
  }

  Elf64_Ehdr* ehdr = static_cast<Elf64_Ehdr*>(map);
  Elf64_Shdr* shdr = reinterpret_cast<Elf64_Shdr*>((char*)map + ehdr->e_shoff);
  Elf64_Shdr* symtab = nullptr;
  Elf64_Shdr* strtab = nullptr;

  for (int i = 0; i < ehdr->e_shnum; ++i) {
    std::cout << "print_all_symbols_from_elf" << std::endl;
    if (shdr[i].sh_type == SHT_SYMTAB) {
      symtab = &shdr[i];
    }
    if (shdr[i].sh_type == SHT_STRTAB && i == ehdr->e_shstrndx) {
      strtab = &shdr[i];
    }
  }
  std::cout << "print_all_symbols_from_elf" << std::endl;

  if (!symtab || !strtab) {
    std::cerr << "[Maestro Tracer] ELF symtab or strtab not found" << std::endl;
    munmap(map, sizeof(Elf64_Ehdr));
    close(fd);
    return;
  }

  Elf64_Sym* symbols =
      reinterpret_cast<Elf64_Sym*>((char*)map + symtab->sh_offset);
  const char* strtab_data = (char*)map + strtab->sh_offset;

  std::cout << "[Maestro Tracer] Symbols in " << binary_path << ":"
            << std::endl;
  for (size_t i = 0; i < symtab->sh_size / sizeof(Elf64_Sym); ++i) {
    std::cout << "  " << std::hex << symbols[i].st_value << " "
              << strtab_data + symbols[i].st_name
              << " (Type: " << (int)ELF64_ST_TYPE(symbols[i].st_info) << ")"
              << std::endl;
  }

  munmap(map, sizeof(Elf64_Ehdr));
  close(fd);
}

hipFunction_t get_hip_function(const char* kernel_name, hipModule_t module) {
  hipFunction_t function;
  hipError_t err = hipModuleGetFunction(&function, module, kernel_name);
  if (err != hipSuccess) {
    std::cerr << "Error: hipModuleGetFunction failed for " << kernel_name
              << " (" << hipGetErrorString(err) << ")" << std::endl;
    return nullptr;
  }
  return function;
}
std::unordered_map<hipModule_t, std::string> loaded_modules;
std::mutex module_mutex;

extern "C" hipError_t hipModuleLoad(hipModule_t* module, const char* fname) {
  static auto real_hipModuleLoad =
      reinterpret_cast<hipError_t (*)(hipModule_t*, const char*)>(
          dlsym(RTLD_NEXT, "hipModuleLoad"));

  hipError_t result = real_hipModuleLoad(module, fname);

  if (result == hipSuccess && module) {
    std::lock_guard<std::mutex> lock(module_mutex);
    loaded_modules[*module] = fname;
    std::cout << "[Maestro Tracer] Loaded HIP Module: " << fname << " -> "
              << *module << std::endl;
  } else {
    std::cerr << "[Maestro Tracer] Failed to load HIP Module: " << fname
              << std::endl;
  }

  return result;
}

std::string get_function_name(const void* function_address) {
  Dl_info info;
  if (dladdr(function_address, &info) && info.dli_sname) {
    std::string name(info.dli_sname);
    return name;
  }
  std::ostringstream oss;
  oss << function_address;
  return "Function " + oss.str() + " not found";
}

struct hsa_executable_symbol_hasher {
  std::size_t operator()(const hsa_executable_symbol_t& symbol) const {
    return std::hash<uint64_t>()(symbol.handle);
  }
};
struct hsa_executable_symbol_compare {
  using result_type = bool;
  using first_argument_type = hsa_executable_symbol_t;
  using second_argument_type = hsa_executable_symbol_t;

  bool operator()(const hsa_executable_symbol_t& lhs,
                  const hsa_executable_symbol_t& rhs) const {
    return lhs.handle == rhs.handle;
  }
};

std::map<void*, std::size_t> hip_pointer_sizes_;
std::map<void*, std::size_t> hsa_pointer_sizes_;
std::map<void*, hipIpcMemHandle_t> pointer_ipc_handles_;
std::unordered_map<hsa_executable_symbol_t,
                   std::string,
                   hsa_executable_symbol_hasher,
                   hsa_executable_symbol_compare>
    symbols_names_;
std::unordered_map<std::string, hsa_executable_t> kernels_executables_;
std::unordered_map<std::uint64_t, hsa_executable_symbol_t> handles_symbols_;

static std::mutex mutex_;

std::jmp_buf jump_buffer;

// std::string get_function_name(const void* function_address) {
//   Dl_info info;
//   if (dladdr(function_address, &info)) {
//     if (info.dli_sname) {
//       return info.dli_sname;
//     } else {
//       return "Function not found";
//     }
//   } else {
//     return "Function not found";
//   }
// }

bool is_logging_enabled() {
  const char* log_env = std::getenv("LOG_MAESTRO_TRACER");
  return log_env != nullptr;
}

void signal_handler(int signal) {
  if (signal == SIGSEGV) {
    std::cerr << "Segmentation fault detected in print_kernel_arguments. "
                 "Recovering..."
              << std::endl;
    std::longjmp(jump_buffer, 1);
  }
}
std::string demangle_name(const char* mangled_name) {
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> result(
      abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status), std::free);
  return (status == 0) ? result.get() : mangled_name;
}

void unsafe_print_kernel_arguments(void** args) {
  if (!args) {
    std::cout << "  Kernel Arguments: (nullptr)" << std::endl;
    return;
  }
  std::signal(SIGSEGV, signal_handler);
  if (setjmp(jump_buffer) == 0) {
    int i = 0;
    while (true) {
      if (args[i] == nullptr)
        break;
      void* actual_ptr = *reinterpret_cast<void**>(args[i]);
      if (actual_ptr == nullptr)
        break;

      std::cout << "    Arg[" << i << "]: " << args[i]
                << " (Stored Pointer: " << std::hex << actual_ptr << std::dec
                << ")" << std::endl;
      ++i;
    }
  } else {
    std::cerr << "  Error: print_kernel_arguments encountered invalid memory "
                 "access. Stopping iteration."
              << std::endl;
  }
}

void print_kernel_arguments_exact(void** args) {
  if (!args) {
    std::cout << "  Kernel Arguments: (nullptr)" << std::endl;
    return;
  }

  std::size_t total_size = 0;
  int num_args = -1;

  // Lookup args itself in hsa_pointer_sizes_
  auto it = hsa_pointer_sizes_.find(args);
  if (it != hsa_pointer_sizes_.end()) {
    total_size = it->second;
    num_args = total_size / sizeof(void*);
  }

  std::cout << "  Kernel Arguments (Detected " << num_args
            << " args):" << std::endl;

  for (int i = 0; i < num_args; ++i) {
    void* actual_ptr = *reinterpret_cast<void**>(args[i]);

    std::cout << "    Arg[" << i << "]: " << args[i]
              << " (Stored Pointer: " << std::hex << actual_ptr << std::dec
              << ")" << std::endl;
  }
}

void print_kernel_arguments(void** args) {
  if (!args) {
    std::cout << "  Kernel Arguments: (nullptr)" << std::endl;
    return;
  }

  std::cout << "  Kernel Arguments:" << std::endl;
  bool found{false};
  for (const auto& entry : hsa_pointer_sizes_) {
    void* base = entry.first;
    std::size_t size = entry.second;
    void* end =
        reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(base) + size);

    void* target = *args;

    std::cout << "  -> Checking " << target << " inside range [" << base
              << " - " << end << "] (Size: " << size << " bytes)" << std::endl;

    if (reinterpret_cast<uintptr_t>(target) >=
            reinterpret_cast<uintptr_t>(base) &&
        reinterpret_cast<uintptr_t>(target) <
            reinterpret_cast<uintptr_t>(end)) {
      std::cout << "  -> args is within this range!" << std::endl;
      found = true;
    }
  }
  if (!found) {
    std::cout << args << " " << *args << " was not found" << std::endl;
  }
}

void printHipIpcMemHandle(const hipIpcMemHandle_t& handle,
                          const std::string& message) {
  const unsigned char* data = reinterpret_cast<const unsigned char*>(&handle);
  std::cout << "[Maestro Tracer] " << message
            << " hipIpcMemHandle_t contents:" << std::endl;
  for (size_t i = 0; i < sizeof(hipIpcMemHandle_t); ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(data[i]) << " ";
    if ((i + 1) % 16 == 0)
      std::cout << std::endl;
  }
  std::cout << std::dec;
}

void* load_hip_function(const char* func_name) {
  static void* hip_lib = dlopen("libamdhip64.so", RTLD_LAZY);
  if (!hip_lib) {
    throw std::runtime_error("Failed to load HIP library");
  }

  void* func = dlsym(hip_lib, func_name);
  if (!func) {
    throw std::runtime_error(std::string("Failed to load function: ") +
                             func_name);
  }

  return func;
}

void* load_hsa_function(const char* func_name) {
  return load_hip_function(func_name);
}

// std::string get_kernel_name(const std::uint64_t kernel_object) {
//   auto handle_find_result = handles_symbols_.find(kernel_object);
//   if (handle_find_result == handles_symbols_.end()) {
//     return "Object " + std::to_string(kernel_object) + " not found.";
//   }
//   auto symbol_find_result = symbols_names_.find(handle_find_result->second);
//   if (handle_find_result == handles_symbols_.end()) {
//     return "Symbol not found.";
//   }
//   return demangle_name(symbol_find_result->second.c_str());
// }

int print_function_callback(struct dl_phdr_info* info,
                            size_t size,
                            void* data) {
  std::cout << "[Maestro Tracer] Library: " << info->dlpi_name << std::endl;

  void* handle = dlopen(info->dlpi_name, RTLD_NOW | RTLD_GLOBAL);
  if (!handle)
    return 0;

  // Iterate through all possible symbols
  for (int i = 0; i < 10000; ++i) {  // Arbitrary limit to avoid infinite loop
    void* func_address = dlsym(handle, std::to_string(i).c_str());
    if (func_address) {
      std::cout << "  - " << i << " -> " << func_address << std::endl;
    }
  }

  dlclose(handle);
  return 0;
}

void print_all_kernel_functions() {
  dl_iterate_phdr(print_function_callback, nullptr);
}
hipFunction_t get_hip_kernel_function(const char* kernel_name) {
  void* func_address = dlsym(RTLD_DEFAULT, kernel_name);
  if (func_address) {
    return reinterpret_cast<hipFunction_t>(func_address);
  }
  return nullptr;
}

std::string get_kernel_name(void* function_address) {
  const char* kernel_list = std::getenv("KERNEL_TO_TRACE");
  if (!kernel_list) {
    return "Unknown Function";
  }

  char* kernels = strdup(kernel_list);
  char* token = std::strtok(kernels, ",");
  while (token) {
    hipFunction_t func = get_hip_kernel_function(token);
    if (func) {
      // if (func && reinterpret_cast<void*>(func) == function_address) {
      std::string name = token;
      free(kernels);
      return name;
    }
    token = std::strtok(nullptr, ",");
  }
  free(kernels);
  return "## Unknown Function";
}

extern "C" hsa_status_t hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute,
    void* value) {
  static auto real_hsa_executable_symbol_get_info =
      reinterpret_cast<hsa_status_t (*)(hsa_executable_symbol_t,
                                        hsa_executable_symbol_info_t, void*)>(
          load_hsa_function("hsa_executable_symbol_get_info"));

  hsa_status_t result =
      real_hsa_executable_symbol_get_info(executable_symbol, attribute, value);

  if (result == HSA_STATUS_SUCCESS) {
    std::lock_guard<std::mutex> lock(mutex_);
    // std::cout << "[Maestro Tracer] Retrieved symbol info for handle "
    //           << executable_symbol.handle << " (Attribute: " << attribute <<
    //           ")"
    //           << std::endl;
    handles_symbols_[*static_cast<std::uint64_t*>(value)] = executable_symbol;
  }

  return result;
}

extern "C" hsa_status_t hsa_executable_get_symbol_by_name(
    hsa_executable_t executable,
    const char* symbol_name,
    const hsa_agent_t* agent,
    hsa_executable_symbol_t* symbol) {
  static auto real_hsa_executable_get_symbol_by_name =
      reinterpret_cast<hsa_status_t (*)(hsa_executable_t, const char*,
                                        const hsa_agent_t*,
                                        hsa_executable_symbol_t*)>(
          load_hsa_function("hsa_executable_get_symbol_by_name"));

  hsa_status_t result = real_hsa_executable_get_symbol_by_name(
      executable, symbol_name, agent, symbol);

  if (result == HSA_STATUS_SUCCESS && symbol) {
    // std::cout << "[Maestro Tracer] Retrieved symbol '"
    //           << (symbol_name ? symbol_name : "(nullptr)")
    //           << "' for executable " << executable.handle << " and agent "
    //           << (agent ? agent->handle : 0)
    //           << " -> Symbol Handle: " << symbol->handle << std::endl;

    {
      std::lock_guard g(mutex_);
      const std::string kernel_name = std::string(symbol_name);
      symbols_names_[*symbol] = kernel_name;
      kernels_executables_[kernel_name] = executable;
    }
  }

  return result;
}

extern "C" hsa_status_t hsa_memory_allocate(hsa_region_t region,
                                            size_t size,
                                            void** ptr) {
  static auto real_hsa_memory_allocate =
      reinterpret_cast<hsa_status_t (*)(hsa_region_t, size_t, void**)>(
          load_hip_function("hsa_memory_allocate"));

  hsa_status_t result = real_hsa_memory_allocate(region, size, ptr);

  if (result == HSA_STATUS_SUCCESS && ptr && *ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    hsa_pointer_sizes_[*ptr] = size;
    std::cout << "[Maestro Tracer] HSA Allocated " << size << " bytes at "
              << *ptr << std::endl;
  }

  return result;
}

extern "C" hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool,
                                                     size_t size,
                                                     uint32_t flags,
                                                     void** ptr) {
  static auto real_hsa_amd_memory_pool_allocate =
      reinterpret_cast<hsa_status_t (*)(hsa_amd_memory_pool_t, size_t, uint32_t,
                                        void**)>(
          load_hip_function("hsa_amd_memory_pool_allocate"));

  hsa_status_t result =
      real_hsa_amd_memory_pool_allocate(pool, size, flags, ptr);

  if (result == HSA_STATUS_SUCCESS && ptr && *ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    hsa_pointer_sizes_[*ptr] = size;
    std::cout << "[Maestro Tracer] HSA Allocated " << size
              << " bytes from pool at " << *ptr << " (Flags: " << flags << ")"
              << std::endl;
  }

  return result;
}

extern "C" hipError_t hipMalloc(void** devPtr, size_t size) {
  static auto real_hipMalloc = reinterpret_cast<hipError_t (*)(void**, size_t)>(
      load_hip_function("hipMalloc"));

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipMalloc called with size: " << size
              << " bytes" << std::endl;
  }

  hipError_t result = real_hipMalloc(devPtr, size);

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipMalloc result: " << result
              << ", ptr: " << *devPtr << std::endl;
  }

  // Now get the IPC handle
  if (result == hipSuccess && *devPtr) {
    std::lock_guard guard(mutex_);
    void* device_ptr = *devPtr;
    hip_pointer_sizes_[device_ptr] = size;

    hipIpcMemHandle_t handle;
    hipError_t ipc_result = hipIpcGetMemHandle(&handle, device_ptr);
    if (ipc_result != hipSuccess) {
      std::cerr << "[Maestro Tracer] Failed to get IPC handle for pointer "
                << device_ptr << ": " << ipc_result << "\n";
    } else if (is_logging_enabled()) {
      std::cout << "[Maestro Tracer] IPC handle obtained successfully\n";
      pointer_ipc_handles_[device_ptr] = handle;
      // send_ipc_bytes(handle);
    }
  }

  return result;
}

extern "C" hipError_t hipLaunchKernel(const void* function_address,
                                      dim3 gridDim,
                                      dim3 blockDim,
                                      void** args,
                                      size_t sharedMem,
                                      hipStream_t stream) {
  static auto real_hipLaunchKernel = reinterpret_cast<hipError_t (*)(
      const void*, dim3, dim3, void**, size_t, hipStream_t)>(
      load_hip_function("hipLaunchKernel"));

  // print_all_symbols_from_elf("/proc/self/exe");

  std::cout << "[Maestro Tracer] hipLaunchKernel" << std::endl;
  std::cout << "  Function Address: " << function_address << std::endl;
  std::cout << "  Function Name: "
            << get_kernel_name(const_cast<void*>(function_address))
            << std::endl;
  // std::cout << "  Function Name: " << get_function_name(function_address)
  //           << std::endl;

  std::cout << "  Grid Dim: (" << gridDim.x << ", " << gridDim.y << ", "
            << gridDim.z << ")" << std::endl;
  std::cout << "  Block Dim: (" << blockDim.x << ", " << blockDim.y << ", "
            << blockDim.z << ")" << std::endl;
  std::cout << "  Shared Memory: " << sharedMem << " bytes" << std::endl;
  std::cout << "  Stream: " << stream << std::endl;

  // print_kernel_arguments(args);

  hipError_t result = real_hipLaunchKernel(function_address, gridDim, blockDim,
                                           args, sharedMem, stream);

  std::cout << "[Maestro Tracer] hipLaunchKernel result: " << result
            << std::endl;

  return result;
}

extern "C" hipError_t hipModuleLaunchKernel(hipFunction_t function,
                                            uint32_t gridDimX,
                                            uint32_t gridDimY,
                                            uint32_t gridDimZ,
                                            uint32_t blockDimX,
                                            uint32_t blockDimY,
                                            uint32_t blockDimZ,
                                            uint32_t sharedMemBytes,
                                            hipStream_t stream,
                                            void** kernelParams,
                                            void** extra) {
  static auto real_hipModuleLaunchKernel = reinterpret_cast<hipError_t (*)(
      hipFunction_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
      uint32_t, hipStream_t, void**, void**)>(
      load_hip_function("hipModuleLaunchKernel"));

  std::cout << "[Maestro Tracer] HIP Module Kernel Launch Detected!"
            << std::endl;
  std::cout << "  Function: " << function << std::endl;
  std::cout << "  Grid Dim: (" << gridDimX << ", " << gridDimY << ", "
            << gridDimZ << ")" << std::endl;
  std::cout << "  Block Dim: (" << blockDimX << ", " << blockDimY << ", "
            << blockDimZ << ")" << std::endl;
  std::cout << "  Shared Memory: " << sharedMemBytes << " bytes" << std::endl;
  std::cout << "  Stream: " << stream << std::endl;

  // print_kernel_arguments(kernelParams);

  hipError_t result = real_hipModuleLaunchKernel(
      function, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
      sharedMemBytes, stream, kernelParams, extra);

  std::cout << "[Maestro Tracer] hipModuleLaunchKernel result: " << result
            << std::endl;

  return result;
}

extern "C" hipError_t hipFree(void* devPtr) {
  static auto real_hipFree =
      reinterpret_cast<hipError_t (*)(void*)>(load_hip_function("hipFree"));

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipFree called for pointer: " << devPtr
              << std::endl;
  }

  hipError_t result = real_hipFree(devPtr);

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipFree result: " << result << std::endl;
  }

  return result;
}

extern "C" hipError_t hipIpcGetMemHandle(hipIpcMemHandle_t* handle,
                                         void* devPtr) {
  static auto real_hipIpcGetMemHandle =
      reinterpret_cast<hipError_t (*)(hipIpcMemHandle_t*, void*)>(
          load_hip_function("hipIpcGetMemHandle"));

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipIpcGetMemHandle called for pointer: "
              << devPtr << std::endl;
  }

  hipError_t result = real_hipIpcGetMemHandle(handle, devPtr);

  // if (is_logging_enabled()) {
  //   std::cout << "[Maestro Tracer] hipIpcGetMemHandle result: " << result
  //             << std::endl;
  //   printHipIpcMemHandle(*handle, "hipIpcGetMemHandle");
  // }

  return result;
}

extern "C" hipError_t hipIpcOpenMemHandle(void** devPtr,
                                          hipIpcMemHandle_t handle,
                                          unsigned int flags) {
  static auto real_hipIpcOpenMemHandle =
      reinterpret_cast<hipError_t (*)(void**, hipIpcMemHandle_t, unsigned int)>(
          load_hip_function("hipIpcOpenMemHandle"));

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipIpcOpenMemHandle called with flags: "
              << flags << std::endl;
    printHipIpcMemHandle(handle, "hipIpcOpenMemHandle");
  }

  hipError_t result = real_hipIpcOpenMemHandle(devPtr, handle, flags);

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipIpcOpenMemHandle result: " << result
              << ", ptr: " << *devPtr << std::endl;
  }

  return result;
}
extern "C" hipError_t hipGetDeviceCount(int* count) {
  static auto real_hipGetDeviceCount = reinterpret_cast<hipError_t (*)(int*)>(
      load_hip_function("hipGetDeviceCount"));

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipGetDeviceCount called" << std::endl;
  }

  hipError_t result = real_hipGetDeviceCount(count);

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipGetDeviceCount result: " << result
              << ", count: " << *count << std::endl;
  }

  return result;
}

extern "C" hipError_t hipSetDevice(int deviceId) {
  static auto real_hipSetDevice =
      reinterpret_cast<hipError_t (*)(int)>(load_hip_function("hipSetDevice"));

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipSetDevice called with deviceId: "
              << deviceId << std::endl;
  }

  hipError_t result = real_hipSetDevice(deviceId);

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipSetDevice result: " << result
              << std::endl;
  }

  return result;
}

extern "C" hipError_t hipGetDevice(int* deviceId) {
  static auto real_hipGetDevice =
      reinterpret_cast<hipError_t (*)(int*)>(load_hip_function("hipGetDevice"));

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipGetDevice called" << std::endl;
  }

  hipError_t result = real_hipGetDevice(deviceId);

  if (is_logging_enabled()) {
    std::cout << "[Maestro Tracer] hipGetDevice result: " << result
              << ", deviceId: " << *deviceId << std::endl;
  }

  return result;
}
