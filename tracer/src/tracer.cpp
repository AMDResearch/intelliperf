
#define __HIP_PLATFORM_AMD__

#include "tracer.hpp"
#include <fmt/core.h>
#include "log.hpp"

#include <cxxabi.h>
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <exception>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>

#include "KernelArguments.hpp"

#define PIPE_NAME "/tmp/kernel_pipe"
#define HANDSHAKE_PIPE "/tmp/handshake_pipe"
#define FILE_NAME "/tmp/ipc_handle.bin"

namespace maestro {

std::mutex tracer::mutex_{};
std::shared_mutex tracer::stop_mutex_{};
tracer* tracer::singleton_{nullptr};

tracer::tracer(HsaApiTable* table,
               uint64_t runtime_version,
               uint64_t failed_tool_count,
               const char* const* failed_tool_names)
    : api_table_{table} {
  LOG_INFO("Saving current APIs.");
  save_hsa_api();
  LOG_INFO("Hooking new APIs.");
  hook_api();
  LOG_INFO("Discovering agents.");
  discover_agents();

  for (const auto& pair : agents_names_) {
    LOG_INFO("Agent Handle: 0x{:x} , Name: {}", pair.first.handle, pair.second);
  }

  // Create the FIFO
  if (mkfifo(PIPE_NAME, 0666) == -1 && errno != EEXIST) {
    perror("mkfifo failed");
  }
  if (mkfifo(HANDSHAKE_PIPE, 0666) == -1 && errno != EEXIST) {
    perror("mkfifo failed");
  }
}

template <typename T, typename Func, std::size_t... Is>
inline void for_each_field_impl(const T& obj,
                                Func func,
                                std::index_sequence<Is...>) {
  (func(std::get<Is>(obj->as_tuple())), ...);
}
template <typename T, typename Func>
inline void for_each_field(const T& obj, Func func) {
  constexpr std::size_t N = std::tuple_size_v<decltype(obj->as_tuple())>;
  for_each_field_impl(obj, func, std::make_index_sequence<N>{});
}

void printHipIpcMemHandle(const hipIpcMemHandle_t& handle,
                          const std::string& message) {
  const unsigned char* data = reinterpret_cast<const unsigned char*>(&handle);
  std::cout << "[HIP SHIM] " << message
            << " hipIpcMemHandle_t contents:" << std::endl;
  for (size_t i = 0; i < sizeof(hipIpcMemHandle_t); ++i) {
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(data[i]) << " ";
    if ((i + 1) % 16 == 0)
      std::cout << std::endl;
  }
  std::cout << std::dec;
  std::cout << std::endl;
}

void writeIpcHandleToFIFO(int fd, const hipIpcMemHandle_t& handle) {
  const uint8_t* handle_bytes = reinterpret_cast<const uint8_t*>(&handle);

  // I do not get why i lose some bytes at the very beginning...
  // for (size_t i = 0; i < 1; i += 8) {
  //   ssize_t bytes_written = write(fd, handle_bytes + i, 8);
  //   fsync(fd);
  // }
  // write(fd, handle_bytes, 0);
  // fsync(fd);

  for (size_t i = 0; i < sizeof(handle); i += 8) {
    ssize_t bytes_written = write(fd, handle_bytes + i, 8);
    fsync(fd);

    std::cerr << "[HIP SHIM] Wrote " << bytes_written << " bytes at offset "
              << i << ":\n";

    for (size_t j = 0; j < 8; ++j) {
      std::cerr << std::hex << std::setfill('0') << std::setw(2)
                << static_cast<int>(handle_bytes[i + j]) << " ";
    }
    std::cerr << "\n";

    if (bytes_written != 8) {
      perror("[HIP SHIM] Failed to write full 8-byte chunk");
      return;
    }
  }

  std::cerr << "[HIP SHIM] Finished writing full IPC handle (64 bytes)\n";
}

void writeIpcHandleToFile(const hipIpcMemHandle_t& handle, size_t ptr_size) {
  std::ofstream file(FILE_NAME, std::ios::binary | std::ios::app);
  if (!file) {
    std::cerr << "[C++] Failed to open file for writing: " << FILE_NAME << "\n";
    return;
  }

  file << "BEGIN\n";
  file.write(reinterpret_cast<const char*>(&handle), sizeof(handle));
  file.write(reinterpret_cast<const char*>(&ptr_size), sizeof(ptr_size));
  file << "END\n";
  file.flush();
  file.close();

  std::cerr << "[C++] Appended IPC handle and size (" << ptr_size
            << " bytes) to file: " << FILE_NAME << "\n";
}
void tracer::send_message_and_wait(void* args) {
  const auto args_struct = reinterpret_cast<KernelArguments*>(args);

  for_each_field(args_struct, [](const auto& field) {
    if constexpr (std::is_pointer_v<std::decay_t<decltype(field)>>) {
      LOG_DETAIL("Field (pointer): {}", static_cast<void*>(field));
    } else {
      LOG_DETAIL("Field: {}", field);
    }
  });

  // int handshake_fd = open(HANDSHAKE_PIPE, O_RDONLY);
  // if (handshake_fd < 0) {
  //   perror("[C++] Failed to open handshake FIFO");
  //   return;
  // }
  // char buffer[10];
  // read(handshake_fd, buffer, sizeof(buffer));
  // close(handshake_fd);

  // if (strncmp(buffer, "READY", 5) != 0) {
  //   std::cerr << "[C++] Unexpected handshake message: " << buffer << "\n";
  //   return;
  // }
  // std::cerr
  //     << "[C++] Received READY signal from Python, writing IPC handle...\n";

  // Export IPC handles
  int fd = open(PIPE_NAME, O_WRONLY);
  if (fd < 0) {
    perror("Failed to open FIFO for writing");
    return;
  }
  for_each_field(args_struct, [fd](const auto& field) {
    if constexpr (std::is_pointer_v<std::decay_t<decltype(field)>>) {
      hipIpcMemHandle_t handle;
      hipError_t ipc_result = hipIpcGetMemHandle(&handle, field);
      if (ipc_result != hipSuccess) {
        std::cerr << "[Maestro Tracer] Failed to get IPC handle for pointer "
                  << field << ": " << ipc_result << "\n";
      }
      // const auto bytes_written = write(fd, &handle, sizeof(handle));
      // fsync(fd);
      printHipIpcMemHandle(handle, "handle");

      size_t ptr_size = 0;
      {
        auto instance = get_instance();
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = instance->pointer_sizes_.find(field);
        if (it != instance->pointer_sizes_.end()) {
          ptr_size = it->second;
        } else {
          std::cerr << "[C++] Warning: Pointer size not found for " << field
                    << "\n";
        }
      }
      writeIpcHandleToFile(handle, ptr_size);
      // writeIpcHandleToFIFO(fd, handle);
      // LOG_DETAIL("Field (pointer): {} -- {}", static_cast<void*>(field),
      //            bytes_written);

      // std::this_thread::sleep_for(std::chrono::seconds(5));
      // if (bytes_written != sizeof(handle)) {
      //   perror("Failed to write IPC handle to FIFO");
      // }
    }
  });

  // close(fd);
  // printf("Intercepted launch_kernel: args=%p. Waiting for Python
  // response...\n",
  //        args);
  fd = open(PIPE_NAME, O_RDONLY);
  if (fd < 0) {
    perror("Failed to open FIFO for reading");
    return;
  }
  char buffer[10];
  read(fd, buffer, sizeof(buffer));
  close(fd);

  printf("Python response received. Continuing execution...\n");
}

void tracer::discover_agents() {
  auto agent_callback = [](hsa_agent_t agent, void* data) -> hsa_status_t {
    auto* agents_map =
        static_cast<std::map<hsa_agent_t, std::string, hsa_agent_compare>*>(
            data);

    char name[64] = {0};
    if (hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name) !=
        HSA_STATUS_SUCCESS) {
      return HSA_STATUS_ERROR;
    }
    (*agents_map)[agent] = std::string(name);
    return HSA_STATUS_SUCCESS;
  };

  hsa_core_call(this, hsa_iterate_agents, agent_callback, &agents_names_);
}

std::string demangle_name(const char* mangled_name) {
  int status = 0;
  std::unique_ptr<char, void (*)(void*)> result(
      abi::__cxa_demangle(mangled_name, nullptr, nullptr, &status), std::free);
  return (status == 0) ? result.get() : mangled_name;
}

std::string tracer::get_kernel_name(const std::uint64_t kernel_object) {
  auto handle_find_result = handles_symbols_.find(kernel_object);
  if (handle_find_result == handles_symbols_.end()) {
    return "Object not found.";
  }
  auto symbol_find_result = symbols_names_.find(handle_find_result->second);
  if (handle_find_result == handles_symbols_.end()) {
    return "Symbol not found.";
  }
  return demangle_name(symbol_find_result->second.c_str());
}

std::string tracer::packet_to_text(const hsa_ext_amd_aql_pm4_packet_t* packet) {
  std::ostringstream buff;
  uint32_t type = get_header_type(packet);

  switch (type) {
    case HSA_PACKET_TYPE_VENDOR_SPECIFIC: {
      buff << "HSA_PACKET_TYPE_VENDOR_SPECIFIC";
      break;
    }
    case HSA_PACKET_TYPE_INVALID: {
      buff << "HSA_PACKET_TYPE_INVALID";
      break;
    }
    case HSA_PACKET_TYPE_KERNEL_DISPATCH: {
      const hsa_kernel_dispatch_packet_t* disp =
          reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);
      uint32_t scope = get_header_release_scope(disp);
      const auto kernel_name = get_kernel_name(disp->kernel_object);
      static const char* kernel_to_trace = std::getenv("KERNEL_TO_TRACE");

      if (kernel_name.starts_with(kernel_to_trace)) {
        buff << ("\nTracing the kernel\n");
      }

      buff << "Dispatch Packet\n"
           << fmt::format("\tKernel name: {}\n", kernel_name)
           << "\tRelease Scope: ";

      if (scope & HSA_FENCE_SCOPE_AGENT) {
        buff << "HSA_FENCE_SCOPE_AGENT";
      } else if (scope & HSA_FENCE_SCOPE_SYSTEM) {
        buff << "HSA_FENCE_SCOPE_SYSTEM";
      } else {
        buff << fmt::format("0x{:x}", scope);
      }

      buff << "\n\tAcquire Scope: ";
      scope = get_header_acquire_scope(disp);
      if (scope & HSA_FENCE_SCOPE_AGENT) {
        buff << "HSA_FENCE_SCOPE_AGENT";
      } else if (scope & HSA_FENCE_SCOPE_SYSTEM) {
        buff << "HSA_FENCE_SCOPE_SYSTEM";
      } else {
        buff << "Unkown Scope";
      }

      buff << fmt::format("\n\tsetup: 0x{:x}\n", disp->setup);
      buff << fmt::format("\tworkgroup_size_x: 0x{:x}\n",
                          disp->workgroup_size_x);
      buff << fmt::format("\tworkgroup_size_y: 0x{:x}\n",
                          disp->workgroup_size_y);
      buff << fmt::format("\tworkgroup_size_z: 0x{:x}\n",
                          disp->workgroup_size_z);
      buff << fmt::format("\tgrid_size_x: 0x{:x}\n", disp->grid_size_x);
      buff << fmt::format("\tgrid_size_y: 0x{:x}\n", disp->grid_size_y);
      buff << fmt::format("\tgrid_size_z: 0x{:x}\n", disp->grid_size_z);
      buff << fmt::format("\tprivate_segment_size: 0x{:x}\n",
                          disp->private_segment_size);
      buff << fmt::format("\tgroup_segment_size: 0x{:x}\n",
                          disp->group_segment_size);
      buff << fmt::format("\tkernel_object: 0x{:x}\n", disp->kernel_object);
      buff << fmt::format("\tkernarg_address: 0x{:x}\n",
                          reinterpret_cast<uintptr_t>(disp->kernarg_address));
      buff << fmt::format("\tcompletion_signal: 0x{:x}",
                          disp->completion_signal.handle);
      break;
    }
    case HSA_PACKET_TYPE_BARRIER_AND: {
      buff << "HSA_PACKET_TYPE_BARRIER_AND";
      break;
    }
    case HSA_PACKET_TYPE_AGENT_DISPATCH: {
      buff << "HSA_PACKET_TYPE_AGENT_DISPATCH";
      break;
    }
    case HSA_PACKET_TYPE_BARRIER_OR: {
      buff << "HSA_PACKET_TYPE_BARRIER_OR";
      break;
    }
    default: {
      buff << "Unsupported packet type";
      break;
    }
  }

  return buff.str();
}

std::optional<void*> tracer::is_traceable_packet(
    const hsa_ext_amd_aql_pm4_packet_t* packet) {
  uint32_t type = get_header_type(packet);
  switch (type) {
    case HSA_PACKET_TYPE_KERNEL_DISPATCH: {
      const hsa_kernel_dispatch_packet_t* disp =
          reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packet);
      uint32_t scope = get_header_release_scope(disp);
      const auto kernel_name = get_kernel_name(disp->kernel_object);
      static const char* kernel_to_trace = std::getenv("KERNEL_TO_TRACE");

      if (kernel_name.starts_with(kernel_to_trace)) {
        return disp->kernarg_address;
      }
    }
  }
  return {};
}

tracer* tracer::get_instance(HsaApiTable* table,
                             uint64_t runtime_version,
                             uint64_t failed_tool_count,
                             const char* const* failed_tool_names) {
  const std::lock_guard<std::mutex> lock(mutex_);
  if (!singleton_) {
    if (table != NULL) {
      singleton_ = new tracer(table, runtime_version, failed_tool_count,
                              failed_tool_names);

    } else {
    }
  }
  return singleton_;
}

tracer::~tracer() {
  delete rocr_api_table_.core_;
  delete rocr_api_table_.amd_ext_;
  delete rocr_api_table_.finalizer_ext_;
  delete rocr_api_table_.image_ext_;
}

hsa_status_t tracer::hsa_executable_get_symbol_by_name(
    hsa_executable_t executable,
    const char* symbol_name,
    const hsa_agent_t* agent,
    hsa_executable_symbol_t* symbol) {
  LOG_DETAIL("Looking up the kernel {} \n\t ({})", symbol_name,
             demangle_name(symbol_name));

  auto instance = get_instance();
  auto result = hsa_core_call(instance, hsa_executable_get_symbol_by_name,
                              executable, symbol_name, agent, symbol);

  {
    std::lock_guard g(mutex_);
    const std::string kernel_name = std::string(symbol_name);
    instance->symbols_names_[*symbol] = kernel_name;
    instance->kernels_executables_[kernel_name] = executable;
  }

  return result;
}

hsa_status_t tracer::hsa_executable_symbol_get_info(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute,
    void* value) {
  auto instance = get_instance();
  auto result = hsa_core_call(instance, hsa_executable_symbol_get_info,
                              executable_symbol, attribute, value);

  if (result == HSA_STATUS_SUCCESS &&
      attribute == HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT) {
    LOG_DETAIL("Looking up the symbol 0x{:x}", executable_symbol.handle);

    std::lock_guard g(mutex_);
    instance->handles_symbols_[*static_cast<std::uint64_t*>(value)] =
        executable_symbol;
  }
  return result;
}

void tracer::save_hsa_api() {
  rocr_api_table_.core_ = new CoreApiTable();
  rocr_api_table_.amd_ext_ = new AmdExtTable();
  rocr_api_table_.finalizer_ext_ = new FinalizerExtTable();
  rocr_api_table_.image_ext_ = new ImageExtTable();

  std::memcpy(rocr_api_table_.core_, api_table_->core_, sizeof(CoreApiTable));
  std::memcpy(rocr_api_table_.amd_ext_, api_table_->amd_ext_,
              sizeof(AmdExtTable));
  std::memcpy(rocr_api_table_.finalizer_ext_, api_table_->finalizer_ext_,
              sizeof(FinalizerExtTable));
  std::memcpy(rocr_api_table_.image_ext_, api_table_->image_ext_,
              sizeof(ImageExtTable));
}
void tracer::restore_hsa_api() {
  copyTables(&rocr_api_table_, api_table_);
}
void tracer::hook_api() {
  api_table_->core_->hsa_queue_create_fn = tracer::hsa_queue_create;
  api_table_->core_->hsa_queue_destroy_fn = tracer::hsa_queue_destroy;

  api_table_->amd_ext_->hsa_amd_memory_pool_allocate_fn =
      tracer::hsa_amd_memory_pool_allocate;
  api_table_->core_->hsa_memory_allocate_fn = tracer::hsa_memory_allocate;

  api_table_->core_->hsa_executable_get_symbol_by_name_fn =
      tracer::hsa_executable_get_symbol_by_name;

  api_table_->core_->hsa_executable_symbol_get_info_fn =
      tracer::hsa_executable_symbol_get_info;
}

hsa_status_t tracer::add_queue(hsa_queue_t* queue, hsa_agent_t agent) {
  std::lock_guard<std::mutex> lock(mm_mutex_);
  auto instance = get_instance();
  auto result = hsa_ext_call(instance, hsa_amd_profiling_set_profiler_enabled,
                             queue, true);
  return result;
}

void tracer::on_submit_packet(const void* in_packets,
                              uint64_t count,
                              uint64_t user_que_idx,
                              void* data,
                              hsa_amd_queue_intercept_packet_writer writer) {
  auto instance = get_instance();
  if (instance) {
    hsa_queue_t* queue = reinterpret_cast<hsa_queue_t*>(data);
    instance->write_packets(
        queue, static_cast<const hsa_ext_amd_aql_pm4_packet_t*>(in_packets),
        count, writer);

    // LOG_DETAIL("Enqueue {} packet(s)", count);
  }
}

void tracer::write_packets(hsa_queue_t* queue,
                           const hsa_ext_amd_aql_pm4_packet_t* packet,
                           uint64_t count,
                           hsa_amd_queue_intercept_packet_writer writer) {
  try {
    // LOG_DETAIL("Executing packet: {}", packet_to_text(packet));
    auto instance = get_instance();

    hsa_signal_t new_signal;
    auto status =
        hsa_core_call(instance, hsa_signal_create, 1, 0, nullptr, &new_signal);

    if (status != HSA_STATUS_SUCCESS) {
      std::cerr << "Failed to create signal\n";
      return;
    }

    // TODO: Did we lose some concurrency here or something?
    hsa_ext_amd_aql_pm4_packet_t modified_packet = *packet;
    hsa_signal_t old_signal = modified_packet.completion_signal;
    modified_packet.completion_signal = new_signal;

    // TODO: I see stale data in output
    modified_packet.header |= (1 << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);

    writer(&modified_packet, count);

    hsa_signal_wait_scacquire(new_signal, HSA_SIGNAL_CONDITION_EQ, 0,
                              UINT64_MAX, HSA_WAIT_STATE_ACTIVE);
    // while (hsa_signal_load_relaxed(new_signal) != 0) {
    // }
    if (old_signal.handle != 0) {
      hsa_core_call(instance, hsa_signal_subtract_relaxed, old_signal, 1);
    }
    hsa_core_call(instance, hsa_signal_destroy, new_signal);

    auto args = is_traceable_packet(packet);
    if (args.has_value()) {
      send_message_and_wait(args.value());
    }
  } catch (const std::exception& e) {
    LOG_ERROR("Write object threw ", e.what());
  }
}

hsa_status_t tracer::hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t pool,
                                                  size_t size,
                                                  uint32_t flags,
                                                  void** ptr) {
  auto instance = get_instance();
  const auto result = hsa_ext_call(instance, hsa_amd_memory_pool_allocate, pool,
                                   size, flags, ptr);
  if (result == HSA_STATUS_SUCCESS && *ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    instance->pointer_sizes_[*ptr] = size;
    std::cout << "[Maestro Tracer] HSA Allocated " << size << " bytes at "
              << *ptr << std::endl;
  }
  return result;
}
hsa_status_t tracer::hsa_memory_allocate(hsa_region_t region,
                                         size_t size,
                                         void** ptr) {
  auto instance = get_instance();
  const auto result =
      hsa_core_call(instance, hsa_memory_allocate, region, size, ptr);
  if (result == HSA_STATUS_SUCCESS && *ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    instance->pointer_sizes_[*ptr] = size;
    std::cout << "[Maestro Tracer] HSA Allocated " << size << " bytes at "
              << *ptr << std::endl;
  }
  return result;
}

hsa_status_t tracer::hsa_queue_create(hsa_agent_t agent,
                                      uint32_t size,
                                      hsa_queue_type32_t type,
                                      void (*callback)(hsa_status_t status,
                                                       hsa_queue_t* source,
                                                       void* data),
                                      void* data,
                                      uint32_t private_segment_size,
                                      uint32_t group_segment_size,
                                      hsa_queue_t** queue) {
  LOG_INFO("Creating maestro-rt queue");

  hsa_status_t result = HSA_STATUS_SUCCESS;
  auto instance = get_instance();
  try {
    result = hsa_ext_call(instance, hsa_amd_queue_intercept_create, agent, size,
                          type, callback, data, private_segment_size,
                          group_segment_size, queue);

    if (result == HSA_STATUS_SUCCESS) {
      auto result = instance->add_queue(*queue, agent);
      if (result != HSA_STATUS_SUCCESS) {
        LOG_ERROR("Failed to add queue {} ", static_cast<int>(result));
      }
      result = hsa_ext_call(instance, hsa_amd_queue_intercept_register, *queue,
                            tracer::on_submit_packet,
                            reinterpret_cast<void*>(*queue));
      if (result != HSA_STATUS_SUCCESS) {
        LOG_ERROR("Failed to register intercept callback with result of ",
                  static_cast<int>(result));
      }
    }
  } catch (const std::exception& e) {
    LOG_ERROR("Interception queue create throw {} error", e.what());
  }
  return result;
}
hsa_status_t tracer::hsa_queue_destroy(hsa_queue_t* queue) {
  LOG_INFO("Destroying maestro-rt queue");
  return hsa_core_call(singleton_, hsa_queue_destroy, queue);
}
}  // namespace maestro

extern "C" {

PUBLIC_API bool OnLoad(HsaApiTable* table,
                       uint64_t runtime_version,
                       uint64_t failed_tool_count,
                       const char* const* failed_tool_names) {
  LOG_INFO("Creating maestro-rt singleton");

  // maestro::detail::print_env_variables();

  maestro::tracer* hook = maestro::tracer::get_instance(
      table, runtime_version, failed_tool_count, failed_tool_names);

  LOG_INFO("Creating maestro-rt singleton completed");

  return true;
}

PUBLIC_API void OnUnload() {}

static void unload_me() __attribute__((destructor));
void unload_me() {}
}
