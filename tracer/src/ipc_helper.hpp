#ifndef IPC_HELPER_HPP
#define IPC_HELPER_HPP

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cstring>
#include <iostream>

#define SHM_NAME "/hip_ipc_handles"  // Shared memory name
#define MAX_HANDLES 128              // Max number of IPC handles to store

struct SharedMemoryBuffer {
  int buffer_size;                         // Max handles we can store
  int head_idx;                            // Next insertion index
  hipIpcMemHandle_t handles[MAX_HANDLES];  // Circular buffer
};

inline void send_ipc_bytes(const hipIpcMemHandle_t& handle) {
  int shm_fd = shm_open(SHM_NAME, O_CREAT | O_RDWR, 0666);
  if (shm_fd == -1) {
    std::cerr << "[IPC Helper] Failed to create/open shared memory\n";
    return;
  }

  // Resize shared memory to fit the circular buffer
  if (ftruncate(shm_fd, sizeof(SharedMemoryBuffer)) == -1) {
    std::cerr << "[IPC Helper] Failed to resize shared memory\n";
    close(shm_fd);
    return;
  }

  // Map the shared memory
  void* mapped_mem = mmap(0, sizeof(SharedMemoryBuffer), PROT_READ | PROT_WRITE,
                          MAP_SHARED, shm_fd, 0);
  if (mapped_mem == MAP_FAILED) {
    std::cerr << "[IPC Helper] Failed to map shared memory\n";
    close(shm_fd);
    return;
  }

  // Cast to the shared buffer struct
  auto* buffer = static_cast<SharedMemoryBuffer*>(mapped_mem);

  // Initialize buffer if needed
  if (buffer->buffer_size != MAX_HANDLES) {
    buffer->buffer_size = MAX_HANDLES;
    buffer->head_idx = 0;
  }

  // Store the IPC handle at the next available index
  int index = buffer->head_idx % MAX_HANDLES;
  buffer->handles[index] = handle;

  // Update the head index
  buffer->head_idx = (buffer->head_idx + 1) % MAX_HANDLES;

  // Unmap and close shared memory
  munmap(mapped_mem, sizeof(SharedMemoryBuffer));
  close(shm_fd);
}

#endif  // IPC_HELPER_HPP