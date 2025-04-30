################################################################################
# MIT License

# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################

from agents.agents import (
    compiler_agent,
    correctness_agent,
    performance_agent,
    optimizer_agent,
)
import logging

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

initial_code = """
#include <thrust/device_vector.h>

#include <cstdint>
#include <iostream>
 
 
__global__ void reduction_kernel(double* input, double* result, std::size_t count) {
  const auto thread_id = threadIdx.x + blockIdx.x * blockDim.x;
  if (thread_id < count) {
    const auto value = input[thread_id];
    atomicAdd(result, value);
  }
}

int main(){
  const std::size_t count{10000000};
  thrust::device_vector<double> input(count, 1);
  thrust::device_vector<double> result(1, 0);

  const std::size_t block_size{512};
  const std::size_t num_blocks{(count + block_size - 1) / block_size};

  reduction_kernel<<<num_blocks, block_size, block_size * sizeof(double)>>>(
      input.data().get(), result.data().get(), count);

  const auto status = hipDeviceSynchronize();
  if (status != hipSuccess || result[0] != count) {
    std::cout << \"Kernel failed.\";
    return -1;
  } else {
    std::cout << \"Success!\";
  }
  std::cout << std::endl;
  return static_cast<int>(result[0]);
}
"""

success, initial_binary = compiler_agent(code=initial_code)
if not success:
    print(f"Initial code compilation failed with message:\n{initial_binary}")
    exit(1)
logging.debug(f"Initial binary is: {initial_binary}")


done = False

while not done:
    prompt = f"Optimize the code: \n {initial_code}"
    # LLM Request
    success, message = optimizer_agent(prompt)
    if not success:
        logging.warning(f"Optimizer failed with message {message}")
        continue
    optimized_code = message

    logging.debug(f"Optimized code is: \n{optimized_code}")

    # Compiler request
    success, message = compiler_agent(code=optimized_code)
    if not success:
        logging.warning(f"Compiler failed with message {message}")
        continue
    optimized_binary = message
    logging.debug(f"Optimized binary is: {optimized_binary}")

    # Correctness
    success, message = correctness_agent(
        reference=initial_binary, updated=optimized_binary
    )
    if not success:
        logging.warning(f"Optimizer failed with message {message}")
        continue
    logging.debug(f"Optimizer result is: {message}")

    # Performance
    success, message = performance_agent(
        reference=initial_binary, updated=optimized_binary
    )
    if not success:
        logging.warning(f"Optimizer failed with message {message}")
        continue
    speedup = message
    logging.debug(f"Optimizer result is: {speedup}")

    # Done
    logging.info(f"Initial code: \n{initial_code}\n=================================")
    logging.info(
        f"Optimized code: \n{optimized_code}\================================="
    )
    logging.info(f"Speedup: {speedup}")

    with open("initial_code.hip", "w") as file:
        file.write(initial_code)
    with open("optimized_code.hip", "w") as file:
        file.write(optimized_code)
    break
