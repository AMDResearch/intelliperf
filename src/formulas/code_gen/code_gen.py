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


import os

def generate_recovered_kernel(kernel_source: str, kernel_name: str, args: list[str]) -> str:
    header_path = "/tmp/recovered_kernel.hip"

    cur_file_dir = os.path.dirname(os.path.abspath(__file__))
    vmem_allocator_path = os.path.join(cur_file_dir, "..", "..", "csrc", "vmem_allocator.hpp")

    # Generate the argument loading lines
    arg_loading_lines = ""
    for i, arg in enumerate(args):
        is_pointer = '*' in arg
        
        while 'const' in arg:
            arg = arg.replace('const', '')
        while '*' in arg:
            arg = arg.replace('*', '')
        arg = ' '.join(arg.split())
        
        argname = f"arg_{i}"
        arg_loading_lines += f"\tauto h_{argname} = load_arg<{arg}>(argv[1], {i});\n"
        if is_pointer:
            arg_loading_lines += f"\tauto d_{argname} = thrust::device_vector<{arg}>(h_{argname});\n"
        else:
            arg_loading_lines += f"\tauto d_{argname} = h_{argname}[0];\n"
        

    block_size = 1024
    num_blocks = 8

    
    kernel_launch_lines = f"\t{kernel_name}<<<{num_blocks}, {block_size}>>>("
    for i, arg in enumerate(args):
        is_pointer = '*' in arg
        argname = f"d_arg_{i}"
        
        if is_pointer:
            kernel_launch_lines += f"{argname}.data().get()"
        else:
            kernel_launch_lines += f"{argname}"
        
        if i == len(args) - 1:
            kernel_launch_lines += f")"
        else:
            kernel_launch_lines += f", "


    header_content = f"""\

#include "{vmem_allocator_path}"
#include <filesystem>
#include <fstream>
#include <vector>
#include <cstdio>
#include<thrust/device_vector.h>

{kernel_source}

template <typename T>
std::vector<T> load_arg(const char* path, int index) {{
  std::ifstream file(path, std::ios::binary);
  if (!file) {{
    throw std::runtime_error(std::string("Failed to open file: ") + path);
  }}

  const std::string begin_marker = "BEGIN";
  const std::string end_marker = "END";

  std::string line;
  std::vector<T> results;
  int current_index = 0;

  while (std::getline(file, line)) {{
    if (line == begin_marker) {{    
      std::size_t ptr_size = 0;
      file.read(reinterpret_cast<char*>(&ptr_size), sizeof(std::size_t));

      if (ptr_size % sizeof(T) != 0) {{
        throw std::runtime_error("ptr_size is not a multiple of sizeof(T)");
      }}

      std::size_t num_elements = ptr_size / sizeof(T);
      std::vector<T> temp(num_elements);
      file.read(reinterpret_cast<char*>(temp.data()), ptr_size);

      std::getline(file, line);
      if (line != end_marker) {{
        throw std::runtime_error("Expected END marker after value block");
      }}

      if (current_index == index) {{
        return temp;
      }}

      ++current_index;
    }}
  }}

  if (index != -1 && index >= current_index) {{
    throw std::out_of_range("Requested index exceeds number of values in file");
  }}

  return results;
}}


int main(int argc, char** argv) {{
    if (argc != 2) {{
        printf("Usage: %s <path-to-args-to-load>\\n", argv[0]);
        return 1;
    }}

{arg_loading_lines}

{kernel_launch_lines}

    return hipDeviceSynchronize();
}}
"""

    with open(header_path, "w") as header_file:
        header_file.write(header_content)
    
    return header_path
