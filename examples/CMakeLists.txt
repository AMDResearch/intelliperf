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

cmake_minimum_required(VERSION 3.10)
project(Examples LANGUAGES HIP CXX)

option(INSTRUMENT "Instrument the code with Omniprobe" OFF)

function(add_example name source_file)
    add_executable(${name} ${source_file})
    # CMake instrumentation integration example
    if(INSTRUMENT)
        set(OMNIPROBE_PATH ${CMAKE_SOURCE_DIR}/../external/omniprobe/install)
        # Only use omniprobe if the path exists
        if(EXISTS ${OMNIPROBE_PATH}/lib/libAMDGCNSubmitAddressMessages-rocm.so)
            # Use the plugin to instrument the code
            target_compile_options(${name} PRIVATE 
                -fpass-plugin=${OMNIPROBE_PATH}/lib/libAMDGCNSubmitAddressMessages-rocm.so
            )
            message(STATUS "Using Omniprobe instrumentation for ${name}")
        else()
            message(WARNING "Omniprobe not found at ${OMNIPROBE_PATH}, skipping instrumentation for ${name}")
        endif()
        # Must include debug information to get line numbers
    endif()
    target_compile_options(${name} PRIVATE -g)    
endfunction()

add_subdirectory(bank_conflict)
add_subdirectory(basic)
add_subdirectory(contention)
add_subdirectory(access_pattern)
