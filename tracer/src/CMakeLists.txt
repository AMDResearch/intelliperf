# SPDX-License-Identifier: MIT
### Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved. ###

include(${PROJECT_SOURCE_DIR}/cmake/RyzenCompilerOptions.cmake)
message("PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
include(FetchContent)

#
# dependencies
#

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG        v1.11.0
)
FetchContent_MakeAvailable(spdlog)

include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
CPMAddPackage("gh:fmtlib/fmt#7.1.3")


find_package(HSA REQUIRED)

#
# tracer target
#



set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")
set(CMAKE_HIP_FLAGS_DEBUG "-O0 -ggdb")


target_include_directories(tracer
    PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
        "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/tracer>"
)


target_sources(tracer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/log.hpp>
        # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/status.hpp>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tracer.hpp>
        # $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/npu_matmul_func.h>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/log.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/status.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tracer.cpp
        # ${CMAKE_CURRENT_SOURCE_DIR}/npu_matmul_func.cpp
)

ryzen_compiler_warnings(tracer)
ryzen_compiler_options(tracer)

target_link_libraries(tracer
    PUBLIC
        spdlog::spdlog
        fmt::fmt
        hsa::hsa
    PRIVATE
)
