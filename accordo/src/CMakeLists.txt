# SPDX-License-Identifier: MIT
### Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved. ###

include(${PROJECT_SOURCE_DIR}/cmake/MaestroCompilerOptions.cmake)
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
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tracer.hpp>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/tracer.cpp
)

maestro_compiler_warnings(tracer)
maestro_compiler_options(tracer)

target_link_libraries(tracer
    PUBLIC
        spdlog::spdlog
        fmt::fmt
        hsa::hsa
    PRIVATE
)
