cmake_minimum_required(VERSION 3.21)
project(atomic_reduction LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(atomic_reduction_bench main.hip kernel.hip)

target_compile_options(atomic_reduction_bench PRIVATE -O3 -g)

set_target_properties(atomic_reduction_bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)




