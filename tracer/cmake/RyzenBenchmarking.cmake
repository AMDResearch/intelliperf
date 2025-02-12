include_guard()

include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/RyzenCompilerOptions.cmake)

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  VERSION 1.9.0
  OPTIONS "BENCHMARK_ENABLE_TESTING Off"
)


function(ryzen_add_benchmark TARGET)
    add_executable(${TARGET} ${ARGN})
    target_link_libraries(${TARGET} PRIVATE ryzendf::ryzendf
                                    PRIVATE benchmark::benchmark)
    target_include_directories(${TARGET} PRIVATE ${benchmark_SOURCE_DIR}/include)
    ryzen_compiler_warnings(${TARGET})
    ryzen_compiler_options(${TARGET})
endfunction()
