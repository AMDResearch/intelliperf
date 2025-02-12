include_guard()

include(${PROJECT_SOURCE_DIR}/cmake/CPM.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/RyzenCompilerOptions.cmake)

CPMAddPackage("gh:fmtlib/fmt#7.1.3")
CPMAddPackage(
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 2.2.1
  OPTIONS "CXXOPTS_BUILD_EXAMPLES NO" "CXXOPTS_BUILD_TESTS NO" "CXXOPTS_ENABLE_INSTALL YES"
)

function(ryzen_add_example TARGET)
    add_executable(${TARGET} ${ARGN})
    target_link_libraries(${TARGET} PRIVATE ryzendf::ryzendf
                                    PRIVATE fmt::fmt
                                    PRIVATE cxxopts)
    ryzen_compiler_warnings(${TARGET})
    ryzen_compiler_options(${TARGET})
endfunction()
